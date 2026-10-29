from __future__ import annotations

import logging
import re
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Iterable

import feedparser
import requests
from bs4 import BeautifulSoup

from .fetch_support import (
    DEFAULT_TIMEOUT,
    HEADERS,
    build_source_health_entry,
    is_google_news_wrapper,
    load_source_item_cache_payload,
    ncbi_get,
    resilient_get,
    resolve_google_news_response,
    save_source_item_cache,
    stamp_items_from_source,
)
from .parsers import (
    clean_extracted_text,
    extract_fda_outbreak_markdown,
    extract_fda_outbreak_table,
    extract_html_links,
    extract_ncdc_news_rows,
    extract_page_text,
    extract_publication_date,
    parse_pubmed_details_xml,
)
from .utils import (
    Item,
    SourceConfig,
    clean_headline,
    normalize_source_name,
    normalize_whitespace,
    parse_datetime,
    strip_html,
)


def fetch_all_sources(
    sources: Iterable[SourceConfig],
    logger: logging.Logger,
    start_date: date,
    end_date: date,
) -> tuple[list[Item], list[dict[str, object]], list[dict[str, object]]]:
    items: list[Item] = []
    failures: list[dict[str, object]] = []
    source_health: list[dict[str, object]] = []
    for source in sources:
        refresh_hours = source.refresh_hours
        fallback_cache_hours = source.fallback_cache_hours or 48
        if refresh_hours:
            cache_payload = load_source_item_cache_payload(source, max_age_hours=refresh_hours)
            if cache_payload is not None:
                cached_items, cached_at, cache_age_hours = cache_payload
                logger.info("Using source item cache for %s due to refresh window (%sh)", source.name, refresh_hours)
                _append_source_result(
                    items,
                    source_health,
                    records=cached_items,
                    source=source,
                    mode="refresh_cache",
                    cached_at=cached_at,
                    cache_age_hours=cache_age_hours,
                )
                continue
        try:
            fetched_items = fetch_source(source, logger, start_date=start_date, end_date=end_date)
            if fetched_items:
                save_source_item_cache(source, fetched_items)
            fetched_at = datetime.now(UTC)
            _append_source_result(
                items,
                source_health,
                records=fetched_items,
                source=source,
                mode="live",
                cached_at=fetched_at,
                cache_age_hours=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            cache_payload = load_source_item_cache_payload(source, max_age_hours=fallback_cache_hours)
            if cache_payload is not None:
                cached_items, cached_at, cache_age_hours = cache_payload
                logger.exception("Source failed, using source item cache: %s (%s)", source.name, exc)
                _append_source_result(
                    items,
                    source_health,
                    records=cached_items,
                    source=source,
                    mode="fallback_cache",
                    cached_at=cached_at,
                    cache_age_hours=cache_age_hours,
                    error=str(exc),
                )
                failures.append(_source_failure(source, f"{exc} [cache fallback used]"))
                continue
            logger.exception("Source failed: %s (%s)", source.name, exc)
            failures.append(_source_failure(source, str(exc)))
            source_health.append(build_source_health_entry(source, mode="failed", item_count=0, error=str(exc)))
    return items, failures, source_health


def _append_source_result(
    all_items: list[Item],
    source_health: list[dict[str, object]],
    *,
    records: list[Item],
    source: SourceConfig,
    mode: str,
    cached_at: datetime,
    cache_age_hours: float,
    error: str | None = None,
) -> None:
    stamp_items_from_source(
        records,
        source=source,
        freshness_state=mode,
        source_cached_at=cached_at,
        source_cache_age_hours=cache_age_hours,
    )
    all_items.extend(records)
    source_health.append(
        build_source_health_entry(
            source,
            mode=mode,
            item_count=len(records),
            cached_at=cached_at,
            cache_age_hours=cache_age_hours,
            error=error,
        )
    )


def _source_failure(source: SourceConfig, error: str) -> dict[str, object]:
    return {
        "source": source.name,
        "source_type": source.type,
        "required": source.required,
        "error": error,
    }


def fetch_source(source: SourceConfig, logger: logging.Logger, start_date: date, end_date: date) -> list[Item]:
    fetcher_map = {
        "rss": fetch_rss,
        "html_list": fetch_html_list,
        "html_page": fetch_html_page,
        "pubmed": fetch_pubmed,
        "medrxiv": fetch_medrxiv_like,
        "biorxiv": fetch_medrxiv_like,
    }
    try:
        fetcher = fetcher_map[source.type]
    except KeyError as exc:
        raise ValueError(f"Unsupported health source type: {source.type}") from exc
    return fetcher(source, logger, start_date=start_date, end_date=end_date)


def fetch_rss(source: SourceConfig, logger: logging.Logger, start_date: date | None = None, end_date: date | None = None) -> list[Item]:
    response = resilient_get(
        source.url,
        headers=HEADERS,
        timeout=source.timeout_seconds or DEFAULT_TIMEOUT,
        logger=logger,
        cache_namespace="rss",
        max_attempts=source.max_attempts or 3,
        verify_ssl=source.verify_ssl,
    )
    response.raise_for_status()
    parsed = feedparser.parse(response.text)
    items: list[Item] = []
    for entry in parsed.entries[: source.max_items or None]:
        publisher = extract_entry_publisher(entry)
        items.append(
            Item(
                title=clean_headline(entry.get("title", "")),
                source=source.name,
                url=entry.get("link", ""),
                category=source.category,
                publisher=publisher,
                published_at=parse_datetime(entry.get("published") or entry.get("updated")),
                summary=clean_extracted_text(strip_html(entry.get("summary", ""))),
                source_type=source.type,
                official=source.official,
                metadata={
                    "feed_id": entry.get("id"),
                    "publisher": publisher,
                    "aggregator_source": source.name,
                    "raw_url": entry.get("link", ""),
                },
            )
        )
    logger.info("Fetched %s RSS items from %s", len(items), source.name)
    return [item for item in items if item.title and item.url]


def fetch_html_list(
    source: SourceConfig,
    logger: logging.Logger,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[Item]:
    if source.name == "FDA Foodborne Outbreaks":
        try:
            response = resilient_get(
                source.url,
                headers=BROWSERISH_HEADERS,
                timeout=source.timeout_seconds or DEFAULT_TIMEOUT,
                logger=logger,
                cache_namespace="html",
                max_attempts=source.max_attempts or 3,
                verify_ssl=source.verify_ssl,
            )
            response.raise_for_status()
            return fetch_fda_outbreak_rows(source, response.text, logger)
        except requests.RequestException as exc:
            logger.warning("FDA direct collection failed (%s); using reader fallback.", exc)
            return fetch_fda_outbreak_reader_rows(source, logger)

    response = resilient_get(
        source.url,
        headers=BROWSERISH_HEADERS,
        timeout=source.timeout_seconds or DEFAULT_TIMEOUT,
        logger=logger,
        cache_namespace="html",
        max_attempts=source.max_attempts or 3,
        verify_ssl=source.verify_ssl,
    )
    response.raise_for_status()
    if source.name == "Nigeria Centre for Disease Control":
        return fetch_ncdc_news_rows(source, response.text, logger)
    items: list[Item] = []
    for entry in extract_html_links(response.text, source.url, source.item_selector or "a")[: source.max_items or None]:
        title = clean_headline(entry["title"])
        if low_signal_title(title):
            continue
        items.append(
            Item(
                title=title,
                source=source.name,
                url=entry["url"],
                category=source.category,
                published_at=extract_display_date(title),
                source_type=source.type,
                official=source.official,
            )
        )
    logger.info("Fetched %s HTML list items from %s", len(items), source.name)
    return items


def fetch_html_page(
    source: SourceConfig,
    logger: logging.Logger,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[Item]:
    """Capture a dated official situation page as one monitor record."""
    response = resilient_get(
        source.url,
        headers=BROWSERISH_HEADERS,
        timeout=source.timeout_seconds or DEFAULT_TIMEOUT,
        logger=logger,
        cache_namespace="html",
        max_attempts=source.max_attempts or 3,
        verify_ssl=source.verify_ssl,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.select_one(source.item_selector) if source.item_selector else soup.body
    page_text = normalize_whitespace((content or soup).get_text(" ", strip=True))
    if not page_text:
        return []
    item = Item(
        title=source.name,
        source=source.name,
        url=source.url or "",
        category=source.category,
        published_at=extract_display_date(page_text),
        summary=build_html_page_summary(page_text),
        source_type=source.type,
        official=source.official,
        metadata={"preserve_source_summary": True},
    )
    logger.info("Fetched official situation page from %s", source.name)
    return [item]


def extract_display_date(text: str) -> datetime | None:
    """Extract a displayed numeric date from a government update caption or page."""
    match = re.search(r"\blast\s+update\s*:\s*(\d{1,2}/\d{1,2}/\d{2,4})", text, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"\((\d{1,2}/\d{1,2}/\d{2,4})\)", text)
    return parse_datetime(match.group(1)) if match else None


def build_html_page_summary(page_text: str) -> str:
    """Keep displayed official situation-page counts visible in the monitor card."""
    count_pattern = re.compile(
        r"\b(?:[A-Z][A-Za-z-]*\s+){0,4}(?:positive|confirmed|reported|suspected)\s+"
        r"(?:cases?|deaths?|hospitalizations?)\s*:\s*[\d,]+",
        flags=re.IGNORECASE,
    )
    counts = [normalize_whitespace(match.group(0)) for match in count_pattern.finditer(page_text)]
    if not counts:
        return page_text[:12000]

    summary = "; ".join(dict.fromkeys(counts)) + "."
    if "preliminary" in page_text.lower():
        summary += " The health department labels these counts preliminary."
    update_match = re.search(r"\blast\s+update\s*:\s*(\d{1,2}/\d{1,2}/\d{2,4})", page_text, flags=re.IGNORECASE)
    if update_match:
        summary += f" Last update: {update_match.group(1)}."
    return summary


def fetch_fda_outbreak_rows(source: SourceConfig, html: str, logger: logging.Logger) -> list[Item]:
    rows = extract_fda_outbreak_table(html, source.url or "", max_items=source.max_items)
    items = [
        Item(
            title=row["title"],
            source=source.name,
            url=row["url"],
            category=source.category,
            published_at=row["published_at"],
            summary=row["summary"],
            source_type=source.type,
            official=source.official,
            metadata=row["metadata"],
        )
        for row in rows
    ]
    logger.info("Fetched %s FDA outbreak rows from %s", len(items), source.name)
    return items


def fetch_fda_outbreak_reader_rows(source: SourceConfig, logger: logging.Logger) -> list[Item]:
    reader_url = f"https://r.jina.ai/http://{source.url}"
    response = resilient_get(
        reader_url,
        headers=HEADERS,
        timeout=source.timeout_seconds or DEFAULT_TIMEOUT,
        logger=logger,
        cache_namespace="fda_reader",
        max_attempts=2,
        verify_ssl=True,
    )
    response.raise_for_status()
    rows = extract_fda_outbreak_markdown(response.text, source.url or "", max_items=source.max_items)
    items = [
        Item(
            title=row["title"],
            source=source.name,
            url=row["url"],
            category=source.category,
            published_at=row["published_at"],
            summary=row["summary"],
            source_type=source.type,
            official=source.official,
            metadata=row["metadata"],
        )
        for row in rows
    ]
    logger.info("Fetched %s FDA outbreak rows through the reader fallback", len(items))
    return items


def fetch_ncdc_news_rows(source: SourceConfig, html: str, logger: logging.Logger) -> list[Item]:
    rows = extract_ncdc_news_rows(html, source.url or "", max_items=source.max_items)
    items = [
        Item(
            title=row["title"],
            source=source.name,
            url=row["url"],
            category=source.category,
            published_at=row["published_at"],
            summary=row["summary"],
            source_type=source.type,
            official=source.official,
        )
        for row in rows
    ]
    logger.info("Fetched %s NCDC news items from %s", len(items), source.name)
    return items


def low_signal_title(title: str) -> bool:
    normalized = normalize_whitespace(title).strip(":- ").lower()
    if len(normalized) < 8:
        return True
    return normalized in {
        "advisory",
        "see",
        "see advisory",
        "read more",
        "learn more",
        "more",
        "click here",
        "details",
    }


def extract_entry_publisher(entry) -> str | None:
    source_info = entry.get("source")
    if isinstance(source_info, dict):
        publisher = normalize_source_name(source_info.get("title"))
        if publisher:
            return publisher
    publisher = normalize_source_name(entry.get("source_title"))
    return publisher or None


def fetch_pubmed(source: SourceConfig, logger: logging.Logger, start_date: date, end_date: date) -> list[Item]:
    search_params = {
        "db": "pubmed",
        "retmode": "json",
        "retmax": source.max_items or 20,
        "sort": "pub date",
        "term": source.search,
        "datetype": "pdat",
        "mindate": start_date.isoformat(),
        "maxdate": end_date.isoformat(),
    }
    search_response = ncbi_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params=search_params,
        headers=HEADERS,
        timeout=source.timeout_seconds or DEFAULT_TIMEOUT,
        logger=logger,
        cache_namespace="pubmed_esearch",
        max_attempts=source.max_attempts or 3,
    )
    search_response.raise_for_status()
    id_list = search_response.json().get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return []

    summary_response = ncbi_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db": "pubmed", "retmode": "json", "id": ",".join(id_list)},
        headers=HEADERS,
        timeout=source.timeout_seconds or DEFAULT_TIMEOUT,
        logger=logger,
        cache_namespace="pubmed_esummary",
        max_attempts=source.max_attempts or 3,
    )
    summary_response.raise_for_status()
    result = summary_response.json().get("result", {})
    detail_response = ncbi_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"},
        headers=HEADERS,
        timeout=source.timeout_seconds or DEFAULT_TIMEOUT,
        logger=logger,
        cache_namespace="pubmed_efetch",
        max_attempts=source.max_attempts or 3,
    )
    detail_response.raise_for_status()
    details = parse_pubmed_details_xml(detail_response.text)
    items: list[Item] = []
    for pmid in id_list:
        entry = result.get(pmid, {})
        detail = details.get(pmid, {})
        title = clean_headline(entry.get("title", ""))
        if not title:
            continue
        article_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        authors = entry.get("authors", [])
        if authors and isinstance(authors[0], dict):
            author_text = ", ".join(author.get("name", "") for author in authors[:3] if author.get("name"))
        else:
            author_text = ", ".join(str(author) for author in authors[:3])
        summary_text = detail.get("abstract") or detail.get("fallback_summary") or normalize_whitespace(author_text)
        metadata = {
            "pmid": pmid,
            "publication_types": detail.get("publication_types", []),
            "publication_status": detail.get("publication_status"),
        }
        items.append(
            Item(
                title=title,
                source=source.name,
                url=article_url,
                category=source.category,
                published_at=detail.get("published_at") or parse_datetime(entry.get("pubdate")),
                summary=summary_text,
                doi=detail.get("doi"),
                journal=detail.get("journal") or entry.get("fulljournalname") or entry.get("source"),
                source_type=source.type,
                official=source.official,
                extracted_text=detail.get("abstract", ""),
                metadata=metadata,
            )
        )
    logger.info("Fetched %s PubMed items", len(items))
    return items


def fetch_medrxiv_like(source: SourceConfig, logger: logging.Logger, start_date: date, end_date: date) -> list[Item]:
    server = "medrxiv" if source.type == "medrxiv" else "biorxiv"
    api_start = max(start_date - timedelta(days=2), date(2020, 1, 1))
    response = resilient_get(
        f"https://api.biorxiv.org/details/{server}/{api_start.isoformat()}/{end_date.isoformat()}",
        headers=HEADERS,
        timeout=source.timeout_seconds or DEFAULT_TIMEOUT,
        logger=logger,
        cache_namespace=f"{server}_details",
        max_attempts=source.max_attempts or 3,
    )
    response.raise_for_status()
    payload = response.json()
    collection = payload.get("collection", [])
    items: list[Item] = []
    terms = [term.strip().lower() for term in (source.search or "").replace("OR", ",").split(",") if term.strip()]
    for entry in collection:
        title = clean_headline(entry.get("title", ""))
        abstract = clean_extracted_text(strip_html(entry.get("abstract", "")))
        haystack = f"{title} {abstract}".lower()
        if terms and not any(term in haystack for term in terms):
            continue
        published_at = parse_datetime(entry.get("date"))
        if published_at and not (start_date <= published_at.date() <= end_date):
            continue
        doi = entry.get("doi")
        items.append(
            Item(
                title=title,
                source=source.name,
                url=f"https://www.{server}.org/content/{doi}v{entry.get('version', '1')}" if doi else "",
                category=source.category,
                published_at=published_at,
                summary=abstract,
                doi=doi,
                journal=server,
                abstract_url=f"https://www.{server}.org/content/{doi}.abstract" if doi else None,
                source_type=source.type,
                official=source.official,
            )
        )
        if len(items) >= (source.max_items or 100):
            break
    logger.info("Fetched %s %s items", len(items), source.name)
    return [item for item in items if item.title and item.url]


def enrich_item_text(item: Item, logger: logging.Logger) -> Item:
    response = None
    if is_google_news_wrapper(item.url):
        response = resolve_google_news_response(item, logger)
        if response is None:
            return item
    if item.source_type in {"pubmed", "medrxiv", "biorxiv"} and item.summary:
        item.extracted_text = item.summary
        return item
    if response is None and item.summary and len(item.summary) > 220:
        item.extracted_text = item.summary
        return item
    if response is None:
        try:
            response = resilient_get(
                item.url,
                headers=HEADERS,
                timeout=DEFAULT_TIMEOUT,
                logger=logger,
                cache_namespace="article",
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.info("Skipping content enrichment for %s: %s", item.url, exc)
            return item

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type:
        return item
    item.extracted_text = extract_page_text(response.text).strip()
    last_modified = response.headers.get("last-modified")
    if last_modified:
        try:
            item.metadata["source_last_modified_at"] = parsedate_to_datetime(last_modified).isoformat()
        except (TypeError, ValueError, OverflowError):
            pass
    if not item.published_at:
        item.published_at = extract_publication_date(response.text)
        item.metadata["publication_date_source"] = (
            "explicit page metadata" if item.published_at else "unknown"
        )
    item.metadata["last_retrieved_at"] = datetime.now(UTC).isoformat()
    return item
