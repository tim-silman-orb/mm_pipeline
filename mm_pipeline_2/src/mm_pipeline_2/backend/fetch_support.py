"""HTTP, caching and Google News support for public-health fetchers."""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests

from .parsers import extract_link_href, extract_meta_content
from .utils import (
    DATA_DIR,
    Item,
    SourceConfig,
    infer_publisher_from_url,
    match_publisher,
    parse_datetime,
)


DEFAULT_TIMEOUT = 20
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
HTTP_CACHE_DIR = DATA_DIR / "http_cache"
SOURCE_ITEM_CACHE_DIR = DATA_DIR / "source_items"
NCBI_MIN_INTERVAL_SECONDS = 0.5
_LAST_NCBI_REQUEST_AT = 0.0
HEADERS = {
    "User-Agent": "epi-dossier/1.0 (+https://github.com/dteichrow/epi-dossier)",
    "Accept": "application/atom+xml,application/rss+xml,application/xml,text/xml,text/html;q=0.9,*/*;q=0.8",
}
BROWSERISH_HEADERS = {
    **HEADERS,
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class CachedHTTPResponse:
    def __init__(
        self,
        *,
        url: str,
        text: str,
        status_code: int,
        headers: dict[str, str],
        from_cache: bool = False,
        fetched_at: float | None = None,
    ) -> None:
        self.url = url
        self.text = text
        self.status_code = status_code
        self.headers = headers
        self.from_cache = from_cache
        self.fetched_at = fetched_at

    def json(self) -> dict:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} response for {self.url}")


def is_google_news_wrapper(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host.endswith("news.google.com")


def resolve_google_news_response(item: Item, logger: logging.Logger) -> requests.Response | None:
    try:
        response = resilient_get(
            item.url,
            headers=BROWSERISH_HEADERS,
            timeout=DEFAULT_TIMEOUT,
            allow_redirects=True,
            logger=logger,
            cache_namespace="google_news",
            max_attempts=1,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.info("Skipping Google News canonical resolution for %s: %s", item.url, exc)
        return None

    resolved_url = resolve_google_news_candidate_url(response)
    if not resolved_url:
        decode_params = extract_google_news_decode_params(response.text, response.url or item.url)
        if decode_params is not None:
            resolved_url = decode_google_news_batchexecute(*decode_params, logger=logger)
    if resolved_url and resolved_url != item.url and not is_google_news_wrapper(resolved_url):
        item.metadata.setdefault("raw_url", item.url)
        item.metadata.setdefault("aggregator_url", item.url)
        item.metadata["resolved_url"] = resolved_url
        item.url = resolved_url
        if not item.publisher:
            item.publisher = infer_publisher_from_url(resolved_url)
    return response


def resolve_google_news_candidate_url(response: requests.Response) -> str | None:
    final_url = response.url
    if final_url and not is_google_news_wrapper(final_url):
        return unwrap_google_redirect_url(final_url)
    document = response.text or ""
    canonical_url = extract_link_href(document, "canonical")
    if canonical_url and not is_google_news_wrapper(canonical_url):
        return unwrap_google_redirect_url(canonical_url)
    og_url = extract_meta_content(
        document,
        [
            ("property", "og:url"),
            ("name", "og:url"),
            ("property", "twitter:url"),
            ("name", "twitter:url"),
        ],
    )
    if og_url and not is_google_news_wrapper(og_url):
        return unwrap_google_redirect_url(og_url)
    embedded_urls = extract_google_news_embedded_urls(document)
    if embedded_urls:
        return embedded_urls[0]
    return None


def extract_google_news_embedded_urls(document: str) -> list[str]:
    normalized = html_lib.unescape(document or "")
    normalized = normalized.replace("\\/", "/").replace("\\u003d", "=").replace("\\u0026", "&")
    candidates: list[str] = []
    candidates.extend(re.findall(r"https?://[^\s\"'<>\\\\]+", normalized))
    candidates.extend(unquote(match) for match in re.findall(r"url=(https?%3A%2F%2F[^&\"'<>]+)", normalized, flags=re.IGNORECASE))
    candidates.extend(
        match
        for match in re.findall(
            r'"(?:targetUrl|canonicalUrl|articleUrl|externalUrl|sourceUrl|shareUrl|url)"\s*:\s*"(https?:[^"]+)"',
            normalized,
            flags=re.IGNORECASE,
        )
    )
    candidates.extend(
        match
        for match in re.findall(
            r"(?:data-n-au|data-url|data-href)\s*=\s*['\"](https?://[^'\"]+)['\"]",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    filtered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = clean_google_candidate_url(candidate)
        if not cleaned:
            continue
        host = urlparse(cleaned).netloc.lower()
        if not host or is_google_news_wrapper(cleaned):
            continue
        if is_known_non_article_url(cleaned):
            continue
        if host.endswith("google.com") or host.endswith("gstatic.com") or host.endswith("googleusercontent.com"):
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        filtered.append(cleaned)
    filtered.sort(key=google_news_candidate_priority)
    return filtered


def extract_google_news_decode_params(document: str, url: str) -> tuple[str, str, str] | None:
    article_id = urlparse(url).path.rstrip("/").split("/")[-1]
    if not article_id:
        return None
    ts_match = re.search(r'data-n-a-ts="(\d+)"', document)
    sig_match = re.search(r'data-n-a-sg="([^"]+)"', document)
    if not ts_match or not sig_match:
        return None
    return article_id, ts_match.group(1), sig_match.group(1)


def decode_google_news_batchexecute(
    article_id: str,
    timestamp: str,
    signature: str,
    *,
    logger: logging.Logger,
) -> str | None:
    payload = [
        [
            [
                "Fbv4je",
                (
                    '["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,null,null,null,null,null,0,1],'
                    '"X","X",1,[1,1,1],1,1,null,0,0,null,0],'
                    f'"{article_id}",{timestamp},"{signature}"]'
                ),
                None,
                "generic",
            ]
        ]
    ]
    try:
        response = requests.post(
            "https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je",
            headers={
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                "Referrer": "https://news.google.com/",
                **BROWSERISH_HEADERS,
            },
            data=f"f.req={quote(json.dumps(payload, separators=(',', ':')))}",
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.info("Google News batchexecute decode failed for %s: %s", article_id, exc)
        return None

    text = html_lib.unescape(response.text or "")
    text = text.replace("\\/", "/").replace("\\u003d", "=").replace("\\u0026", "&")
    text = text.replace('\\"', '"')
    match = re.search(r'\["garturlres","(https?://[^"]+)"', text)
    if not match:
        match = re.search(r'garturlres\\*",\\*"(https?://[^"\\]+)', text)
    if not match:
        return None
    decoded_url = clean_google_candidate_url(match.group(1))
    return decoded_url


def google_news_candidate_priority(url: str) -> tuple[int, int, int, int]:
    publisher = infer_publisher_from_url(url)
    profile = match_publisher(publisher, url)
    tier_rank = {
        "wire": 0,
        "major_newsroom": 1,
        "specialist_health": 2,
        "official": 3,
        "general": 4,
    }.get(profile.tier if profile else "general", 5)
    domain_score = 0 if publisher else 1
    path_penalty = 1 if is_known_non_article_url(url) else 0
    article_penalty = 0 if looks_like_article_url(url) else 1
    return (domain_score, tier_rank, article_penalty, path_penalty, len(url))


def clean_google_candidate_url(candidate: str) -> str | None:
    cleaned = candidate.rstrip(").,;\"'")
    cleaned = unwrap_google_redirect_url(cleaned)
    if not cleaned.startswith(("http://", "https://")):
        return None
    return cleaned


def unwrap_google_redirect_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.lower().endswith("google.com"):
        query = parse_qs(parsed.query)
        for key in ("url", "q", "u"):
            values = query.get(key)
            if values and values[0].startswith(("http://", "https://")):
                return unquote(values[0])
    return url


def looks_like_article_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    raw_path = parsed.path.lower()
    path = raw_path.strip("/")
    if not path:
        return False
    if any(
        token in raw_path
        for token in (
            "/article/",
            "/articles/",
            "/story/",
            "/stories/",
            "/news/",
            "/world/",
            "/health/",
            "/science/",
            "/us/",
            "/uk/",
            "/europe/",
            "/asia/",
        )
    ):
        return True
    if re.search(r"/20\d{2}/\d{2}/\d{2}/", parsed.path):
        return True
    if host.endswith("reuters.com") and path.count("/") >= 2:
        return True
    if host.endswith("nytimes.com") and path.count("/") >= 3:
        return True
    if host.endswith("theguardian.com") and path.count("/") >= 3:
        return True
    if host.endswith("bbc.com") and path.startswith("news/"):
        return True
    return False


def is_known_non_article_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    lowered_path = path.lower()
    if lowered_path in {"/css", "/css2"}:
        return True
    if lowered_path.endswith((".js", ".css", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".woff", ".woff2", ".ttf", ".map")):
        return True
    return (
        host in {"www.w3.org", "w3.org"}
        or host in {"www.google-analytics.com", "google-analytics.com", "www.googletagmanager.com", "googletagmanager.com", "stats.g.doubleclick.net"}
        or host in {"fonts.googleapis.com", "fonts.gstatic.com"}
        or host in {"angular.dev", "angular.io", "developer.mozilla.org", "schema.org", "www.schema.org"}
        or path == "/license"
    )


def resilient_get(
    url: str,
    *,
    headers: dict[str, str],
    timeout: int,
    logger: logging.Logger,
    cache_namespace: str,
    params: dict | None = None,
    allow_redirects: bool = True,
    max_attempts: int = 3,
    max_cache_age_hours: int = 48,
    verify_ssl: bool = True,
) -> CachedHTTPResponse | requests.Response:
    cache_path = cache_file_for_request(cache_namespace, url, params)
    last_error: Exception | None = None
    cached_response: CachedHTTPResponse | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                allow_redirects=allow_redirects,
                verify=verify_ssl,
            )
            response.raise_for_status()
            cache_response(cache_path, response)
            return response
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if should_fallback_to_cache_immediately(exc):
                if cached_response is None:
                    cached_response = load_cached_response(cache_path, max_cache_age_hours=max_cache_age_hours)
                if cached_response is not None:
                    logger.info("Using cached response for %s after immediate transport failure: %s", url, exc)
                    return cached_response
            if not should_retry_exception(exc):
                break
            if attempt < max_attempts:
                sleep_seconds = retry_sleep_seconds(exc, attempt)
                if sleep_seconds > 0:
                    logger.info("Retrying %s after %ss (%s)", url, sleep_seconds, exc)
                    time.sleep(sleep_seconds)

    cached = load_cached_response(cache_path, max_cache_age_hours=max_cache_age_hours)
    if cached is not None:
        logger.info("Using cached response for %s after upstream failure: %s", url, last_error)
        return cached
    raise last_error or RuntimeError(f"Failed to fetch {url}")


def ncbi_get(
    url: str,
    *,
    headers: dict[str, str],
    timeout: int,
    logger: logging.Logger,
    cache_namespace: str,
    params: dict | None = None,
    allow_redirects: bool = True,
    max_attempts: int = 3,
    max_cache_age_hours: int = 48,
    verify_ssl: bool = True,
) -> CachedHTTPResponse | requests.Response:
    throttle_ncbi_requests()
    return resilient_get(
        url,
        headers=headers,
        timeout=timeout,
        logger=logger,
        cache_namespace=cache_namespace,
        params=params,
        allow_redirects=allow_redirects,
        max_attempts=max_attempts,
        max_cache_age_hours=max_cache_age_hours,
        verify_ssl=verify_ssl,
    )


def throttle_ncbi_requests() -> None:
    global _LAST_NCBI_REQUEST_AT
    now = time.time()
    wait_seconds = (_LAST_NCBI_REQUEST_AT + NCBI_MIN_INTERVAL_SECONDS) - now
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    _LAST_NCBI_REQUEST_AT = time.time()


def source_cache_path(source: SourceConfig) -> Path:
    SOURCE_ITEM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(source.name.encode("utf-8")).hexdigest()
    return SOURCE_ITEM_CACHE_DIR / f"{digest}.json"


def save_source_item_cache(source: SourceConfig, items: list[Item]) -> None:
    path = source_cache_path(source)
    cached_at = time.time()
    payload = {
        "source": source.name,
        "cached_at": cached_at,
        "items": [serialize_item(item) for item in items],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_source_item_cache(source: SourceConfig, *, max_age_hours: int) -> list[Item] | None:
    payload = load_source_item_cache_payload(source, max_age_hours=max_age_hours)
    if payload is None:
        return None
    items, _, _ = payload
    return items


def load_source_item_cache_payload(
    source: SourceConfig,
    *,
    max_age_hours: int,
) -> tuple[list[Item], datetime, float] | None:
    path = source_cache_path(source)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cached_at = payload.get("cached_at")
    if not isinstance(cached_at, (int, float)):
        return None
    age_seconds = time.time() - cached_at
    if age_seconds > max_age_hours * 3600:
        return None
    item_payloads = payload.get("items", [])
    if not isinstance(item_payloads, list):
        return None
    cached_at_dt = datetime.fromtimestamp(cached_at, tz=UTC)
    return (
        [deserialize_item(entry) for entry in item_payloads if isinstance(entry, dict)],
        cached_at_dt,
        round(age_seconds / 3600, 2),
    )


def stamp_items_from_source(
    items: list[Item],
    *,
    source: SourceConfig,
    freshness_state: str,
    source_cached_at: datetime | None,
    source_cache_age_hours: float | None,
) -> None:
    for item in items:
        item.metadata["source_name"] = source.name
        item.metadata["source_official"] = source.official
        item.metadata["source_outbreak_signal"] = source.outbreak_signal
        item.metadata["source_freshness"] = freshness_state
        if source_cached_at is not None:
            item.metadata["source_cached_at"] = source_cached_at.isoformat()
        if source_cache_age_hours is not None:
            item.metadata["source_cache_age_hours"] = source_cache_age_hours


def build_source_health_entry(
    source: SourceConfig,
    *,
    mode: str,
    item_count: int,
    cached_at: datetime | None = None,
    cache_age_hours: float | None = None,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "source": source.name,
        "source_type": source.type,
        "official": source.official,
        "required": source.required,
        "mode": mode,
        "item_count": item_count,
        "cached_at": cached_at.isoformat() if cached_at else None,
        "cache_age_hours": cache_age_hours,
        "error": error,
    }


def serialize_item(item: Item) -> dict:
    return {
        "title": item.title,
        "source": item.source,
        "url": item.url,
        "category": item.category,
        "publisher": item.publisher,
        "published_at": item.published_at.isoformat() if item.published_at else None,
        "summary": item.summary,
        "why_it_matters": item.why_it_matters,
        "caveats": item.caveats,
        "relevance_score": item.relevance_score,
        "doi": item.doi,
        "journal": item.journal,
        "abstract_url": item.abstract_url,
        "source_type": item.source_type,
        "official": item.official,
        "extracted_text": item.extracted_text,
        "metadata": item.metadata,
    }


def deserialize_item(payload: dict) -> Item:
    return Item(
        title=str(payload.get("title", "")),
        source=str(payload.get("source", "")),
        url=str(payload.get("url", "")),
        category=str(payload.get("category", "")),
        publisher=payload.get("publisher"),
        published_at=parse_datetime(payload.get("published_at")),
        summary=str(payload.get("summary", "")),
        why_it_matters=str(payload.get("why_it_matters", "")),
        caveats=str(payload.get("caveats", "")),
        relevance_score=int(payload.get("relevance_score", 1) or 1),
        doi=payload.get("doi"),
        journal=payload.get("journal"),
        abstract_url=payload.get("abstract_url"),
        source_type=payload.get("source_type"),
        official=bool(payload.get("official", False)),
        extracted_text=str(payload.get("extracted_text", "")),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def retry_sleep_seconds(exc: Exception, attempt: int) -> int:
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(1, min(int(retry_after), 5))
            except ValueError:
                pass
    return min(2 ** (attempt - 1), 5)


def should_retry_exception(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if response is None:
        return True
    return response.status_code in RETRYABLE_STATUS_CODES


def should_fallback_to_cache_immediately(exc: Exception) -> bool:
    return isinstance(exc, requests.RequestException) and getattr(exc, "response", None) is None


def cache_file_for_request(namespace: str, url: str, params: dict | None) -> Path:
    HTTP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"url": url, "params": params or {}}, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return HTTP_CACHE_DIR / f"{namespace}_{digest}.json"


def cache_response(cache_path: Path, response: requests.Response) -> None:
    raw_status = getattr(response, "status_code", 200)
    status_code = raw_status if isinstance(raw_status, int) else 200
    if status_code >= 400:
        return
    raw_headers = getattr(response, "headers", {})
    header_map = dict(raw_headers) if isinstance(raw_headers, dict) else {}
    payload = {
        "url": str(getattr(response, "url", "")),
        "status_code": status_code,
        "headers": header_map,
        "text": str(getattr(response, "text", "")),
        "fetched_at": time.time(),
    }
    cache_path.write_text(json.dumps(payload), encoding="utf-8")


def load_cached_response(cache_path: Path, *, max_cache_age_hours: int) -> CachedHTTPResponse | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    fetched_at = payload.get("fetched_at")
    if not isinstance(fetched_at, (int, float)):
        return None
    if (time.time() - fetched_at) > max_cache_age_hours * 3600:
        return None
    return CachedHTTPResponse(
        url=str(payload.get("url", "")),
        text=str(payload.get("text", "")),
        status_code=int(payload.get("status_code", 200)),
        headers={str(key): str(value) for key, value in dict(payload.get("headers", {})).items()},
        from_cache=True,
        fetched_at=float(fetched_at),
    )
