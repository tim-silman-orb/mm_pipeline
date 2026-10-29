from __future__ import annotations

import hashlib
import logging

from mm_pipeline.config import SourceConfig
from mm_pipeline.models import Article, CollectionWindow, SourceResult

from .http import HttpClient
from .parsers import extract_links, extract_meta, extract_page_text, parse_datetime


class HTMLListAdapter:
    def __init__(self, config: SourceConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.http = HttpClient(
            logger=logger,
            timeout_seconds=int(config.options.get("timeout_seconds", 30)),
            max_attempts=int(config.options.get("max_attempts", 3)),
        )

    def collect(self, window: CollectionWindow) -> SourceResult:
        if not self.config.url:
            raise ValueError(f"HTML source {self.config.id} requires a URL")
        selector = self.config.options.get("item_selector")
        if not selector:
            raise ValueError(f"HTML list source {self.config.id} requires item_selector")
        response = self.http.get(self.config.url)
        rows = extract_links(response.text, base_url=self.config.url, selector=str(selector))
        maximum = int(self.config.options.get("max_items", len(rows)))
        enrich = bool(self.config.options.get("enrich", True))
        records: list[Article] = []
        warnings: list[str] = []
        for row in rows[:maximum]:
            content = ""
            summary = ""
            published = None
            if enrich:
                try:
                    item_response = self.http.get(row["url"])
                    content = extract_page_text(
                        item_response.text,
                        selector=self.config.options.get("content_selector"),
                    )
                    summary = extract_meta(
                        item_response.text,
                        ("name", "description"),
                        ("property", "og:description"),
                    )
                    published = parse_datetime(
                        extract_meta(
                            item_response.text,
                            ("property", "article:published_time"),
                            ("name", "date"),
                        )
                    )
                except Exception as exc:  # Keep the list record if enrichment fails.
                    warnings.append(f"Could not enrich {row['url']}: {exc}")
            if published and not (window.start <= published < window.end):
                continue
            source_id = hashlib.sha256(row["url"].encode()).hexdigest()[:24]
            records.append(
                Article(
                    platform=self.config.platform,
                    source_id=source_id,
                    published_at=published,
                    source_name=self.config.name,
                    headline=row["title"],
                    content=content,
                    summary=summary,
                    url=row["url"],
                )
            )
        return SourceResult(self.config.id, records, warnings)


class HTMLPageAdapter:
    def __init__(self, config: SourceConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.http = HttpClient(logger=logger)

    def collect(self, window: CollectionWindow) -> SourceResult:
        if not self.config.url:
            raise ValueError(f"HTML page source {self.config.id} requires a URL")
        response = self.http.get(self.config.url)
        title = extract_meta(response.text, ("property", "og:title"), ("name", "title"))
        content = extract_page_text(
            response.text,
            selector=self.config.options.get("content_selector"),
        )
        source_id = hashlib.sha256(self.config.url.encode()).hexdigest()[:24]
        article = Article(
            platform=self.config.platform,
            source_id=source_id,
            published_at=None,
            source_name=self.config.name,
            headline=title or self.config.name,
            content=content,
            summary="",
            url=self.config.url,
        )
        return SourceResult(self.config.id, [article])
