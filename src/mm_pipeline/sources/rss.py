from __future__ import annotations

import hashlib
import logging

import feedparser

from mm_pipeline.config import SourceConfig
from mm_pipeline.models import Article, CollectionWindow, SourceResult

from .http import HttpClient
from .parsers import clean_text, parse_datetime


class RSSAdapter:
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
            raise ValueError(f"RSS source {self.config.id} requires a URL")
        response = self.http.get(
            self.config.url,
            verify=bool(self.config.options.get("verify_ssl", True)),
        )
        parsed = feedparser.parse(response.content)
        maximum = int(self.config.options.get("max_items", len(parsed.entries)))
        records: list[Article] = []
        warnings: list[str] = []
        for entry in parsed.entries[:maximum]:
            published = parse_datetime(
                entry.get("published") or entry.get("updated") or entry.get("created")
            )
            if published and not (window.start <= published < window.end):
                continue
            url = clean_text(entry.get("link"))
            title = clean_text(entry.get("title"))
            summary = clean_text(entry.get("summary") or entry.get("description"))
            source_id = clean_text(entry.get("id") or entry.get("guid") or url)
            if not source_id:
                source_id = hashlib.sha256(f"{title}|{summary}".encode()).hexdigest()[:24]
            records.append(
                Article(
                    platform=self.config.platform,
                    source_id=source_id,
                    published_at=published,
                    source_name=self.config.name,
                    headline=title,
                    content="",
                    summary=summary,
                    url=url,
                    language=clean_text(entry.get("language")),
                )
            )
        if getattr(parsed, "bozo", False):
            warnings.append(f"Feed parser warning: {parsed.bozo_exception}")
        return SourceResult(self.config.id, records, warnings)
