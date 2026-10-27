from __future__ import annotations

import logging
import os
from typing import Any

from mm_pipeline.config import SourceConfig
from mm_pipeline.models import Article, CollectionWindow, SourceResult

from .http import HttpClient
from .parsers import clean_text, parse_datetime


class OPointAdapter:
    endpoint = "https://api.opoint.com/storedsearch/feed/"

    def __init__(self, config: SourceConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.http = HttpClient(logger=logger)

    def collect(self, window: CollectionWindow) -> SourceResult:
        token_env = str(self.config.options.get("token_env", "OPOINT_TOKEN"))
        token = os.getenv(token_env, "").strip()
        if not token:
            raise ValueError(f"Required environment variable is missing: {token_env}")
        search_id = str(self.config.options.get("stored_search_id", "")).strip()
        if not search_id:
            raise ValueError(f"OPoint source {self.config.id} requires stored_search_id")
        response = self.http.get(
            self.endpoint,
            headers={"Authorization": f"Token {token}", "Accept": "application/json"},
            params={
                "from": int(window.start.timestamp()),
                "to": int(window.end.timestamp()),
                "format": "json",
                "num_art": int(self.config.options.get("max_items", 25_000)),
            },
        )
        documents: Any = response.json().get("searchresult", {}).get("document", [])
        if isinstance(documents, dict):
            documents = [documents]
        records: list[Article] = []
        seen_headlines: set[str] = set()
        for document in documents:
            if str(document.get("stored_search_id", "")) != search_id:
                continue
            headline = clean_text(_nested(document, "header", "text"))
            if headline and headline.casefold() in seen_headlines:
                continue
            if headline:
                seen_headlines.add(headline.casefold())
            summary = clean_text(_nested(document, "summary", "text"))
            body = clean_text(_nested(document, "body", "text"))
            records.append(
                Article(
                    platform=self.config.platform,
                    source_id=clean_text(document.get("id_article")),
                    published_at=parse_datetime(_nested(document, "local_time", "text")),
                    source_name=clean_text(_nested(document, "first_source", "name")) or self.config.name,
                    headline=headline,
                    content="\n\n".join(part for part in (headline, summary, body) if part),
                    summary=summary,
                    url=clean_text(document.get("url")),
                    language=clean_text(_nested(document, "language", "text")),
                    country=clean_text(document.get("countryname")),
                    author=clean_text(document.get("author")),
                    search_id=search_id,
                )
            )
        return SourceResult(self.config.id, records)


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key, "")
    return current
