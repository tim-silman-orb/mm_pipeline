from __future__ import annotations

import logging
import os

from mm_pipeline.config import SourceConfig
from mm_pipeline.models import Article, CollectionWindow, SourceResult

from .http import HttpClient
from .parsers import clean_text, parse_datetime


class XAdapter:
    endpoint = "https://api.x.com/2/tweets/search/recent"

    def __init__(self, config: SourceConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.http = HttpClient(logger=logger)

    def collect(self, window: CollectionWindow) -> SourceResult:
        token_env = str(self.config.options.get("bearer_token_env", "X_BEARER_TOKEN"))
        token = os.getenv(token_env, "").strip()
        if not token:
            raise ValueError(f"Required environment variable is missing: {token_env}")
        query = str(self.config.options.get("query", "")).strip()
        if not query:
            raise ValueError(f"X source {self.config.id} requires a query")
        response = self.http.get(
            self.endpoint,
            headers={"Authorization": f"Bearer {token}"},
            params={
                "query": query,
                "max_results": int(self.config.options.get("max_results", 10)),
                "start_time": window.start.isoformat().replace("+00:00", "Z"),
                "end_time": window.end.isoformat().replace("+00:00", "Z"),
                "tweet.fields": "id,text,created_at,lang,author_id",
                "expansions": "author_id",
                "user.fields": "id,name,username",
            },
        )
        payload = response.json()
        users = {
            str(user.get("id")): user
            for user in payload.get("includes", {}).get("users", [])
        }
        records: list[Article] = []
        for post in payload.get("data", []):
            user = users.get(str(post.get("author_id")), {})
            username = clean_text(user.get("username"))
            post_id = clean_text(post.get("id"))
            records.append(
                Article(
                    platform=self.config.platform,
                    source_id=post_id,
                    published_at=parse_datetime(post.get("created_at")),
                    source_name=clean_text(user.get("name")) or self.config.name,
                    headline="",
                    content=clean_text(post.get("text")),
                    summary="",
                    url=f"https://x.com/{username}/status/{post_id}" if username else "",
                    language=clean_text(post.get("lang")),
                    author=username,
                    search_id=query,
                )
            )
        return SourceResult(self.config.id, records)
