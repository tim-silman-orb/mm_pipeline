from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class CollectionWindow:
    """Half-open UTC collection interval: start <= timestamp < end."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("Collection window datetimes must be timezone-aware")
        if self.start >= self.end:
            raise ValueError("Collection window start must be before end")


@dataclass(frozen=True)
class Article:
    platform: str
    source_id: str
    published_at: datetime | None
    source_name: str
    headline: str
    content: str
    summary: str
    url: str
    language: str = ""
    country: str = ""
    author: str = ""
    search_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def article_id(self) -> str:
        source_id = self.source_id.strip()
        if not source_id:
            material = "\x1f".join((self.url, self.headline, self.source_name))
            source_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        return f"{self.platform.strip() or 'unknown'}_{source_id}"

    @property
    def analysis_text(self) -> str:
        return self.content.strip() or self.summary.strip()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["published_at"] = (
            self.published_at.astimezone(UTC).isoformat()
            if self.published_at is not None
            else None
        )
        payload["article_id"] = self.article_id
        payload["analysis_text"] = self.analysis_text
        return payload


@dataclass
class SourceResult:
    source_id: str
    records: list[Article]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SourceFailure:
    source_id: str
    required: bool
    error: str


@dataclass
class PipelineResult:
    run_id: str
    project_id: str
    window: CollectionWindow
    articles: list[Article]
    source_results: list[SourceResult]
    failures: list[SourceFailure]
    output_locations: dict[str, str]
    fatal_errors: list[str] = field(default_factory=list)

    @property
    def required_failures(self) -> list[SourceFailure]:
        return [failure for failure in self.failures if failure.required]

    @property
    def succeeded(self) -> bool:
        return not self.fatal_errors
