"""Models and small helpers shared by the retained health-source backend.

This is deliberately narrower than the original project-wide ``utils.py``:
briefing, website, email and atlas helpers do not belong in the data pipeline.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlparse, urlunparse

import yaml
from dateutil import parser as date_parser


ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"


@dataclass
class SourceConfig:
    name: str
    type: str
    category: str
    official: bool = False
    outbreak_signal: bool = False
    occupational_relevance: bool = False
    historical_relevance: bool = False
    required: bool = True
    url: str | None = None
    search: str | None = None
    item_selector: str | None = None
    terms_hint: list[str] = field(default_factory=list)
    max_items: int | None = None
    refresh_hours: int | None = None
    fallback_cache_hours: int | None = 48
    timeout_seconds: int | None = None
    max_attempts: int | None = None
    verify_ssl: bool = True


@dataclass
class PublisherProfile:
    key: str
    name: str
    domains: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    tier: str = "general"
    access: str = "open"


@dataclass
class Item:
    title: str
    source: str
    url: str
    category: str
    publisher: str | None = None
    published_at: datetime | None = None
    summary: str = ""
    why_it_matters: str = ""
    caveats: str = ""
    relevance_score: int = 1
    doi: str | None = None
    journal: str | None = None
    abstract_url: str | None = None
    source_type: str | None = None
    official: bool = False
    extracted_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.url = normalize_public_url(self.url)
        if self.abstract_url:
            self.abstract_url = normalize_public_url(self.abstract_url)
        for key in ("raw_url", "aggregator_url", "resolved_url"):
            value = self.metadata.get(key)
            if isinstance(value, str):
                self.metadata[key] = normalize_public_url(value)


def configure_project_root(project_root: str | Path) -> None:
    """Point config and cache paths at the standalone repository root."""
    global ROOT_DIR, CONFIG_DIR, DATA_DIR

    ROOT_DIR = Path(project_root).expanduser().resolve()
    CONFIG_DIR = ROOT_DIR / "config"
    DATA_DIR = ROOT_DIR / "data"
    load_publishers.cache_clear()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def load_publishers() -> tuple[PublisherProfile, ...]:
    raw = load_yaml(CONFIG_DIR / "publishers.yml").get("publishers", [])
    return tuple(PublisherProfile(**entry) for entry in raw)


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    try:
        return date_parser.parse(str(value))
    except (ValueError, TypeError, OverflowError):
        return None


def normalize_public_url(url: str) -> str:
    return (
        str(url or "")
        .strip()
        .replace("\\=", "=")
        .replace("\\&", "&")
        .replace("\\?", "?")
    )


def canonicalize_url(url: str) -> str:
    parsed = urlparse(normalize_public_url(url))
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    cleaned = parsed._replace(
        scheme=parsed.scheme.lower() or "https",
        netloc=netloc,
        fragment="",
        query="&".join(f"{key}={value}" for key, value in query),
    )
    normalized = urlunparse(cleaned)
    return normalized[:-1] if normalized.endswith("/") else normalized


def absolute_url(base_url: str, href: str) -> str:
    return urljoin(base_url, href)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_source_name(text: str | None) -> str:
    if not text:
        return ""
    normalized = normalize_whitespace(text)
    return re.sub(r"\s+\|.*$", "", normalized).strip()


def infer_publisher_from_url(url: str) -> str | None:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    for profile in load_publishers():
        if any(host == domain or host.endswith(f".{domain}") for domain in profile.domains):
            return profile.name
    return None


def match_publisher(name: str | None, url: str | None = None) -> PublisherProfile | None:
    normalized_name = normalize_source_name(name).lower()
    host = urlparse(url).netloc.lower() if url else ""
    if host.startswith("www."):
        host = host[4:]
    for profile in load_publishers():
        aliases = [profile.name, *profile.aliases]
        if normalized_name and any(
            normalized_name == normalize_source_name(alias).lower() for alias in aliases
        ):
            return profile
        if host and any(host == domain or host.endswith(f".{domain}") for domain in profile.domains):
            return profile
    return None


def strip_html(text: str) -> str:
    unescaped = html.unescape(text or "")
    return normalize_whitespace(re.sub(r"<[^>]+>", " ", unescaped))


def clean_headline(text: str) -> str:
    cleaned = strip_html(text)
    if " - " in cleaned:
        head, tail = cleaned.rsplit(" - ", 1)
        if 1 <= len(tail.split()) <= 5:
            cleaned = head
    return normalize_whitespace(cleaned.rstrip(" -|:"))
