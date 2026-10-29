from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from dateutil import parser as date_parser


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = date_parser.parse(str(value))
        except (TypeError, ValueError, OverflowError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def extract_links(html: str, *, base_url: str, selector: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for node in soup.select(selector):
        link = node if node.name == "a" else node.find("a")
        if link is None or not link.get("href"):
            continue
        url = urljoin(base_url, str(link["href"]))
        if url in seen:
            continue
        seen.add(url)
        rows.append({"title": clean_text(link.get_text(" ")), "url": url})
    return rows


def extract_page_text(html: str, *, selector: str | None = None) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for node in soup.select("script, style, nav, footer, noscript"):
        node.decompose()
    root = soup.select_one(selector) if selector else None
    root = root or soup.select_one("article") or soup.select_one("main") or soup.body or soup
    paragraphs = [clean_text(node.get_text(" ")) for node in root.select("p")]
    usable = [paragraph for paragraph in paragraphs if len(paragraph) >= 30]
    return "\n\n".join(usable)


def extract_meta(html: str, *keys: tuple[str, str]) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for attribute, value in keys:
        node = soup.find("meta", attrs={attribute: value})
        if node and node.get("content"):
            return clean_text(node["content"])
    return ""
