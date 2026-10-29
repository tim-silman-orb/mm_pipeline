from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from .config import SourceConfig
from .exceptions import ConfigurationError
from .models import CollectionWindow, SourceResult
from .sources.html import HTMLListAdapter, HTMLPageAdapter
from .sources.opoint import OPointAdapter
from .sources.rss import RSSAdapter
from .sources.telegram import TelegramAdapter
from .sources.x import XAdapter


AdapterFactory = Callable[[SourceConfig, logging.Logger], Any]


class SourceAdapter(Protocol):
    """Contract implemented by every source mechanism."""

    def collect(self, window: CollectionWindow) -> SourceResult:
        ...


ADAPTERS: dict[str, AdapterFactory] = {
    "rss": RSSAdapter,
    "html_list": HTMLListAdapter,
    "html_page": HTMLPageAdapter,
    "opoint": OPointAdapter,
    "telegram": TelegramAdapter,
    "x": XAdapter,
}


def register_adapter(name: str, factory: AdapterFactory) -> None:
    if not name.strip():
        raise ValueError("Adapter name cannot be empty")
    ADAPTERS[name] = factory


def create_adapter(source: SourceConfig, logger: logging.Logger) -> Any:
    try:
        factory = ADAPTERS[source.adapter]
    except KeyError as exc:
        available = ", ".join(sorted(ADAPTERS))
        raise ConfigurationError(
            f"Unknown adapter {source.adapter!r} for source {source.id!r}. "
            f"Available adapters: {available}"
        ) from exc
    return factory(source, logger)
