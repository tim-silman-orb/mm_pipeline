from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import PipelineConfig
from .workspace import ArtifactWorkspace


COMMON_COLUMNS = [
    "platform",
    "source_id",
    "published_at",
    "country",
    "language",
    "source_name",
    "author",
    "headline",
    "content",
    "summary",
    "url",
    "search_id",
]


def required_environment(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


@dataclass(frozen=True)
class RunWindow:
    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")

    @property
    def lookback_days(self) -> int:
        return max(1, math.ceil((self.end_date - self.start_date).days) + 1)


class SourceCollector(ABC):
    """Small OOP boundary around the existing source-specific functions."""

    source_name: str

    def __init__(
        self,
        *,
        config: PipelineConfig,
        settings: dict[str, Any],
        workspace: ArtifactWorkspace,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.settings = settings
        self.workspace = workspace
        self.logger = logger
        from .backend import utils as backend_utils

        backend_utils.configure_project_root(config.project_root)

    @abstractmethod
    def collect(self, window: RunWindow) -> pd.DataFrame:
        """Return records in the legacy canonical schema."""


class OPointCollector(SourceCollector):
    source_name = "opoint"

    def collect(self, window: RunWindow) -> pd.DataFrame:
        from .backend.opoint_functions import collect_opoint, standardise_opoint

        token = required_environment(str(self.settings.get("token_env", "TOKEN")))
        search_env = str(self.settings.get("stored_search_id_env", "")).strip()
        search_value = os.getenv(search_env, "").strip() if search_env else ""
        stored_search_id = int(search_value or self.settings["stored_search_id"])
        raw = collect_opoint(
            token=token,
            stored_search_id=stored_search_id,
            days_back=window.lookback_days,
            num_art=int(self.settings.get("max_items", 25_000)),
        )
        result = standardise_opoint(raw)
        self.logger.info("Collected %s OPoint article(s)", len(result))
        return result


class HealthCollector(SourceCollector):
    source_name = "health"

    def collect(self, window: RunWindow) -> pd.DataFrame:
        mode = str(self.settings.get("mode", "live")).lower()
        health_file = self.workspace.path("health_items.jsonl")
        if mode == "file":
            if not health_file.exists():
                raise FileNotFoundError(f"Health snapshot not found: {health_file}")
            result = pd.read_json(health_file, lines=True)
            self.logger.info("Loaded %s health record(s) from %s", len(result), health_file)
            return result
        if mode != "live":
            raise ValueError("health.mode must be either 'live' or 'file'")

        from .backend.fetchers import enrich_item_text, fetch_all_sources
        from .backend.utils import SourceConfig

        source_path = self.config.project_path(str(self.settings["source_config"]))
        with source_path.open("r", encoding="utf-8") as handle:
            raw_sources = (yaml.safe_load(handle) or {}).get("sources", [])
        sources = [SourceConfig(**entry) for entry in raw_sources]

        items, failures, source_health = fetch_all_sources(
            sources=sources,
            logger=self.logger.getChild("fetch"),
            start_date=window.start_date,
            end_date=window.end_date,
        )
        if failures:
            self.logger.warning("%s health source(s) failed", len(failures))

        if bool(self.settings.get("enrich", True)):
            enrichment_logger = self.logger.getChild("enrichment")
            enriched = 0
            for item in items:
                if "Google News" in item.source:
                    continue
                enrich_item_text(item, enrichment_logger)
                enriched += 1
                if enriched % 50 == 0:
                    self.logger.info("Health enrichment progress: %s articles", enriched)
            self.logger.info("Health enrichment complete: %s articles", enriched)

        result = pd.DataFrame(
            [
                {
                    "platform": "health_source",
                    "source_id": item.url or item.title,
                    "published_at": item.published_at,
                    "country": "",
                    "language": "en",
                    "source_name": item.source,
                    "author": "",
                    "headline": item.title,
                    "content": item.extracted_text or "",
                    "summary": item.summary or "",
                    "url": item.url,
                    "search_id": "",
                }
                for item in items
            ],
            columns=COMMON_COLUMNS,
        )
        result.to_json(
            health_file,
            orient="records",
            lines=True,
            force_ascii=False,
            date_format="iso",
        )
        health_manifest = self.workspace.path("health_source_status.json")
        health_manifest.write_text(
            json.dumps(
                {"failures": failures, "sources": source_health},
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        self.logger.info("Collected %s health record(s)", len(result))
        return result


class XCollector(SourceCollector):
    source_name = "x"

    def collect(self, window: RunWindow) -> pd.DataFrame:
        del window
        from .backend.x_functions import pull_recent_x, standardise_x

        token = required_environment(str(self.settings.get("bearer_token_env", "X_BEARER_TOKEN")))
        raw = pull_recent_x(
            bearer_token=token,
            query=str(self.settings.get("query", "Ebola")),
            max_results=int(self.settings.get("max_results", 10)),
        )
        result = standardise_x(raw)
        self.logger.info("Collected %s X post(s)", len(result))
        return result


class TelegramCollector(SourceCollector):
    source_name = "telegram"

    def collect(self, window: RunWindow) -> pd.DataFrame:
        del window
        try:
            from telethon import TelegramClient
            from telethon.sessions import StringSession
        except ImportError as exc:
            raise RuntimeError("Install mm-pipeline-2[telegram] to enable Telegram") from exc
        from .backend.telegram_functions import collect_telegram_messages, standardise_telegram

        api_id = int(required_environment(str(self.settings.get("api_id_env", "API_ID"))))
        api_hash = required_environment(str(self.settings.get("api_hash_env", "API_HASH")))
        session_string = required_environment(str(self.settings.get("session_env", "SESSION_STRING")))
        channels = list(self.settings.get("channels", []))
        limit = int(self.settings.get("limit", 100))

        async def run() -> pd.DataFrame:
            client = TelegramClient(StringSession(session_string), api_id, api_hash)
            await client.connect()
            try:
                if not await client.is_user_authorized():
                    raise RuntimeError("Telegram session is not authorised")
                raw = await collect_telegram_messages(client, channels, limit=limit)
                return standardise_telegram(raw)
            finally:
                await client.disconnect()

        result = asyncio.run(run())
        self.logger.info("Collected %s Telegram message(s)", len(result))
        return result


COLLECTOR_TYPES: dict[str, type[SourceCollector]] = {
    "opoint": OPointCollector,
    "health": HealthCollector,
    "x": XCollector,
    "telegram": TelegramCollector,
}


class MasterDatasetBuilder:
    """Single home for schema completion, topic filtering and deduplication."""

    def __init__(self, topic_filter: dict[str, Any]) -> None:
        self.topic_filter = topic_filter

    def build(self, frames: list[pd.DataFrame], *, topic_terms: list[str] | None = None) -> pd.DataFrame:
        if not frames:
            raise ValueError("No source datasets were collected")
        canonical: list[pd.DataFrame] = []
        for original in frames:
            frame = original.copy()
            for column in COMMON_COLUMNS:
                if column not in frame.columns:
                    frame[column] = ""
            canonical.append(frame[COMMON_COLUMNS])

        combined = pd.concat(canonical, ignore_index=True)
        combined["article_id"] = (
            combined["platform"].fillna("unknown").astype(str).str.strip()
            + "_"
            + combined["source_id"].fillna("").astype(str).str.strip()
        )
        content = combined["content"].fillna("").astype(str).str.strip()
        summary = combined["summary"].fillna("").astype(str).str.strip()
        combined["analysis_text"] = content.where(content.ne(""), summary)

        terms = topic_terms if topic_terms is not None else list(self.topic_filter.get("terms", []))
        platforms = set(self.topic_filter.get("platforms", []))
        fields = list(self.topic_filter.get("fields", ["headline", "content", "summary"]))
        if terms:
            text = combined[fields].fillna("").astype(str).agg(" ".join, axis=1)
            matches = text.str.contains("|".join(terms), case=False, na=False, regex=True)
            targeted = combined["platform"].isin(platforms) if platforms else pd.Series(True, index=combined.index)
            combined = combined[(~targeted) | matches].copy()

        combined = combined.drop_duplicates(subset=["article_id"], keep="last").copy()
        combined = combined[
            combined["analysis_text"].fillna("").astype(str).str.strip().ne("")
        ].copy()
        return combined.reset_index(drop=True)


@dataclass(frozen=True)
class CollectionResult:
    dataframe: pd.DataFrame
    output_path: Path
    source_counts: dict[str, int]


class DataCollectionPipeline:
    def __init__(
        self,
        *,
        config: PipelineConfig,
        workspace: ArtifactWorkspace,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.logger = logger

    def run(
        self,
        *,
        window: RunWindow,
        selected_sources: set[str] | None = None,
        topic_terms: list[str] | None = None,
    ) -> CollectionResult:
        frames: list[pd.DataFrame] = []
        counts: dict[str, int] = {}
        known = set(self.config.sources)
        if selected_sources:
            unknown = selected_sources - known
            if unknown:
                raise ValueError(f"Unknown source(s): {sorted(unknown)}")

        for name, settings in self.config.sources.items():
            enabled = bool(settings.get("enabled", False))
            if selected_sources is not None:
                enabled = name in selected_sources
            if not enabled:
                continue
            collector_type = COLLECTOR_TYPES.get(name)
            if collector_type is None:
                raise ValueError(f"No collector class registered for source: {name}")
            collector = collector_type(
                config=self.config,
                settings=settings,
                workspace=self.workspace,
                logger=self.logger.getChild(name),
            )
            frame = collector.collect(window)
            frames.append(frame)
            counts[name] = len(frame)

        master = MasterDatasetBuilder(self.config.topic_filter).build(
            frames,
            topic_terms=topic_terms,
        )
        output = self.workspace.path(str(self.config.extraction.get("master_data_file", "master_data.jsonl")))
        master.to_json(
            output,
            orient="records",
            lines=True,
            force_ascii=False,
            date_format="iso",
        )
        manifest = {
            "project": self.config.project,
            "created_at": datetime.now(UTC).isoformat(),
            "window_start": window.start_date.isoformat(),
            "window_end": window.end_date.isoformat(),
            "source_counts": counts,
            "prepared_records": len(master),
            "output": str(output),
        }
        self.workspace.path("collection_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.logger.info("Prepared %s master record(s) at %s", len(master), output)
        return CollectionResult(master, output, counts)
