from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml
from dateutil import parser as date_parser
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .exceptions import ConfigurationError
from .models import CollectionWindow


class ProjectInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    timezone: str = "UTC"


class WindowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_lookback_days: int = Field(default=7, ge=1, le=3650)


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    adapter: str = Field(min_length=1)
    name: str = ""
    platform: str = ""
    enabled: bool = True
    required: bool = True
    url: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def populate_display_fields(self) -> "SourceConfig":
        if not self.name:
            self.name = self.id
        if not self.platform:
            self.platform = self.adapter
        return self


class TopicFilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    mode: Literal["literal_any", "literal_all", "regex"] = "literal_any"
    fields: list[str] = Field(default_factory=lambda: ["headline", "content", "summary"])
    terms: list[str] = Field(default_factory=list)
    apply_to_platforms: list[str] = Field(default_factory=list)


class ExtractionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider: Literal["bedrock"] = "bedrock"
    model_id: str = "amazon.nova-lite-v1:0"
    max_article_characters: int = Field(default=30_000, ge=1000)
    max_retries: int = Field(default=4, ge=1, le=10)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["local", "s3"] = "local"
    base_uri: str = "output"


class FailurePolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fail_on_required_source: bool = True
    fail_on_empty_dataset: bool = True


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    project: ProjectInfo
    window: WindowConfig = Field(default_factory=WindowConfig)
    sources: list[SourceConfig]
    topic_filter: TopicFilterConfig = Field(default_factory=TopicFilterConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    failure_policy: FailurePolicyConfig = Field(default_factory=FailurePolicyConfig)
    config_path: Path | None = Field(default=None, exclude=True)

    @property
    def enabled_sources(self) -> list[SourceConfig]:
        return [source for source in self.sources if source.enabled]

    @model_validator(mode="after")
    def require_unique_source_ids(self) -> "PipelineConfig":
        ids = [source.id for source in self.sources]
        duplicates = sorted({source_id for source_id in ids if ids.count(source_id) > 1})
        if duplicates:
            raise ValueError(f"Duplicate source ID(s): {', '.join(duplicates)}")
        return self


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Top-level YAML value must be a mapping: {path}")
    return payload


def _resolve_sources(raw: dict[str, Any], config_path: Path) -> list[dict[str, Any]]:
    catalog_entries: dict[str, Any] = {}
    catalog_value = raw.pop("source_catalog", None)
    if catalog_value:
        catalog_path = Path(catalog_value)
        if not catalog_path.is_absolute():
            catalog_path = (config_path.parent / catalog_path).resolve()
        catalog_entries = _read_yaml(catalog_path).get("sources", {})
        if not isinstance(catalog_entries, dict):
            raise ConfigurationError("source_catalog.yml 'sources' must be a mapping")

    resolved: list[dict[str, Any]] = []
    for entry in raw.get("sources", []):
        if not isinstance(entry, dict):
            raise ConfigurationError("Every source entry must be a mapping")
        if "ref" not in entry:
            resolved.append(entry)
            continue
        reference = str(entry["ref"])
        if reference not in catalog_entries:
            raise ConfigurationError(f"Unknown source catalog reference: {reference}")
        merged = deepcopy(catalog_entries[reference])
        merged.setdefault("id", reference)
        overrides = {key: value for key, value in entry.items() if key != "ref"}
        if "options" in overrides:
            merged["options"] = {**merged.get("options", {}), **overrides.pop("options")}
        merged.update(overrides)
        resolved.append(merged)
    return resolved


def load_project_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    raw = _read_yaml(config_path)
    raw["sources"] = _resolve_sources(raw, config_path)
    try:
        config = PipelineConfig.model_validate(raw)
    except Exception as exc:  # Pydantic provides the detailed field errors.
        raise ConfigurationError(f"Invalid project configuration: {exc}") from exc
    config.config_path = config_path
    return config


def apply_overrides(
    config: PipelineConfig,
    *,
    selected_sources: list[str] | None = None,
    output_uri: str | None = None,
) -> PipelineConfig:
    updated = config.model_copy(deep=True)
    if selected_sources:
        requested = set(selected_sources)
        known = {source.id for source in updated.sources}
        unknown = sorted(requested - known)
        if unknown:
            raise ConfigurationError(f"Unknown --source value(s): {', '.join(unknown)}")
        for source in updated.sources:
            source.enabled = source.id in requested
    if output_uri:
        updated.output.base_uri = output_uri
        updated.output.backend = "s3" if output_uri.startswith("s3://") else "local"
    return updated


def resolve_window(
    config: PipelineConfig,
    *,
    start: str | None = None,
    end: str | None = None,
    lookback_days: int | None = None,
    now: datetime | None = None,
) -> CollectionWindow:
    def parse_utc(value: str) -> datetime:
        parsed = date_parser.isoparse(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    if start and not end:
        raise ConfigurationError("--start requires --end")
    if start and lookback_days is not None:
        raise ConfigurationError("Use either --start/--end or --lookback-days, not both")

    end_time = parse_utc(end) if end else (now or datetime.now(UTC)).astimezone(UTC)
    if start:
        start_time = parse_utc(start)
    else:
        days = lookback_days or config.window.default_lookback_days
        if days < 1:
            raise ConfigurationError("lookback days must be at least 1")
        start_time = end_time - timedelta(days=days)
    return CollectionWindow(start=start_time, end=end_time)
