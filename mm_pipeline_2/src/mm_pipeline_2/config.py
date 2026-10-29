from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    """Validated project configuration with paths resolved from the YAML file."""

    config_path: Path
    project: str
    project_root: Path
    artifact_uri: str
    aws: dict[str, Any]
    sources: dict[str, dict[str, Any]]
    topic_filter: dict[str, Any]
    extraction: dict[str, Any]
    clustering: dict[str, Any]

    def source(self, name: str) -> dict[str, Any]:
        if name not in self.sources:
            raise KeyError(f"Unknown source configuration: {name}")
        return self.sources[name]

    def project_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("The top-level YAML value must be a mapping")
    if raw.get("schema_version") != 1:
        raise ValueError("Only schema_version 1 is supported")

    required = ["project", "project_root", "artifact_uri", "sources", "extraction", "clustering"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing configuration keys: {missing}")

    project_root = Path(str(raw["project_root"])).expanduser()
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()
    if not (project_root / "config").is_dir():
        raise FileNotFoundError(f"Project config directory not found: {project_root / 'config'}")

    sources = raw["sources"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("sources must be a non-empty mapping")

    return PipelineConfig(
        config_path=config_path,
        project=str(raw["project"]),
        project_root=project_root,
        artifact_uri=str(raw["artifact_uri"]),
        aws=dict(raw.get("aws", {})),
        sources={str(key): dict(value or {}) for key, value in sources.items()},
        topic_filter=dict(raw.get("topic_filter", {})),
        extraction=dict(raw["extraction"]),
        clustering=dict(raw["clustering"]),
    )
