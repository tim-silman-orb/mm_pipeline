from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import boto3

from .config import OutputConfig


class Storage(Protocol):
    """Persistence contract used by the pipeline."""

    def save_run(
        self,
        *,
        project_id: str,
        run_id: str,
        datasets: dict[str, list[dict[str, Any]]],
        manifest: dict[str, Any],
    ) -> dict[str, str]:
        ...


class LocalStorage:
    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path).expanduser().resolve()

    def save_run(
        self,
        *,
        project_id: str,
        run_id: str,
        datasets: dict[str, list[dict[str, Any]]],
        manifest: dict[str, Any],
    ) -> dict[str, str]:
        run_dir = self.base_path / project_id / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        locations: dict[str, str] = {}
        for name, rows in datasets.items():
            path = run_dir / f"{name}.jsonl"
            text = "".join(
                json.dumps(row, ensure_ascii=False, default=str) + "\n"
                for row in rows
            )
            _atomic_write(path, text)
            locations[name] = str(path)
        manifest_path = run_dir / "manifest.json"
        _atomic_write(
            manifest_path,
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        )
        locations["manifest"] = str(manifest_path)
        return locations


class S3Storage:
    def __init__(self, base_uri: str, *, client: Any | None = None) -> None:
        parsed = urlparse(base_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"Invalid S3 URI: {base_uri}")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.client = client or boto3.client("s3")

    def save_run(
        self,
        *,
        project_id: str,
        run_id: str,
        datasets: dict[str, list[dict[str, Any]]],
        manifest: dict[str, Any],
    ) -> dict[str, str]:
        base_key = "/".join(
            part for part in (self.prefix, project_id, run_id) if part
        )
        locations: dict[str, str] = {}
        for name, rows in datasets.items():
            key = f"{base_key}/{name}.jsonl"
            body = "".join(
                json.dumps(row, ensure_ascii=False, default=str) + "\n"
                for row in rows
            )
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body.encode("utf-8"),
                ContentType="application/x-ndjson",
            )
            locations[name] = f"s3://{self.bucket}/{key}"
        manifest_key = f"{base_key}/manifest.json"
        self.client.put_object(
            Bucket=self.bucket,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        locations["manifest"] = f"s3://{self.bucket}/{manifest_key}"
        return locations


def create_storage(config: OutputConfig) -> Storage:
    if config.backend == "s3":
        return S3Storage(config.base_uri)
    return LocalStorage(config.base_uri)


def _atomic_write(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
