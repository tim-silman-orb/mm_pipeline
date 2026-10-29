from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse


class ArtifactWorkspace:
    """Local working directory with optional whole-workspace S3 synchronisation."""

    def __init__(self, uri: str, *, project_root: Path, logger: logging.Logger) -> None:
        self.uri = uri
        self.project_root = project_root
        self.logger = logger
        parsed = urlparse(uri)
        self.is_s3 = parsed.scheme == "s3"
        if self.is_s3:
            if not parsed.netloc:
                raise ValueError(f"Invalid S3 artifact URI: {uri}")
            self.bucket = parsed.netloc
            self.prefix = parsed.path.strip("/")
            configured = os.getenv("MM_PIPELINE_WORKDIR", "").strip()
            self.root = (
                Path(configured).expanduser().resolve()
                if configured
                else Path(tempfile.gettempdir()) / "mm_pipeline_2" / self.prefix.replace("/", "_")
            )
        else:
            path = Path(uri).expanduser()
            self.root = path.resolve() if path.is_absolute() else (project_root / path).resolve()
            self.bucket = ""
            self.prefix = ""
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative_path: str | Path) -> Path:
        path = (self.root / relative_path).resolve()
        if self.root != path and self.root not in path.parents:
            raise ValueError(f"Artifact path escapes the workspace: {relative_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def prepare(self) -> None:
        if not self.is_s3:
            return
        import boto3

        client = boto3.client("s3")
        paginator = client.get_paginator("list_objects_v2")
        downloaded = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for item in page.get("Contents", []):
                key = str(item["Key"])
                relative = key[len(self.prefix) :].lstrip("/") if self.prefix else key
                if not relative or key.endswith("/"):
                    continue
                destination = self.path(relative)
                client.download_file(self.bucket, key, str(destination))
                downloaded += 1
        self.logger.info("Downloaded %s artifact(s) from %s", downloaded, self.uri)

    def publish(self) -> None:
        if not self.is_s3:
            return
        import boto3

        client = boto3.client("s3")
        uploaded = 0
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root).as_posix()
            key = "/".join(part for part in (self.prefix, relative) if part)
            client.upload_file(str(path), self.bucket, key)
            uploaded += 1
        self.logger.info("Uploaded %s artifact(s) to %s", uploaded, self.uri)

