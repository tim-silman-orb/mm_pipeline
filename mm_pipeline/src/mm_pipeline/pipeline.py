from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any

from .config import PipelineConfig
from .extraction import BedrockDevelopmentExtractor
from .models import CollectionWindow, PipelineResult, SourceFailure, SourceResult
from .registry import create_adapter
from .storage import Storage, create_storage
from .transforms import (
    apply_topic_filter,
    article_rows,
    deduplicate_articles,
    discard_unusable_articles,
    validate_articles,
)


AdapterFactory = Callable[[Any, logging.Logger], Any]


def run_pipeline(
    config: PipelineConfig,
    *,
    window: CollectionWindow,
    logger: logging.Logger | None = None,
    adapter_factory: AdapterFactory = create_adapter,
    storage: Storage | None = None,
    extractor: Any | None = None,
) -> PipelineResult:
    logger = logger or logging.getLogger("mm_pipeline")
    run_id = _run_id(config.project.id, window)
    source_results: list[SourceResult] = []
    failures: list[SourceFailure] = []

    for source in config.enabled_sources:
        source_logger = logger.getChild(f"source.{source.id}")
        source_logger.info("Collecting with adapter %s", source.adapter)
        try:
            result = adapter_factory(source, source_logger).collect(window)
            source_results.append(result)
            source_logger.info("Collected %s record(s)", len(result.records))
        except Exception as exc:
            source_logger.exception("Source collection failed")
            failures.append(SourceFailure(source.id, source.required, str(exc)))

    collected = [article for result in source_results for article in result.records]
    filtered = apply_topic_filter(collected, config.topic_filter)
    deduplicated = deduplicate_articles(filtered)
    articles = discard_unusable_articles(deduplicated)
    validate_articles(articles)

    fatal_errors: list[str] = []
    required_failures = [failure for failure in failures if failure.required]
    if config.failure_policy.fail_on_required_source and required_failures:
        fatal_errors.append(
            "Required source failure(s): "
            + ", ".join(failure.source_id for failure in required_failures)
        )
    if config.failure_policy.fail_on_empty_dataset and not articles:
        fatal_errors.append("Prepared master dataset is empty")

    datasets: dict[str, list[dict[str, Any]]] = {
        "master_data": article_rows(articles),
    }
    if config.extraction.enabled and articles and not fatal_errors:
        extractor = extractor or BedrockDevelopmentExtractor(config.extraction, logger)
        datasets["developments"] = extractor.extract(articles)

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "project": config.project.id,
        "window_start": window.start.isoformat(),
        "window_end": window.end.isoformat(),
        "config_hash": _config_hash(config),
        "source_counts": {
            result.source_id: len(result.records) for result in source_results
        },
        "records_collected": len(collected),
        "records_after_filter": len(filtered),
        "records_after_deduplication": len(deduplicated),
        "records_prepared": len(articles),
        "failures": [failure.__dict__ for failure in failures],
        "warnings": {
            result.source_id: result.warnings
            for result in source_results
            if result.warnings
        },
        "fatal_errors": fatal_errors,
        "status": "failed" if fatal_errors else "succeeded",
    }
    storage = storage or create_storage(config.output)
    locations = storage.save_run(
        project_id=config.project.id,
        run_id=run_id,
        datasets=datasets,
        manifest=manifest,
    )
    return PipelineResult(
        run_id=run_id,
        project_id=config.project.id,
        window=window,
        articles=articles,
        source_results=source_results,
        failures=failures,
        output_locations=locations,
        fatal_errors=fatal_errors,
    )


def _run_id(project_id: str, window: CollectionWindow) -> str:
    start = window.start.strftime("%Y%m%dT%H%M%SZ")
    end = window.end.strftime("%Y%m%dT%H%M%SZ")
    return f"{project_id}_{start}_{end}"


def _config_hash(config: PipelineConfig) -> str:
    text = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
