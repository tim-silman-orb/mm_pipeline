from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .cleanup import CanonicalStoryCleaner
from .collection import DataCollectionPipeline, RunWindow
from .config import PipelineConfig
from .workspace import ArtifactWorkspace


def create_bedrock_runtime(config: PipelineConfig) -> Any:
    import boto3

    region = os.getenv("AWS_REGION", str(config.aws.get("region", "eu-west-2")))
    profile_env = str(config.aws.get("profile_env", "AWS_PROFILE"))
    profile = os.getenv(profile_env, "").strip()
    session_kwargs: dict[str, Any] = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
    return boto3.Session(**session_kwargs).client("bedrock-runtime")


class PipelineStage:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        workspace: ArtifactWorkspace | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.workspace = workspace or ArtifactWorkspace(
            config.artifact_uri,
            project_root=config.project_root,
            logger=self.logger.getChild("workspace"),
        )


class CollectionStage(PipelineStage):
    def run(
        self,
        *,
        window: RunWindow,
        selected_sources: set[str] | None = None,
        topic_terms: list[str] | None = None,
        sync: bool = True,
    ) -> pd.DataFrame:
        if sync:
            self.workspace.prepare()
        result = DataCollectionPipeline(
            config=self.config,
            workspace=self.workspace,
            logger=self.logger,
        ).run(
            window=window,
            selected_sources=selected_sources,
            topic_terms=topic_terms,
        )
        if sync:
            self.workspace.publish()
        return result.dataframe


class ExtractionStage(PipelineStage):
    def run(
        self,
        *,
        limit: int | None = None,
        bedrock_runtime: Any | None = None,
        sync: bool = True,
    ) -> pd.DataFrame:
        if sync:
            self.workspace.prepare()
        from .backend.extraction_functions import ExtractionConfig, run_extraction

        settings = self.config.extraction
        master_file = self.workspace.path(str(settings.get("master_data_file", "master_data.jsonl")))
        if not master_file.exists():
            raise FileNotFoundError(f"Master dataset not found: {master_file}")
        articles = pd.read_json(master_file, lines=True, dtype=False).fillna("")
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be at least 1")
            articles = articles.head(limit).copy()

        model_env = str(settings.get("model_id_env", "EXTRACTION_MODEL_ID"))
        model_default = str(settings.get("model_id", "amazon.nova-lite-v1:0"))
        model_id = os.getenv(model_env, "").strip() or model_default
        legacy_config = ExtractionConfig(
            input_file=master_file,
            extraction_file=self.workspace.path(str(settings["extraction_file"])),
            article_log_file=self.workspace.path(str(settings["article_log_file"])),
            article_id_column="article_id",
            headline_column="headline",
            body_column="analysis_text",
            date_column="published_at",
            source_column="source_name",
            language_column="language",
            url_column="url",
            model_id=model_id,
            number_of_articles=limit,
            max_article_characters=int(settings.get("max_article_characters", 30_000)),
            request_delay_seconds=float(settings.get("request_delay_seconds", 0.5)),
            max_retries=int(settings.get("max_retries", 4)),
        )
        self.logger.info("Extracting developments from %s article(s)", len(articles))
        result = run_extraction(
            articles_df=articles,
            config=legacy_config,
            bedrock_runtime=bedrock_runtime or create_bedrock_runtime(self.config),
        )
        if sync:
            self.workspace.publish()
        return result


class ConversationClassifier:
    @staticmethod
    def classify(row: pd.Series) -> str:
        platform = str(row.get("platform", "")).strip().lower()
        article_id = str(row.get("article_id", "")).strip().lower()
        source_name = str(row.get("source_name", "")).strip().lower()
        url = str(row.get("url", "")).strip().lower()
        if platform == "opoint":
            conversation = "news"
        elif platform in {"x", "telegram"}:
            conversation = "social"
        elif platform == "health_source":
            conversation = "public_health"
        elif article_id.startswith("opoint_"):
            conversation = "news"
        elif article_id.startswith(("x_", "telegram_")):
            conversation = "social"
        elif article_id.startswith("health_source_"):
            conversation = "public_health"
        else:
            conversation = "unclassified"
        if "google news" in source_name or "news.google.com" in url or "news.google.com" in article_id:
            conversation = "news"
        return conversation

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        result = dataframe.copy()
        result["conversation_type"] = result.apply(self.classify, axis=1)
        return result


@dataclass(frozen=True)
class ClusteringResult:
    grouped_streams: dict[str, pd.DataFrame]
    summary_streams: dict[str, pd.DataFrame]
    combined_summary_file: Path


class ClusteringStage(PipelineStage):
    def run(
        self,
        *,
        selected_streams: set[str] | None = None,
        bedrock_runtime: Any | None = None,
        sync: bool = True,
    ) -> ClusteringResult:
        if sync:
            self.workspace.prepare()
        from .backend.cluster_functions import ClusterConfig, run_clustering

        settings = self.config.clustering
        input_file = self.workspace.path(str(settings["extracted_input_file"]))
        if not input_file.exists():
            raise FileNotFoundError(f"Extracted developments not found: {input_file}")
        developments = pd.read_csv(input_file, dtype=str).fillna("")
        developments = ConversationClassifier().apply(developments)

        stream_settings = dict(settings.get("streams", {}))
        if selected_streams:
            unknown = selected_streams - set(stream_settings)
            if unknown:
                raise ValueError(f"Unknown stream(s): {sorted(unknown)}")
        output_root = self.workspace.path(str(settings.get("output_dir", "clustering/streams")))
        output_root.mkdir(parents=True, exist_ok=True)
        runtime = bedrock_runtime or create_bedrock_runtime(self.config)
        grouping_env = str(settings.get("grouping_model_id_env", "GROUPING_MODEL_ID"))
        embedding_env = str(settings.get("embedding_model_id_env", "EMBEDDING_MODEL_ID"))
        grouping_model = os.getenv(grouping_env, "").strip() or str(settings["grouping_model_id"])
        embedding_model = os.getenv(embedding_env, "").strip() or str(settings["embedding_model_id"])

        cleanup_settings = dict(settings.get("cleanup", {}))
        cleanup_enabled = bool(cleanup_settings.get("enabled", False))
        cleanup_streams = set(cleanup_settings.get("streams", []))
        cleanup_env = str(cleanup_settings.get("model_id_env", "CLEANUP_MODEL_ID"))
        cleanup_default = str(cleanup_settings.get("model_id", "")) or grouping_model
        cleaner = CanonicalStoryCleaner(
            output_root=output_root,
            bedrock_runtime=runtime,
            model_id=os.getenv(cleanup_env, "").strip() or cleanup_default,
            candidate_distance=float(cleanup_settings.get("candidate_distance", 0.40)),
            max_batch_size=int(cleanup_settings.get("max_batch_size", 30)),
            passes=int(cleanup_settings.get("passes", 2)),
            max_retries=int(settings.get("max_retries", 4)),
            logger=self.logger.getChild("cleanup"),
        )

        grouped_streams: dict[str, pd.DataFrame] = {}
        raw_summaries: dict[str, pd.DataFrame] = {}
        final_summaries: dict[str, pd.DataFrame] = {}
        for stream, stream_config in stream_settings.items():
            enabled = bool((stream_config or {}).get("enabled", False))
            if selected_streams is not None:
                enabled = stream in selected_streams
            if not enabled:
                continue
            stream_df = developments[developments["conversation_type"].eq(stream)].copy()
            if stream_df.empty:
                self.logger.warning("Skipping %s: no developments", stream)
                continue
            stream_dir = output_root / stream
            stream_dir.mkdir(parents=True, exist_ok=True)
            cluster_config = ClusterConfig(
                extracted_input_file=input_file,
                grouped_developments_file=stream_dir / f"{stream}_grouped_developments.csv",
                group_summaries_file=stream_dir / f"{stream}_group_summaries.csv",
                embedding_cache_file=stream_dir / f"{stream}_embeddings.npz",
                embedding_cache_ids=stream_dir / f"{stream}_embedding_ids.json",
                grouping_model_id=grouping_model,
                embedding_model_id=embedding_model,
                conversation_type=stream,
                embedding_distance_threshold=float(settings.get("embedding_distance_threshold", 0.25)),
                max_llm_cluster_size=int(settings.get("max_llm_cluster_size", 25)),
                max_retries=int(settings.get("max_retries", 4)),
                request_delay_seconds=float(settings.get("request_delay_seconds", 0.5)),
                embedding_request_delay_seconds=float(settings.get("embedding_request_delay_seconds", 0.05)),
                embedding_dimensions=int(settings.get("embedding_dimensions", 1024)),
            )
            self.logger.info("Clustering %s developments for %s", len(stream_df), stream)
            grouped, summaries = run_clustering(
                config=cluster_config,
                bedrock_runtime=runtime,
                developments_df=stream_df,
                save_outputs=True,
            )
            grouped_streams[stream] = grouped
            raw_summaries[stream] = summaries
            if cleanup_enabled and stream in cleanup_streams and not summaries.empty:
                cleaned_summaries, cleaned_grouped, _ = cleaner.clean(
                    stream=stream,
                    summaries=summaries,
                    grouped=grouped,
                    cluster_config=cluster_config,
                )
                final_summaries[stream] = cleaned_summaries
                grouped_streams[stream] = cleaned_grouped
            else:
                final_summaries[stream] = summaries

        raw_combined = pd.concat(list(raw_summaries.values()), ignore_index=True) if raw_summaries else pd.DataFrame()
        final_combined = pd.concat(list(final_summaries.values()), ignore_index=True) if final_summaries else pd.DataFrame()
        raw_file = output_root / "conversation_group_summaries_raw_stage2.csv"
        combined_file = output_root / "conversation_group_summaries.csv"
        raw_combined.to_csv(raw_file, index=False, encoding="utf-8-sig")
        final_combined.to_csv(combined_file, index=False, encoding="utf-8-sig")
        self.logger.info("Saved %s final story group(s) to %s", len(final_combined), combined_file)
        if sync:
            self.workspace.publish()
        return ClusteringResult(grouped_streams, final_summaries, combined_file)


class FullPipeline(PipelineStage):
    def run(
        self,
        *,
        window: RunWindow,
        selected_sources: set[str] | None = None,
        topic_terms: list[str] | None = None,
        extraction_limit: int | None = None,
        selected_streams: set[str] | None = None,
    ) -> ClusteringResult:
        self.workspace.prepare()
        runtime = create_bedrock_runtime(self.config)
        CollectionStage(
            self.config,
            workspace=self.workspace,
            logger=self.logger.getChild("collect"),
        ).run(
            window=window,
            selected_sources=selected_sources,
            topic_terms=topic_terms,
            sync=False,
        )
        self.workspace.publish()
        ExtractionStage(
            self.config,
            workspace=self.workspace,
            logger=self.logger.getChild("extract"),
        ).run(limit=extraction_limit, bedrock_runtime=runtime, sync=False)
        self.workspace.publish()
        result = ClusteringStage(
            self.config,
            workspace=self.workspace,
            logger=self.logger.getChild("cluster"),
        ).run(selected_streams=selected_streams, bedrock_runtime=runtime, sync=False)
        self.workspace.publish()
        return result
