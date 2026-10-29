from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [text]
    except (json.JSONDecodeError, TypeError):
        return [text]


def unique_clean(values: Iterable[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


class CanonicalStoryCleaner:
    """Stage 3 notebook cleanup expressed as a reusable, stateful service."""

    def __init__(
        self,
        *,
        output_root: Path,
        bedrock_runtime: Any,
        model_id: str,
        candidate_distance: float = 0.40,
        max_batch_size: int = 30,
        passes: int = 2,
        max_retries: int = 4,
        logger: logging.Logger,
    ) -> None:
        self.output_root = output_root
        self.bedrock_runtime = bedrock_runtime
        self.model_id = model_id
        self.candidate_distance = candidate_distance
        self.max_batch_size = max_batch_size
        self.passes = passes
        self.max_retries = max_retries
        self.logger = logger

    def clean(
        self,
        *,
        stream: str,
        summaries: pd.DataFrame,
        grouped: pd.DataFrame,
        cluster_config: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        current = summaries.copy().reset_index(drop=True)
        current["current_group_id"] = current["group_id"].astype(str)
        current["original_group_ids"] = current["group_id"].map(lambda value: [str(value)])

        for pass_number in range(1, self.passes + 1):
            before = len(current)
            current = self._process_pass(
                stream=stream,
                current=current,
                cluster_config=cluster_config,
                pass_number=pass_number,
            )
            if len(current) == before:
                self.logger.info("%s cleanup stable after pass %s", stream, pass_number)
                break

        cleaned_summaries = self._aggregate_summaries(stream, summaries, current)
        cleaned_grouped, mapping = self._map_developments(grouped, current)
        stream_dir = self.output_root / stream
        stream_dir.mkdir(parents=True, exist_ok=True)
        cleaned_summaries.to_csv(
            stream_dir / f"{stream}_cleaned_story_summaries.csv",
            index=False,
            encoding="utf-8-sig",
        )
        cleaned_grouped.to_csv(
            stream_dir / f"{stream}_cleaned_grouped_developments.csv",
            index=False,
            encoding="utf-8-sig",
        )
        mapping.to_csv(
            stream_dir / f"{stream}_canonical_story_map.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return cleaned_summaries, cleaned_grouped, mapping

    def _group_text(self, row: pd.Series) -> str:
        values = [
            row.get("group_name", ""),
            row.get("group_summary", ""),
            row.get("event_types", ""),
            row.get("locations", ""),
        ]
        return "\n".join(str(value).strip() for value in values if str(value).strip())

    def _embed_groups(
        self,
        *,
        stream: str,
        groups: pd.DataFrame,
        cluster_config: Any,
        pass_number: int,
    ) -> np.ndarray:
        from .backend.cluster_functions import call_bedrock_embedding

        texts = [self._group_text(row) for _, row in groups.iterrows()]
        keys = [
            f"{row['current_group_id']}|{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
            for (_, row), text in zip(groups.iterrows(), texts, strict=True)
        ]
        stream_dir = self.output_root / stream
        stream_dir.mkdir(parents=True, exist_ok=True)
        cache_file = stream_dir / f"{stream}_cleanup_pass_{pass_number}_embeddings.npz"
        key_file = stream_dir / f"{stream}_cleanup_pass_{pass_number}_embedding_keys.json"
        if cache_file.exists() and key_file.exists():
            try:
                cached_keys = json.loads(key_file.read_text(encoding="utf-8"))
                cached = np.load(cache_file)["embeddings"]
                if cached_keys == keys and len(cached) == len(groups):
                    self.logger.info("Using cached cleanup embeddings for %s pass %s", stream, pass_number)
                    return cached
            except Exception as exc:  # Cache corruption should not fail the run.
                self.logger.warning("Ignoring cleanup embedding cache: %s", exc)

        embeddings: list[np.ndarray] = []
        for number, text in enumerate(texts, start=1):
            if number == 1 or number % 25 == 0 or number == len(texts):
                self.logger.info("Embedding cleanup group %s/%s", number, len(texts))
            embeddings.append(
                call_bedrock_embedding(
                    config=cluster_config,
                    bedrock_runtime=self.bedrock_runtime,
                    text=text,
                )
            )
            time.sleep(cluster_config.embedding_request_delay_seconds)
        matrix = np.vstack(embeddings)
        np.savez_compressed(cache_file, embeddings=matrix)
        key_file.write_text(json.dumps(keys, indent=2), encoding="utf-8")
        return matrix

    @staticmethod
    def _labels(
        embeddings: np.ndarray,
        *,
        distance_threshold: float | None = None,
        n_clusters: int | None = None,
    ) -> np.ndarray:
        from sklearn.cluster import AgglomerativeClustering

        kwargs: dict[str, Any] = {"linkage": "average"}
        if n_clusters is None:
            kwargs.update({"n_clusters": None, "distance_threshold": distance_threshold})
        else:
            kwargs["n_clusters"] = n_clusters
        try:
            model = AgglomerativeClustering(metric="cosine", **kwargs)
        except TypeError:
            model = AgglomerativeClustering(affinity="cosine", **kwargs)
        return model.fit_predict(embeddings)

    def _split_indices(self, indices: list[int], embeddings: np.ndarray) -> list[list[int]]:
        if len(indices) <= self.max_batch_size:
            return [indices]
        subset = embeddings[indices]
        labels = self._labels(
            subset,
            n_clusters=math.ceil(len(indices) / self.max_batch_size),
        )
        return [
            [indices[position] for position in np.where(labels == label)[0]]
            for label in sorted(set(labels))
        ]

    def _prompt(self, batch: pd.DataFrame, *, stream: str, pass_number: int) -> str:
        stories = [
            {
                "story_group_id": str(row["current_group_id"]),
                "group_name": str(row.get("group_name", "")),
                "group_summary": str(row.get("group_summary", "")),
                "development_count": int(row.get("development_count", 0) or 0),
                "article_count": int(row.get("article_count", 0) or 0),
                "event_types": parse_json_list(row.get("event_types", "")),
                "locations": parse_json_list(row.get("locations", "")),
            }
            for _, row in batch.iterrows()
        ]
        return f"""
You are performing a conservative cleanup of already-good Ebola {stream} story clusters.

These are existing story groups, not raw articles. Some groups are duplicate or near-duplicate
descriptions of the SAME underlying event. Merge those. Preserve groups that are merely related.

Candidate story groups:
{json.dumps(stories, ensure_ascii=False, indent=2)}

Return valid JSON only:
{{
  "canonical_stories": [
    {{
      "canonical_story_name": "Concise factual name for the underlying story",
      "canonical_story_summary": "One or two sentences describing the shared event",
      "member_group_ids": ["existing_group_id"],
      "merge_confidence": "high, medium, or low",
      "merge_reason": "Brief reason these groups are the same story, or why the singleton is retained"
    }}
  ]
}}

Rules:
1. Every supplied story_group_id must appear exactly once.
2. Never invent a story_group_id.
3. Merge only when groups describe the same underlying event/development.
4. Different wording, outlet angle, or slightly different numerical phrasing can still be the same story.
5. Do not merge merely because groups share the same outbreak, country, organisation, or broad theme.
6. Separate case updates, border responses, vaccination, risk assessments, treatment, and security incidents
   unless they genuinely describe the same event.
7. When uncertain, keep groups separate.
8. Prefer a neutral canonical story name over sensational wording.
9. This is cleanup pass {pass_number}; precision matters more than aggressive merging.
""".strip()

    @staticmethod
    def _singletons(batch: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {
                "canonical_story_name": str(row.get("group_name", "")),
                "canonical_story_summary": str(row.get("group_summary", "")),
                "member_group_ids": [str(row["current_group_id"])],
                "merge_confidence": "low",
                "merge_reason": "Retained as a singleton because cleanup was not applied or validated.",
            }
            for _, row in batch.iterrows()
        ]

    def _validated_stories(self, result: dict[str, Any], batch: pd.DataFrame) -> list[dict[str, Any]]:
        expected = batch["current_group_id"].astype(str).tolist()
        stories = result.get("canonical_stories", [])
        if not isinstance(stories, list):
            return self._singletons(batch)
        observed: list[str] = []
        for story in stories:
            members = story.get("member_group_ids", [])
            if not isinstance(members, list):
                return self._singletons(batch)
            members = [str(value) for value in members]
            if not members or not set(members).issubset(set(expected)):
                return self._singletons(batch)
            observed.extend(members)
        if len(observed) != len(expected) or len(set(observed)) != len(observed) or set(observed) != set(expected):
            return self._singletons(batch)
        return stories

    def _process_pass(
        self,
        *,
        stream: str,
        current: pd.DataFrame,
        cluster_config: Any,
        pass_number: int,
    ) -> pd.DataFrame:
        if len(current) <= 1:
            return current.copy()
        from .backend.common import call_bedrock_json

        embeddings = self._embed_groups(
            stream=stream,
            groups=current,
            cluster_config=cluster_config,
            pass_number=pass_number,
        )
        working = current.copy().reset_index(drop=True)
        working["_candidate_label"] = self._labels(
            embeddings,
            distance_threshold=self.candidate_distance,
        )
        output: list[dict[str, Any]] = []
        for _, candidate in working.groupby("_candidate_label", sort=True):
            for indices in self._split_indices(candidate.index.tolist(), embeddings):
                batch = working.loc[indices].copy()
                if len(batch) == 1:
                    stories = self._singletons(batch)
                else:
                    try:
                        response = call_bedrock_json(
                            self.bedrock_runtime,
                            prompt=self._prompt(batch, stream=stream, pass_number=pass_number),
                            model_id=self.model_id,
                            max_tokens=5000,
                            max_retries=self.max_retries,
                            operation_name=f"{stream} canonical-story cleanup",
                        )
                        stories = self._validated_stories(response, batch)
                    except Exception as exc:
                        self.logger.warning("Cleanup call failed; retaining singletons: %s", exc)
                        stories = self._singletons(batch)

                lookup = batch.set_index("current_group_id", drop=False).to_dict("index")
                for story in stories:
                    member_ids = [str(value) for value in story["member_group_ids"]]
                    member_rows = [lookup[value] for value in member_ids]
                    originals = unique_clean(
                        original
                        for row in member_rows
                        for original in row["original_group_ids"]
                    )
                    canonical_id = f"{stream}_story_{hashlib.sha1('|'.join(originals).encode()).hexdigest()[:12]}"
                    output.append(
                        {
                            "current_group_id": canonical_id,
                            "group_id": canonical_id,
                            "group_name": str(story.get("canonical_story_name", "")).strip(),
                            "group_summary": str(story.get("canonical_story_summary", "")).strip(),
                            "development_count": sum(int(row.get("development_count", 0) or 0) for row in member_rows),
                            "article_count": sum(int(row.get("article_count", 0) or 0) for row in member_rows),
                            "original_group_ids": originals,
                            "cleanup_merge_confidence": str(story.get("merge_confidence", "")).strip(),
                            "cleanup_merge_reason": str(story.get("merge_reason", "")).strip(),
                        }
                    )
        result = pd.DataFrame(output).reset_index(drop=True)
        saved = result.copy()
        saved["original_group_ids"] = saved["original_group_ids"].map(
            lambda values: json.dumps(values, ensure_ascii=False)
        )
        output_file = self.output_root / stream / f"{stream}_cleanup_pass_{pass_number}.csv"
        saved.to_csv(output_file, index=False, encoding="utf-8-sig")
        self.logger.info("%s cleanup pass %s: %s -> %s groups", stream, pass_number, len(current), len(result))
        return result

    @staticmethod
    def _importance(values: Iterable[Any]) -> str:
        order = {"high": 0, "medium": 1, "low": 2}
        cleaned = [str(value).strip().lower() for value in values if str(value).strip()]
        return min(cleaned, key=lambda value: order.get(value, 99)) if cleaned else ""

    def _aggregate_summaries(
        self,
        stream: str,
        original: pd.DataFrame,
        final_groups: pd.DataFrame,
    ) -> pd.DataFrame:
        lookup = original.set_index("group_id", drop=False).to_dict("index")
        rows: list[dict[str, Any]] = []
        for _, final in final_groups.iterrows():
            member_ids = final["original_group_ids"]
            members = [lookup[value] for value in member_ids if value in lookup]

            def merged(field: str) -> list[str]:
                return unique_clean(
                    item for row in members for item in parse_json_list(row.get(field, ""))
                )

            article_ids = merged("article_ids")
            sources = merged("sources")
            event_types = merged("event_types")
            locations = merged("locations")
            development_ids = merged("development_ids")
            developments = merged("developments")
            development_count = sum(int(row.get("development_count", 0) or 0) for row in members)
            rows.append(
                {
                    "conversation_type": stream,
                    "group_id": final["current_group_id"],
                    "group_name": final["group_name"],
                    "group_summary": final["group_summary"],
                    "group_type": "singleton" if development_count <= 1 else "recurring",
                    "importance": self._importance(row.get("importance", "") for row in members),
                    "development_count": development_count,
                    "article_count": len(article_ids),
                    "source_count": len(sources),
                    "sources": json.dumps(sources, ensure_ascii=False),
                    "event_types": json.dumps(event_types, ensure_ascii=False),
                    "locations": json.dumps(locations, ensure_ascii=False),
                    "development_ids": json.dumps(development_ids, ensure_ascii=False),
                    "article_ids": json.dumps(article_ids, ensure_ascii=False),
                    "developments": json.dumps(developments, ensure_ascii=False),
                    "member_group_ids": json.dumps(member_ids, ensure_ascii=False),
                    "member_group_count": len(member_ids),
                    "cleanup_merge_confidence": final.get("cleanup_merge_confidence", ""),
                    "cleanup_merge_reason": final.get("cleanup_merge_reason", ""),
                }
            )
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(
                ["development_count", "article_count"], ascending=[False, False]
            ).reset_index(drop=True)
        return result

    @staticmethod
    def _map_developments(
        grouped: pd.DataFrame,
        final_groups: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows = [
            {
                "provisional_group_id": original,
                "canonical_story_id": final["current_group_id"],
                "canonical_story_name": final["group_name"],
                "canonical_story_summary": final["group_summary"],
                "cleanup_merge_confidence": final.get("cleanup_merge_confidence", ""),
                "cleanup_merge_reason": final.get("cleanup_merge_reason", ""),
            }
            for _, final in final_groups.iterrows()
            for original in final["original_group_ids"]
        ]
        mapping = pd.DataFrame(rows)
        cleaned = grouped.merge(
            mapping,
            on="provisional_group_id",
            how="left",
            validate="many_to_one",
        )
        return cleaned, mapping
