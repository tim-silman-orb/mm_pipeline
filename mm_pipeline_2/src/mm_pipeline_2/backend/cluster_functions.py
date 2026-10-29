from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from .common import call_bedrock_json, clean_value, safe_save_csv


@dataclass(frozen=True)
class ClusterConfig:
    extracted_input_file: Path
    grouped_developments_file: Path
    group_summaries_file: Path

    embedding_cache_file: Path
    embedding_cache_ids: Path

    grouping_model_id: str
    embedding_model_id: str

    # Optional stream guard. When set, only developments from this
    # conversation stream are clustered (e.g. news, social, public_health).
    conversation_type: Optional[str] = None

    embedding_distance_threshold: float = 0.25
    max_llm_cluster_size: int = 100
    max_retries: int = 4
    request_delay_seconds: float = 0.5
    embedding_request_delay_seconds: float = 0.05
    embedding_dimensions: int = 1024


# -----------------------------------------------------------------------------
# Loading / validation
# -----------------------------------------------------------------------------

def load_developments(
    config: ClusterConfig,
    developments_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load extracted Ebola developments from a DataFrame or CSV file."""

    if developments_df is None:
        df = pd.read_csv(config.extracted_input_file, dtype=str).fillna("")
    else:
        df = developments_df.copy().fillna("")

    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    required_columns = ["development_id", "normalised_development"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # Optional stream-specific clustering. This prevents news, social and
    # public-health developments from being mixed into the same story groups.
    if config.conversation_type:
        if "conversation_type" not in df.columns:
            raise KeyError(
                "conversation_type is required when ClusterConfig.conversation_type is set."
            )

        target_stream = clean_value(config.conversation_type).lower()
        stream_values = (
            df["conversation_type"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df = df[stream_values.eq(target_stream)].copy()

        print(
            f"Selected {len(df)} developments from conversation stream: "
            f"{target_stream}"
        )

    # Optional future human-review override, without requiring review today.
    if "include_in_grouping" in df.columns:
        include_values = df["include_in_grouping"].astype(str).str.strip().str.lower()
        if include_values.ne("").any():
            df = df[include_values.isin(["yes", "y", "true", "1"])].copy()

    if "corrected_normalised_development" in df.columns:
        corrected = df["corrected_normalised_development"].astype(str).str.strip()
        mask = corrected.ne("")
        df.loc[mask, "normalised_development"] = corrected.loc[mask]

    df["normalised_development"] = df["normalised_development"].astype(str)
    df = df[df["normalised_development"].str.strip().ne("")].copy()

    df = df.drop_duplicates(subset=["development_id"], keep="last")
    df = df.reset_index(drop=True)

    return df


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------

def call_bedrock_embedding(
    config: ClusterConfig,
    bedrock_runtime: Any,
    text: str,
) -> np.ndarray:
    """Create one normalized Titan text embedding."""

    last_error: Optional[Exception] = None
    request_body = {
        "inputText": text,
        "dimensions": config.embedding_dimensions,
        "normalize": True,
    }

    for attempt in range(1, config.max_retries + 1):
        try:
            response = bedrock_runtime.invoke_model(
                modelId=config.embedding_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body),
            )
            response_body = json.loads(response["body"].read())
            embedding = response_body.get("embedding")
            if not embedding:
                raise ValueError("Embedding response did not contain an embedding.")
            return np.asarray(embedding, dtype=np.float32)
        except Exception as error:
            last_error = error
            print(
                f"Embedding attempt {attempt}/{config.max_retries} failed: {error}"
            )
            if attempt < config.max_retries:
                time.sleep(2 ** attempt)

    raise RuntimeError("Bedrock embedding request failed.") from last_error


def load_cached_embeddings(
    config: ClusterConfig,
    development_ids: Sequence[str],
) -> Optional[np.ndarray]:
    """Load cache only when IDs, order, model and dimensions match exactly."""

    if (
        not config.embedding_cache_file.exists()
        or not config.embedding_cache_ids.exists()
    ):
        return None

    try:
        with config.embedding_cache_ids.open("r", encoding="utf-8") as file:
            cached_ids = json.load(file)

        if list(development_ids) != cached_ids:
            print("Embedding cache does not match current development set; regenerating.")
            return None

        cache = np.load(config.embedding_cache_file)
        embeddings = cache["embeddings"]
        cached_model_id = str(cache["embedding_model_id"].item())
        cached_dimensions = int(cache["embedding_dimensions"].item())

        expected_shape = (len(development_ids), config.embedding_dimensions)
        if cached_model_id != config.embedding_model_id:
            return None
        if cached_dimensions != config.embedding_dimensions:
            return None
        if embeddings.shape != expected_shape:
            return None

        print(f"Loaded {len(embeddings)} embeddings from cache.")
        return embeddings
    except Exception as error:
        print(f"Could not load embedding cache: {error}")
        return None


def save_embedding_cache(
    config: ClusterConfig,
    development_ids: Sequence[str],
    embeddings: np.ndarray,
) -> None:
    config.embedding_cache_file.parent.mkdir(parents=True, exist_ok=True)
    config.embedding_cache_ids.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        config.embedding_cache_file,
        embeddings=embeddings,
        embedding_model_id=config.embedding_model_id,
        embedding_dimensions=config.embedding_dimensions,
    )

    with config.embedding_cache_ids.open("w", encoding="utf-8") as file:
        json.dump(list(development_ids), file, ensure_ascii=False, indent=2)


def generate_development_embeddings(
    config: ClusterConfig,
    bedrock_runtime: Any,
    developments_df: pd.DataFrame,
) -> np.ndarray:
    """Generate or load one normalized embedding per extracted development."""

    development_ids = developments_df["development_id"].astype(str).tolist()
    cached = load_cached_embeddings(config, development_ids)
    if cached is not None:
        return cached

    embeddings: List[np.ndarray] = []
    total = len(developments_df)

    for position, (_, row) in enumerate(developments_df.iterrows(), start=1):
        text = clean_value(row.get("normalised_development", ""))
        print(f"Embedding development {position}/{total}: {text[:90]}")
        embeddings.append(
            call_bedrock_embedding(
                config=config,
                bedrock_runtime=bedrock_runtime,
                text=text,
            )
        )
        time.sleep(config.embedding_request_delay_seconds)

    embedding_matrix = np.vstack(embeddings)
    save_embedding_cache(config, development_ids, embedding_matrix)
    return embedding_matrix


# -----------------------------------------------------------------------------
# Semantic pre-clustering
# -----------------------------------------------------------------------------

def make_agglomerative_model(
    *,
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
) -> AgglomerativeClustering:
    kwargs: Dict[str, Any] = {
        "n_clusters": n_clusters,
        "linkage": "average",
    }
    if distance_threshold is not None:
        kwargs["distance_threshold"] = distance_threshold

    try:
        return AgglomerativeClustering(metric="cosine", **kwargs)
    except TypeError:
        return AgglomerativeClustering(affinity="cosine", **kwargs)


def add_embedding_clusters(
    config: ClusterConfig,
    developments_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    """Assign every development to a semantic candidate cluster."""

    df = developments_df.copy()
    if len(df) == 1:
        df["embedding_cluster_id"] = "EC001"
        df["embedding_cluster_size"] = 1
        return df

    cluster_model = make_agglomerative_model(
        n_clusters=None,
        distance_threshold=config.embedding_distance_threshold,
    )
    raw_labels = cluster_model.fit_predict(embeddings)

    first_position: Dict[int, int] = {}
    for position, label in enumerate(raw_labels):
        first_position.setdefault(int(label), position)

    ordered_labels = sorted(first_position, key=first_position.get)
    label_map = {
        raw_label: f"EC{index:03d}"
        for index, raw_label in enumerate(ordered_labels, start=1)
    }

    df["embedding_cluster_id"] = [label_map[int(label)] for label in raw_labels]
    sizes = df["embedding_cluster_id"].value_counts().to_dict()
    df["embedding_cluster_size"] = (
        df["embedding_cluster_id"].map(sizes).astype(int)
    )
    return df


def recursively_split_large_cluster(
    row_indices: List[int],
    embeddings: np.ndarray,
    max_size: int,
) -> List[List[int]]:
    if len(row_indices) <= max_size:
        return [row_indices]

    subset_embeddings = embeddings[row_indices]
    try:
        split_model = make_agglomerative_model(n_clusters=2)
        split_labels = split_model.fit_predict(subset_embeddings)
        left = [
            row_indices[index]
            for index, label in enumerate(split_labels)
            if int(label) == 0
        ]
        right = [
            row_indices[index]
            for index, label in enumerate(split_labels)
            if int(label) == 1
        ]
        if not left or not right:
            raise ValueError("Semantic split produced an empty subcluster.")
        return (
            recursively_split_large_cluster(left, embeddings, max_size)
            + recursively_split_large_cluster(right, embeddings, max_size)
        )
    except Exception as error:
        print(f"Warning: semantic subdivision failed; using chunks: {error}")
        return [
            row_indices[start : start + max_size]
            for start in range(0, len(row_indices), max_size)
        ]


def build_semantic_batches(
    config: ClusterConfig,
    clustered_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> List[pd.DataFrame]:
    batches: List[pd.DataFrame] = []

    for cluster_id, cluster_df in clustered_df.groupby(
        "embedding_cluster_id", sort=True
    ):
        row_indices = cluster_df.index.astype(int).tolist()
        semantic_parts = recursively_split_large_cluster(
            row_indices=row_indices,
            embeddings=embeddings,
            max_size=config.max_llm_cluster_size,
        )

        for subcluster_number, part_indices in enumerate(semantic_parts, start=1):
            batch_df = clustered_df.loc[part_indices].copy()
            batch_df["embedding_subcluster_id"] = (
                f"{cluster_id}_S{subcluster_number:03d}"
            )
            batches.append(batch_df)

    return batches


# -----------------------------------------------------------------------------
# LLM story/development grouping
# -----------------------------------------------------------------------------

def build_grouping_prompt(
    batch_df: pd.DataFrame,
    batch_number: int,
    conversation_type: Optional[str] = None,
) -> str:
    developments = []

    for _, row in batch_df.iterrows():
        developments.append(
            {
                "development_id": clean_value(row.get("development_id", "")),
                "development": clean_value(row.get("normalised_development", "")),
                "event_type": clean_value(row.get("event_type", "")),
                "locations": clean_value(row.get("locations", "")),
                "published_at": clean_value(row.get("published_at", "")),
                "source_name": clean_value(row.get("source_name", "")),
            }
        )

    developments_json = json.dumps(developments, ensure_ascii=False, indent=2)

    stream_label = clean_value(conversation_type) or "single"

    return f"""
You are grouping extracted Ebola monitoring developments into coherent real-world stories
within the {stream_label} conversation stream.

The records below were placed together by semantic embeddings. Similar wording does not
necessarily mean they describe the same event. Your task is to identify which developments
refer to the same underlying outbreak event, public-health action, scientific finding,
risk assessment, or operational development.

Do not compare this stream with other information environments. The goal at this stage is
to describe the internal structure of the {stream_label} conversation accurately.

Developments:
{developments_json}

Return valid JSON only:

{{
  "groups": [
    {{
      "provisional_group_id": "B{batch_number:03d}_G001",
      "group_name": "Short factual headline for the shared story",
      "group_summary": "One or two sentences summarising the shared development without adding unsupported facts.",
      "development_ids": ["development_id"],
      "group_type": "recurring or singleton",
      "importance": "high, medium, or low",
      "grouping_confidence": "high, medium, or low",
      "grouping_reason": "Brief explanation of why these records describe the same story."
    }}
  ]
}}

Rules:
1. Group records only when they describe the same underlying real-world development or story.
2. Do not merge records merely because they concern Ebola, the same country, or the same organisation.
3. Use dates, locations, actors, event type and factual content to distinguish separate events.
4. Multiple reports of the same case update, outbreak declaration, vaccination campaign,
   response action, or scientific finding may be grouped together.
5. A later materially different update should normally be a separate group unless it is clearly
   another report of the same event.
6. Preserve unusual or one-off developments as singleton groups.
7. Every development ID must appear exactly once.
8. Do not invent, omit, or duplicate development IDs.
9. Prefer under-merging to over-merging when uncertain.
10. Group names must be factual and concise, not generic taxonomy labels.
11. Do not infer facts that are absent from the supplied records.
12. Do not include Markdown or commentary outside the JSON.
""".strip()


def validate_and_repair_groups(
    result: Any,
    batch_df: pd.DataFrame,
    batch_number: int,
) -> List[Dict[str, Any]]:
    if not isinstance(result, dict):
        raise ValueError("Grouping response was not an object.")

    groups = result.get("groups", [])
    if not isinstance(groups, list):
        raise ValueError("'groups' was not a list.")

    expected_ids = batch_df["development_id"].astype(str).tolist()
    expected_set = set(expected_ids)
    text_lookup = (
        batch_df.set_index("development_id")["normalised_development"]
        .astype(str)
        .to_dict()
    )

    seen_ids = set()
    cleaned_groups: List[Dict[str, Any]] = []

    for group_number, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            continue
        supplied_ids = group.get("development_ids", [])
        if not isinstance(supplied_ids, list):
            continue

        retained_ids = []
        for development_id in supplied_ids:
            development_id = clean_value(development_id)
            if development_id not in expected_set:
                print(f"Removing invented development ID: {development_id}")
                continue
            if development_id in seen_ids:
                print(f"Removing duplicate assignment: {development_id}")
                continue
            retained_ids.append(development_id)
            seen_ids.add(development_id)

        if not retained_ids:
            continue

        group["development_ids"] = retained_ids
        group["group_type"] = "singleton" if len(retained_ids) == 1 else "recurring"

        if not clean_value(group.get("provisional_group_id", "")):
            group["provisional_group_id"] = (
                f"B{batch_number:03d}_G{group_number:03d}"
            )

        if len(retained_ids) == 1 and not clean_value(group.get("group_name", "")):
            group["group_name"] = text_lookup[retained_ids[0]]

        cleaned_groups.append(group)

    missing_ids = expected_set - seen_ids
    for singleton_number, development_id in enumerate(sorted(missing_ids), start=1):
        text = text_lookup.get(development_id, "Development requiring manual review.")
        print(f"Preserving omitted development as singleton: {development_id}")
        cleaned_groups.append(
            {
                "provisional_group_id": (
                    f"B{batch_number:03d}_AUTO_{singleton_number:03d}"
                ),
                "group_name": text,
                "group_summary": text,
                "development_ids": [development_id],
                "group_type": "singleton",
                "importance": "medium",
                "grouping_confidence": "low",
                "grouping_reason": (
                    "Automatically preserved because the grouping response omitted this development."
                ),
            }
        )

    return cleaned_groups


def run_provisional_grouping(
    config: ClusterConfig,
    bedrock_runtime: Any,
    clustered_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    """
    Group developments within semantically coherent candidate batches.

    Resilience features:
    - Resume from grouped_developments_file when it already contains completed
      development IDs.
    - Checkpoint after every completed semantic batch.
    - If an LLM batch still exceeds the output-token limit (or repeatedly fails),
      recursively split that batch into smaller halves rather than aborting the
      whole run.
    """

    output_records: List[Dict[str, Any]] = []
    completed_ids: set[str] = set()

    # Resume from the most recent checkpoint when present.
    if config.grouped_developments_file.exists():
        try:
            existing_df = pd.read_csv(
                config.grouped_developments_file,
                dtype=str,
            ).fillna("")

            if (
                not existing_df.empty
                and "development_id" in existing_df.columns
            ):
                existing_df = existing_df.drop_duplicates(
                    subset=["development_id"],
                    keep="last",
                ).reset_index(drop=True)

                output_records = existing_df.to_dict("records")
                completed_ids = set(
                    existing_df["development_id"]
                    .astype(str)
                    .tolist()
                )

                print(
                    f"Resuming grouping from checkpoint: "
                    f"{len(completed_ids)} developments already completed."
                )

        except (pd.errors.EmptyDataError, OSError, ValueError) as error:
            print(
                f"Could not load grouping checkpoint; starting fresh: {error}"
            )

    semantic_batches = build_semantic_batches(
        config,
        clustered_df,
        embeddings,
    )
    total_batches = len(semantic_batches)

    def process_batch(
        batch_df: pd.DataFrame,
        batch_number: int,
        split_label: str = "",
    ) -> List[Dict[str, Any]]:
        """Process one LLM batch, recursively splitting it if necessary."""

        batch_df = batch_df[
            ~batch_df["development_id"]
            .astype(str)
            .isin(completed_ids)
        ].copy()

        if batch_df.empty:
            return []

        embedding_cluster_id = clean_value(
            batch_df["embedding_cluster_id"].iloc[0]
        )
        embedding_subcluster_id = clean_value(
            batch_df["embedding_subcluster_id"].iloc[0]
        )

        suffix = f" {split_label}" if split_label else ""

        print(
            f"\nGrouping semantic batch {batch_number}/{total_batches}"
            f"{suffix} | {embedding_subcluster_id} | "
            f"{len(batch_df)} developments"
        )

        try:
            result = call_bedrock_json(
                bedrock_runtime,
                prompt=build_grouping_prompt(
                    batch_df,
                    batch_number,
                    conversation_type=config.conversation_type,
                ),
                model_id=config.grouping_model_id,
                max_tokens=4000,
                max_retries=config.max_retries,
                operation_name="Development grouping",
            )

            groups = validate_and_repair_groups(
                result,
                batch_df,
                batch_number,
            )

            lookup = (
                batch_df
                .set_index("development_id", drop=False)
                .to_dict("index")
            )

            records: List[Dict[str, Any]] = []

            for group in groups:
                for development_id in group["development_ids"]:
                    original = lookup[development_id]

                    records.append(
                        {
                            **original,
                            "embedding_cluster_id": embedding_cluster_id,
                            "embedding_subcluster_id": embedding_subcluster_id,
                            "provisional_group_id": clean_value(
                                group.get("provisional_group_id", "")
                            ),
                            "provisional_group_name": clean_value(
                                group.get("group_name", "")
                            ),
                            "provisional_group_summary": clean_value(
                                group.get("group_summary", "")
                            ),
                            "provisional_group_type": clean_value(
                                group.get("group_type", "")
                            ),
                            "group_importance": clean_value(
                                group.get("importance", "")
                            ),
                            "grouping_confidence": clean_value(
                                group.get("grouping_confidence", "")
                            ),
                            "grouping_reason": clean_value(
                                group.get("grouping_reason", "")
                            ),
                        }
                    )

            return records

        except RuntimeError as error:
            # A smaller batch normally solves max-token failures. This also
            # protects the pipeline from one pathological LLM response.
            if len(batch_df) > 1:
                split_point = max(1, len(batch_df) // 2)

                left_df = batch_df.iloc[:split_point].copy()
                right_df = batch_df.iloc[split_point:].copy()

                print(
                    f"Batch failed ({error}). "
                    f"Splitting {len(batch_df)} developments into "
                    f"{len(left_df)} + {len(right_df)}."
                )

                return (
                    process_batch(
                        left_df,
                        batch_number,
                        split_label=f"{split_label}A".strip(),
                    )
                    +
                    process_batch(
                        right_df,
                        batch_number,
                        split_label=f"{split_label}B".strip(),
                    )
                )

            # Preserve a single development locally rather than aborting an
            # otherwise successful multi-hundred-batch run.
            row = batch_df.iloc[0].to_dict()
            development_id = clean_value(
                row.get("development_id", "")
            )
            development_text = clean_value(
                row.get("normalised_development", "")
            )

            print(
                f"Warning: preserving failed singleton locally: "
                f"{development_id}"
            )

            return [
                {
                    **row,
                    "embedding_cluster_id": embedding_cluster_id,
                    "embedding_subcluster_id": embedding_subcluster_id,
                    "provisional_group_id": (
                        f"B{batch_number:03d}_FALLBACK_{development_id}"
                    ),
                    "provisional_group_name": development_text,
                    "provisional_group_summary": development_text,
                    "provisional_group_type": "singleton",
                    "group_importance": clean_value(
                        row.get("importance", "medium")
                    ),
                    "grouping_confidence": "low",
                    "grouping_reason": (
                        "Local singleton fallback after repeated "
                        "LLM grouping failure."
                    ),
                }
            ]

    for batch_number, batch_df in enumerate(
        semantic_batches,
        start=1,
    ):
        batch_ids = set(
            batch_df["development_id"]
            .astype(str)
            .tolist()
        )

        if batch_ids.issubset(completed_ids):
            print(
                f"Skipping semantic batch {batch_number}/{total_batches}: "
                "already checkpointed."
            )
            continue

        new_records = process_batch(
            batch_df,
            batch_number,
        )

        if new_records:
            output_records.extend(new_records)

            newly_completed = {
                str(record.get("development_id", ""))
                for record in new_records
                if str(record.get("development_id", ""))
            }
            completed_ids.update(newly_completed)

            checkpoint_df = pd.DataFrame(
                output_records
            ).drop_duplicates(
                subset=["development_id"],
                keep="last",
            )

            safe_save_csv(
                checkpoint_df,
                config.grouped_developments_file,
            )

            print(
                f"Checkpoint saved: "
                f"{len(completed_ids)} developments grouped."
            )

        time.sleep(
            config.request_delay_seconds
        )

    final_df = pd.DataFrame(
        output_records
    )

    if not final_df.empty:
        final_df = final_df.drop_duplicates(
            subset=["development_id"],
            keep="last",
        ).reset_index(drop=True)

    return final_df


# -----------------------------------------------------------------------------
# Group summaries / briefing-ready table
# -----------------------------------------------------------------------------

def _unique_nonempty(values: pd.Series) -> List[str]:
    cleaned = [clean_value(value) for value in values.tolist()]
    return sorted({value for value in cleaned if value})


def create_group_summaries(grouped_df: pd.DataFrame) -> pd.DataFrame:
    """Create one briefing-ready row per provisional development group."""

    summary_rows: List[Dict[str, Any]] = []

    for group_id, group_df in grouped_df.groupby("provisional_group_id", dropna=False):
        article_ids = (
            _unique_nonempty(group_df["article_id"])
            if "article_id" in group_df.columns
            else []
        )
        source_names = (
            _unique_nonempty(group_df["source_name"])
            if "source_name" in group_df.columns
            else []
        )
        event_types = (
            _unique_nonempty(group_df["event_type"])
            if "event_type" in group_df.columns
            else []
        )
        locations = (
            _unique_nonempty(group_df["locations"])
            if "locations" in group_df.columns
            else []
        )
        developments = _unique_nonempty(group_df["normalised_development"])
        conversation_types = (
            _unique_nonempty(group_df["conversation_type"])
            if "conversation_type" in group_df.columns
            else []
        )
        conversation_type = (
            conversation_types[0]
            if len(conversation_types) == 1
            else ""
        )

        summary_rows.append(
            {
                "group_id": group_id,
                "conversation_type": conversation_type,
                "group_name": clean_value(group_df["provisional_group_name"].iloc[0]),
                "group_summary": clean_value(
                    group_df["provisional_group_summary"].iloc[0]
                ),
                "group_type": (
                    "singleton" if len(group_df) == 1 else "recurring"
                ),
                "importance": clean_value(group_df["group_importance"].iloc[0]),
                "development_count": len(group_df),
                "article_count": len(article_ids),
                "source_count": len(source_names),
                "sources": json.dumps(source_names, ensure_ascii=False),
                "event_types": json.dumps(event_types, ensure_ascii=False),
                "locations": json.dumps(locations, ensure_ascii=False),
                "embedding_cluster_ids": json.dumps(
                    _unique_nonempty(group_df["embedding_cluster_id"]),
                    ensure_ascii=False,
                ),
                "development_ids": json.dumps(
                    group_df["development_id"].astype(str).tolist(),
                    ensure_ascii=False,
                ),
                "article_ids": json.dumps(article_ids, ensure_ascii=False),
                "developments": json.dumps(developments, ensure_ascii=False),
                "grouping_confidence": clean_value(
                    group_df["grouping_confidence"].iloc[0]
                ),
                "researcher_review": "",
                "corrected_group_name": "",
                "corrected_group_summary": "",
                "group_review_notes": "",
            }
        )

    result = pd.DataFrame(summary_rows)
    if not result.empty:
        importance_order = {"high": 0, "medium": 1, "low": 2}
        result["_importance_order"] = (
            result["importance"].str.lower().map(importance_order).fillna(3)
        )
        result = result.sort_values(
            ["_importance_order", "development_count", "article_count"],
            ascending=[True, False, False],
        ).drop(columns=["_importance_order"])
        result = result.reset_index(drop=True)

    return result


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def run_clustering(
    *,
    config: ClusterConfig,
    bedrock_runtime: Any,
    developments_df: Optional[pd.DataFrame] = None,
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run embeddings, semantic pre-clustering and LLM story grouping for one stream."""

    df = load_developments(config=config, developments_df=developments_df)

    if config.conversation_type:
        print(
            f"Loaded {len(df)} extracted developments for "
            f"{config.conversation_type}."
        )
    else:
        print(f"Loaded {len(df)} extracted developments.")

    if df.empty:
        raise ValueError("No developments were available for grouping.")

    embeddings = generate_development_embeddings(
        config=config,
        bedrock_runtime=bedrock_runtime,
        developments_df=df,
    )

    clustered_df = add_embedding_clusters(
        config=config,
        developments_df=df,
        embeddings=embeddings,
    )

    cluster_count = clustered_df["embedding_cluster_id"].nunique()
    print(f"\nEmbedding stage created {cluster_count} semantic candidate clusters.")

    grouped_df = run_provisional_grouping(
        config=config,
        bedrock_runtime=bedrock_runtime,
        clustered_df=clustered_df,
        embeddings=embeddings,
    )

    group_summaries_df = create_group_summaries(grouped_df)

    if save_outputs:
        safe_save_csv(grouped_df, config.grouped_developments_file)
        safe_save_csv(group_summaries_df, config.group_summaries_file)
        print("\nSaved:")
        print(config.grouped_developments_file)
        print(config.group_summaries_file)

    return grouped_df, group_summaries_df


def main_cluster(*, config: ClusterConfig, bedrock_runtime: Any) -> None:
    """CSV-oriented compatibility wrapper."""
    run_clustering(
        config=config,
        bedrock_runtime=bedrock_runtime,
        developments_df=None,
        save_outputs=True,
    )


