# Package Imports

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .common import call_bedrock_json, clean_value, safe_save_csv


@dataclass(frozen=True)
class ExtractionConfig:
    input_file: Path
    extraction_file: Path
    article_log_file: Path

    article_id_column: str
    headline_column: str
    body_column: str
    date_column: str
    source_column: str
    language_column: str
    url_column: str

    model_id: str
    number_of_articles: int | None = None
    max_article_characters: int = 30_000
    request_delay_seconds: float = 0.5
    max_retries: int = 4


def make_article_id(
    row: pd.Series,
    row_number: int,
    config: ExtractionConfig,
) -> str:
    existing_id = clean_value(row.get(config.article_id_column, ""))

    if existing_id:
        return existing_id

    identifying_text = "|".join(
        [
            clean_value(row.get(config.headline_column, "")),
            clean_value(row.get(config.url_column, "")),
            str(row_number),
        ]
    )

    digest = hashlib.sha1(
        identifying_text.encode("utf-8")
    ).hexdigest()[:12]

    return f"article_{digest}"


def truncate_text(
    text: str,
    max_characters: int,
) -> str:
    if len(text) <= max_characters:
        return text

    first_length = int(max_characters * 0.75)
    final_length = max_characters - first_length

    return (
        text[:first_length]
        + "\n\n[ARTICLE TRUNCATED]\n\n"
        + text[-final_length:]
    )


def build_extraction_prompt(
    article_id: str,
    headline: str,
    body: str,
    source: str,
    publication_date: str,
    language: str,
) -> str:
    return f"""
You are analysing reporting for an Ebola monitoring system.

Determine whether the article is materially relevant to Ebola. If it is, extract every distinct substantive Ebola development reported in the article.

A development is a concrete update, event, action, change, finding, risk assessment, or material piece of information that would be useful for Ebola monitoring.

Examples include:
- outbreak detection or confirmation
- case updates
- death updates
- geographic spread
- outbreak containment or an outbreak ending
- cross-border transmission or cross-border risk
- surveillance, screening, testing, or laboratory activity
- contact tracing
- vaccination or vaccine development
- treatment or clinical-care developments
- health-system response
- public-health response measures
- community engagement or behaviour
- humanitarian access, conflict, displacement, or security constraints affecting the response
- funding or international assistance
- scientific or epidemiological findings
- official risk assessments or guidance
- misinformation, rumours, distrust, or harmful narratives

Article ID: {article_id}
Headline: {headline}
Source: {source}
Publication date: {publication_date}
Language: {language}

--- BEGIN ARTICLE ---
{body}
--- END ARTICLE ---

Return valid JSON only using exactly this structure:

{{
  "article_id": "{article_id}",
  "ebola_relevant": true,
  "article_summary": "A concise factual summary of the article's Ebola relevance.",
  "developments": [
    {{
      "development_text": "The development as described in the article.",
      "normalised_development": "A concise standalone English sentence describing the development.",
      "evidence_text": "The shortest supporting passage copied directly from the article.",
      "event_type": "One category from the supplied list.",
      "locations": [],
      "organisations": [],
      "people": [],
      "case_count": null,
      "death_count": null,
      "cross_border_relevance": "yes, no, or unclear",
      "importance": "high, medium, or low",
      "extraction_confidence": "high, medium, or low"
    }}
  ]
}}

Event-type options:
- outbreak_detection
- case_update
- death_update
- geographic_spread
- outbreak_containment
- outbreak_end
- cross_border_event
- surveillance_or_testing
- contact_tracing
- vaccination
- treatment_or_clinical_care
- health_system_response
- public_health_response
- community_engagement
- humanitarian_or_security_constraint
- funding_or_assistance
- research_or_scientific_finding
- official_risk_assessment
- misinformation_or_rumour
- other

Rules:
1. Extract only developments materially related to Ebola.
2. One object must represent one distinct underlying development.
3. Extract all distinct substantive developments in the article.
4. Do not invent unsupported facts.
5. evidence_text must be copied from the article.
6. Keep evidence_text in the article's original language.
7. Write normalised_development in English.
8. Use the most specific event_type available.
9. Use null for case_count or death_count when no clear numeric count is stated for that specific development.
10. Do not infer numeric counts from vague wording such as "many", "several", or "rising".
11. If the article discusses misinformation, rumours, distrust, or false claims about Ebola, treat that as an Ebola development and use misinformation_or_rumour.
12. If the article is not materially relevant to Ebola, return ebola_relevant as false and developments as an empty list.
13. If the article is relevant to Ebola but contains no distinct substantive development, return ebola_relevant as true and developments as an empty list.
14. Return JSON only.
""".strip()


def load_existing_extractions(
    config: ExtractionConfig,
) -> pd.DataFrame:
    if not config.extraction_file.exists():
        return pd.DataFrame()

    try:
        extracted_df = pd.read_csv(
            config.extraction_file,
            dtype=str,
        ).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    required_columns = {"development_id", "article_id"}
    missing_columns = required_columns - set(extracted_df.columns)

    if missing_columns:
        raise KeyError(
            "Existing extraction file is missing required "
            f"columns: {sorted(missing_columns)}. "
            "If this was created by the old misinformation extractor, "
            "rename or delete it before running this version."
        )

    return extracted_df


def load_completed_article_ids(
    config: ExtractionConfig,
    extracted_df: pd.DataFrame,
) -> set[str]:
    if not config.article_log_file.exists():
        return set()

    try:
        log_df = pd.read_csv(
            config.article_log_file,
            dtype=str,
        ).fillna("")
    except pd.errors.EmptyDataError:
        return set()

    required_columns = {"article_id", "status"}

    if not required_columns.issubset(log_df.columns):
        return set()

    latest_results = log_df.drop_duplicates(
        subset=["article_id"],
        keep="last",
    )

    empty_result_ids = set(
        latest_results.loc[
            latest_results["status"].isin(
                ["not_relevant", "no_development"]
            ),
            "article_id",
        ].astype(str)
    )

    saved_article_ids: set[str] = set()

    if (
        not extracted_df.empty
        and "article_id" in extracted_df.columns
    ):
        saved_article_ids = set(
            extracted_df["article_id"].astype(str)
        )

    logged_completed_ids = set(
        latest_results.loc[
            latest_results["status"] == "completed",
            "article_id",
        ].astype(str)
    )

    unsafe_completed_ids = logged_completed_ids - saved_article_ids

    for article_id in sorted(unsafe_completed_ids):
        print(
            f"Reprocessing {article_id}: the log says completed "
            "but no saved developments were found."
        )

    return (
        empty_result_ids
        | (logged_completed_ids & saved_article_ids)
    )


def append_article_log(
    config: ExtractionConfig,
    log_record: dict,
) -> None:
    new_record_df = pd.DataFrame([log_record])

    if config.article_log_file.exists():
        try:
            existing_log_df = pd.read_csv(
                config.article_log_file,
                dtype=str,
            ).fillna("")
        except pd.errors.EmptyDataError:
            existing_log_df = pd.DataFrame()

        log_df = pd.concat(
            [existing_log_df, new_record_df],
            ignore_index=True,
        )
    else:
        log_df = new_record_df

    safe_save_csv(
        log_df,
        config.article_log_file,
    )


def checkpoint_article_developments(
    config: ExtractionConfig,
    extracted_df: pd.DataFrame,
    article_id: str,
    article_development_records: list[dict[str, Any]],
) -> pd.DataFrame:
    if (
        not extracted_df.empty
        and "article_id" not in extracted_df.columns
    ):
        raise KeyError(
            "Existing extraction data has no 'article_id' column."
        )

    article_was_saved = (
        not extracted_df.empty
        and extracted_df["article_id"]
        .astype(str)
        .eq(article_id)
        .any()
    )

    if extracted_df.empty:
        retained_df = extracted_df.copy()
    else:
        retained_df = extracted_df.loc[
            ~extracted_df["article_id"]
            .astype(str)
            .eq(article_id)
        ].copy()

    new_developments_df = pd.DataFrame(
        article_development_records
    )

    if new_developments_df.empty:
        updated_df = retained_df.reset_index(drop=True)
    else:
        updated_df = pd.concat(
            [retained_df, new_developments_df],
            ignore_index=True,
        )

        updated_df = (
            updated_df
            .drop_duplicates(
                subset=["development_id"],
                keep="last",
            )
            .reset_index(drop=True)
        )

    if article_development_records or article_was_saved:
        safe_save_csv(
            updated_df,
            config.extraction_file,
        )

    return updated_df


def flatten_development(
    config: ExtractionConfig,
    row: pd.Series,
    article_id: str,
    development_number: int,
    development: dict,
    article_summary: str,
    ebola_relevant: bool,
) -> dict:
    return {
        "development_id": (
            f"{article_id}_development_{development_number:02d}"
        ),
        "article_id": article_id,
        "platform": clean_value(row.get("platform", "")),
        "source_id": clean_value(row.get("source_id", "")),
        "headline": clean_value(
            row.get(config.headline_column, "")
        ),
        "publication_date": clean_value(
            row.get(config.date_column, "")
        ),
        "country": clean_value(row.get("country", "")),
        "source": clean_value(
            row.get(config.source_column, "")
        ),
        "author": clean_value(row.get("author", "")),
        "article_language": clean_value(
            row.get(config.language_column, "")
        ),
        "url": clean_value(
            row.get(config.url_column, "")
        ),
        "search_id": clean_value(row.get("search_id", "")),
        "ebola_relevant": str(bool(ebola_relevant)).lower(),
        "article_summary": clean_value(article_summary),
        "development_text": clean_value(
            development.get("development_text", "")
        ),
        "normalised_development": clean_value(
            development.get("normalised_development", "")
        ),
        "evidence_text": clean_value(
            development.get("evidence_text", "")
        ),
        "event_type": clean_value(
            development.get("event_type", "")
        ),
        "locations": json.dumps(
            development.get("locations", []),
            ensure_ascii=False,
        ),
        "organisations": json.dumps(
            development.get("organisations", []),
            ensure_ascii=False,
        ),
        "people": json.dumps(
            development.get("people", []),
            ensure_ascii=False,
        ),
        "case_count": clean_value(
            development.get("case_count", "")
        ),
        "death_count": clean_value(
            development.get("death_count", "")
        ),
        "cross_border_relevance": clean_value(
            development.get("cross_border_relevance", "")
        ),
        "importance": clean_value(
            development.get("importance", "")
        ),
        "extraction_confidence": clean_value(
            development.get("extraction_confidence", "")
        ),
        "review_status": "",
        "include_in_grouping": "",
        "corrected_normalised_development": "",
        "corrected_event_type": "",
        "review_notes": "",
    }


def run_extraction(
    articles_df: pd.DataFrame,
    *,
    config: ExtractionConfig,
    bedrock_runtime: Any,
) -> pd.DataFrame:
    extracted_df = load_existing_extractions(config)

    completed_article_ids = load_completed_article_ids(
        config,
        extracted_df,
    )

    for row_number, (_, row) in enumerate(
        articles_df.iterrows(),
        start=1,
    ):
        article_id = make_article_id(
            row,
            row_number,
            config=config,
        )

        if article_id in completed_article_ids:
            print(
                f"Skipping {article_id}: already processed."
            )
            continue

        headline = clean_value(
            row.get(config.headline_column, "")
        )
        body = clean_value(
            row.get(config.body_column, "")
        )

        if not body:
            append_article_log(
                config=config,
                log_record={
                    "article_id": article_id,
                    "headline": headline,
                    "status": "failed",
                    "ebola_relevant": "",
                    "development_count": 0,
                    "message": "Empty article body",
                },
            )
            continue

        print(
            f"Extracting article "
            f"{row_number}/{len(articles_df)}: "
            f"{headline[:80]}"
        )

        prompt = build_extraction_prompt(
            article_id=article_id,
            headline=headline,
            body=truncate_text(
                body,
                config.max_article_characters,
            ),
            source=clean_value(
                row.get(config.source_column, "")
            ),
            publication_date=clean_value(
                row.get(config.date_column, "")
            ),
            language=clean_value(
                row.get(config.language_column, "")
            ),
        )

        try:
            result = call_bedrock_json(
                bedrock_runtime,
                prompt=prompt,
                model_id=config.model_id,
                max_tokens=5000,
                max_retries=config.max_retries,
                operation_name="Extraction",
            )

            ebola_relevant = bool(
                result.get("ebola_relevant", False)
            )
            article_summary = clean_value(
                result.get("article_summary", "")
            )
            developments = result.get("developments", [])

            if not isinstance(developments, list):
                raise ValueError(
                    "'developments' was not a list."
                )

            valid_developments = []

            for development in developments:
                if not isinstance(development, dict):
                    continue

                normalised_development = clean_value(
                    development.get(
                        "normalised_development",
                        "",
                    )
                )

                if normalised_development:
                    valid_developments.append(development)

            article_development_records = [
                flatten_development(
                    config=config,
                    row=row,
                    article_id=article_id,
                    development_number=development_number,
                    development=development,
                    article_summary=article_summary,
                    ebola_relevant=ebola_relevant,
                )
                for development_number, development in enumerate(
                    valid_developments,
                    start=1,
                )
            ]

            extracted_df = checkpoint_article_developments(
                config=config,
                extracted_df=extracted_df,
                article_id=article_id,
                article_development_records=article_development_records,
            )

            if not ebola_relevant:
                status = "not_relevant"
            elif valid_developments:
                status = "completed"
            else:
                status = "no_development"

            append_article_log(
                config=config,
                log_record={
                    "article_id": article_id,
                    "headline": headline,
                    "status": status,
                    "ebola_relevant": str(
                        ebola_relevant
                    ).lower(),
                    "development_count": len(
                        valid_developments
                    ),
                    "message": article_summary,
                },
            )

            print(
                f"Ebola relevant: {ebola_relevant} | "
                f"Extracted {len(valid_developments)} development(s)."
            )

        except Exception as error:
            print(
                f"FAILED: {type(error).__name__}: {error}"
            )

            append_article_log(
                config=config,
                log_record={
                    "article_id": article_id,
                    "headline": headline,
                    "status": "failed",
                    "ebola_relevant": "",
                    "development_count": 0,
                    "message": str(error),
                },
            )

        time.sleep(config.request_delay_seconds)

    return extracted_df


def main_extract(
    *,
    config: ExtractionConfig,
    bedrock_runtime: Any,
) -> None:
    read_kwargs = {
        "filepath_or_buffer": config.input_file,
        "dtype": str,
    }

    if config.number_of_articles is not None:
        read_kwargs["nrows"] = config.number_of_articles

    articles_df = pd.read_csv(
        **read_kwargs
    ).fillna("")

    articles_df = articles_df.loc[
        :,
        ~articles_df.columns.astype(str).str.startswith(
            "Unnamed"
        ),
    ]

    if config.body_column not in articles_df.columns:
        raise KeyError(
            f"Required column not found: {config.body_column}"
        )

    print(f"Loaded {len(articles_df)} articles.")

    extracted_df = run_extraction(
        articles_df=articles_df,
        bedrock_runtime=bedrock_runtime,
        config=config,
    )

    print(
        f"\nSaved {len(extracted_df)} developments to:"
    )
    print(config.extraction_file)
    print("\nDevelopment extraction complete.")


