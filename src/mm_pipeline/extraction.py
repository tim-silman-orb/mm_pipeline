from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import boto3

from .config import ExtractionConfig
from .models import Article


class BedrockDevelopmentExtractor:
    def __init__(
        self,
        config: ExtractionConfig,
        logger: logging.Logger,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        if client is None:
            session_kwargs: dict[str, str] = {}
            if region := os.getenv("AWS_REGION"):
                session_kwargs["region_name"] = region
            if profile := os.getenv("AWS_PROFILE"):
                session_kwargs["profile_name"] = profile
            client = boto3.Session(**session_kwargs).client("bedrock-runtime")
        self.client = client

    def extract(self, articles: list[Article]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for article in articles:
            payload = self._extract_article(article)
            for index, development in enumerate(
                payload.get("developments", []),
                start=1,
            ):
                if not isinstance(development, dict):
                    continue
                rows.append(
                    {
                        "development_id": f"{article.article_id}:{index}",
                        "article_id": article.article_id,
                        "headline": article.headline,
                        "source_name": article.source_name,
                        "published_at": article.to_dict()["published_at"],
                        "url": article.url,
                        **development,
                    }
                )
        return rows

    def _extract_article(self, article: Article) -> dict[str, Any]:
        prompt = _build_development_prompt(
            article,
            max_characters=self.config.max_article_characters,
        )
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.client.converse(
                    modelId=self.config.model_id,
                    messages=[
                        {"role": "user", "content": [{"text": prompt}]}
                    ],
                    inferenceConfig={"maxTokens": 3000, "temperature": 0.0},
                )
                text = "".join(
                    block.get("text", "")
                    for block in response
                    .get("output", {})
                    .get("message", {})
                    .get("content", [])
                )
                payload = _parse_json(text)
                if not isinstance(payload.get("developments", []), list):
                    raise ValueError("Bedrock response 'developments' is not a list")
                return payload
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "Extraction attempt %s/%s failed for %s: %s",
                    attempt,
                    self.config.max_retries,
                    article.article_id,
                    exc,
                )
                if attempt < self.config.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(
            f"Extraction failed for {article.article_id}: {last_error}"
        )


def _build_development_prompt(
    article: Article,
    *,
    max_characters: int,
) -> str:
    text = article.analysis_text[:max_characters]
    return f"""
Extract distinct factual developments from the article below.

Return JSON only, using this structure:
{{
  "developments": [
    {{
      "normalised_development": "concise factual statement",
      "event_type": "short category",
      "locations": ["place"],
      "importance": 1,
      "extraction_confidence": 0.0
    }}
  ]
}}

Do not infer facts that are absent from the text. If there is no concrete
development, return {{"developments": []}}.

Headline: {article.headline}
Source: {article.source_name}
Published: {article.published_at}

Text:
{text}
""".strip()


def _parse_json(text: str) -> dict[str, Any]:
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Bedrock response must be a JSON object")
    return value
