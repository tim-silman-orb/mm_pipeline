from __future__ import annotations

import re

from .config import TopicFilterConfig
from .exceptions import DatasetValidationError
from .models import Article


def apply_topic_filter(
    articles: list[Article],
    config: TopicFilterConfig,
) -> list[Article]:
    """Apply configured topic terms to selected platforms."""

    if not config.enabled or not config.terms:
        return list(articles)

    platforms = set(config.apply_to_platforms)
    literal_terms = [term.casefold().strip() for term in config.terms if term.strip()]
    regexes = (
        [re.compile(term, flags=re.IGNORECASE) for term in config.terms]
        if config.mode == "regex"
        else []
    )

    def keep(article: Article) -> bool:
        if platforms and article.platform not in platforms:
            return True
        values = [str(getattr(article, field, "") or "") for field in config.fields]
        haystack = " ".join(values)
        if config.mode == "regex":
            return any(pattern.search(haystack) for pattern in regexes)
        folded = haystack.casefold()
        matches = [term in folded for term in literal_terms]
        return all(matches) if config.mode == "literal_all" else any(matches)

    return [article for article in articles if keep(article)]


def deduplicate_articles(articles: list[Article]) -> list[Article]:
    """Preserve source order while removing repeated IDs and URLs."""

    output: list[Article] = []
    article_ids: set[str] = set()
    urls: set[str] = set()
    for article in articles:
        url_key = article.url.strip().casefold()
        if article.article_id in article_ids:
            continue
        if url_key and url_key in urls:
            continue
        article_ids.add(article.article_id)
        if url_key:
            urls.add(url_key)
        output.append(article)
    return output


def discard_unusable_articles(articles: list[Article]) -> list[Article]:
    """Discard records with neither content nor a summary."""

    return [article for article in articles if article.analysis_text]


def validate_articles(articles: list[Article]) -> None:
    """Validate the invariants expected by extraction and downstream stages."""

    ids = [article.article_id for article in articles]
    if len(ids) != len(set(ids)):
        raise DatasetValidationError("Master dataset contains duplicate article IDs")
    for article in articles:
        if not article.platform.strip():
            raise DatasetValidationError("Every article requires a platform")
        if not article.source_name.strip():
            raise DatasetValidationError(
                f"Article {article.article_id} has no source_name"
            )
        if not article.analysis_text:
            raise DatasetValidationError(
                f"Article {article.article_id} has no usable analysis text"
            )


def article_rows(articles: list[Article]) -> list[dict]:
    """Convert canonical records to serialisable master-dataset rows."""

    return [article.to_dict() for article in articles]
