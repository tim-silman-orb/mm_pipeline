from datetime import date
from pathlib import Path

import pandas as pd

from mm_pipeline_2.collection import MasterDatasetBuilder, RunWindow
from mm_pipeline_2.backend.utils import Item, clean_headline
from mm_pipeline_2.config import load_config
from mm_pipeline_2.stages import ConversationClassifier


def test_example_config_loads():
    config = load_config(Path(__file__).parents[1] / "config" / "ebola.yml")
    assert config.project == "ebola"
    assert config.source("opoint")["enabled"] is True


def test_run_window_is_inclusive():
    window = RunWindow(date(2026, 10, 23), date(2026, 10, 29))
    assert window.lookback_days == 7


def test_master_builder_preserves_legacy_filter_and_schema():
    frame = pd.DataFrame(
        [
            {
                "platform": "health_source",
                "source_id": "keep",
                "headline": "Ebola update",
                "content": "",
                "summary": "Relevant summary",
            },
            {
                "platform": "health_source",
                "source_id": "drop",
                "headline": "Unrelated notice",
                "content": "",
                "summary": "Nothing relevant",
            },
            {
                "platform": "opoint",
                "source_id": "news",
                "headline": "Any news record",
                "content": "Body",
                "summary": "",
            },
        ]
    )
    result = MasterDatasetBuilder(
        {
            "platforms": ["health_source"],
            "fields": ["headline", "content", "summary"],
            "terms": ["ebola"],
        }
    ).build([frame])
    assert result["article_id"].tolist() == ["health_source_keep", "opoint_news"]
    assert result.loc[result["source_id"].eq("keep"), "analysis_text"].item() == "Relevant summary"


def test_conversation_classifier_matches_notebook_rules():
    dataframe = pd.DataFrame(
        [
            {"platform": "opoint", "article_id": "opoint_1"},
            {"platform": "health_source", "article_id": "health_source_1"},
            {
                "platform": "health_source",
                "article_id": "health_source_2",
                "source_name": "Google News",
                "url": "https://news.google.com/item",
            },
        ]
    )
    result = ConversationClassifier().apply(dataframe)
    assert result["conversation_type"].tolist() == ["news", "public_health", "news"]


def test_health_helpers_preserve_the_required_legacy_behaviour():
    item = Item(
        title="Example",
        source="Example source",
        url=r"https://example.org/story?x\=1\&y\=2",
        category="health",
    )
    assert item.url == "https://example.org/story?x=1&y=2"
    assert clean_headline("Ebola update - Example News") == "Ebola update"
