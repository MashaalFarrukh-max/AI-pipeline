import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEK4_DIR = PROJECT_ROOT / "week_04_nlp_metadata"

sys.path.insert(0, str(WEEK4_DIR))

from src.topic_modeling import clean_text, train_topic_model


def test_clean_text_removes_extra_whitespace():
    text = "  Artificial   Intelligence   "
    assert clean_text(text) == "Artificial Intelligence"


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""


def test_clean_text_removes_html():
    text = "<p>Hello World</p>"
    assert clean_text(text) == "Hello World"


def test_train_topic_model_returns_results():
    documents = [
        {"id": "1", "title": "Music", "text": "music songs album band music songs"},
        {"id": "2", "title": "Music", "text": "music band guitar album music"},
        {"id": "3", "title": "Technology", "text": "computer software technology systems"},
        {"id": "4", "title": "Technology", "text": "software computer systems technology"}
    ]

    model, vectorizer, results, keywords = train_topic_model(
        documents,
        n_topics=2
    )

    assert len(results) == 4
    assert len(keywords) == 2
    assert "topic_id" in results[0]
    assert "topic_probability" in results[0]


def test_train_topic_model_requires_two_documents():
    documents = [
        {"id": "1", "title": "Music", "text": "music songs album"}
    ]

    with pytest.raises(ValueError):
        train_topic_model(documents)