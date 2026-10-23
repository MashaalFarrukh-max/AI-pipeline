
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from week_03_rag_llm_integration.src.rag_pipeline import RAGPipeline


@pytest.fixture
def pipeline():
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        with patch(
            "week_03_rag_llm_integration.src.rag_pipeline.load_model"
        ) as mock_load_model:
            with patch(
                "week_03_rag_llm_integration.src.rag_pipeline.VectorStore"
            ) as mock_vector_store:
                with patch(
                    "week_03_rag_llm_integration.src.rag_pipeline.Groq"
                ) as mock_groq:
                    embedding_model = MagicMock()
                    embedding_model.encode.return_value = np.array([0.1, 0.2])

                    mock_load_model.return_value = embedding_model

                    store = mock_vector_store.return_value
                    store.search.return_value = {
                        "ids": [["chunk-1"]],
                        "documents": [[
                            "Predictive learning uses new data to help "
                            "an AI model understand its environment."
                        ]],
                        "metadatas": [[{"title": "Predictive learning"}]],
                        "distances": [[0.20]],
                    }

                    groq_client = mock_groq.return_value
                    groq_client.chat.completions.create.return_value = (
                        MagicMock(
                            choices=[
                                MagicMock(
                                    message=MagicMock(
                                        content=(
                                            "Predictive learning uses new "
                                            "data to build understanding."
                                        )
                                    )
                                )
                            ]
                        )
                    )

                    instance = RAGPipeline(n_results=5, max_distance=0.65)
                    yield instance


def test_empty_question_raises_error(pipeline):
    with pytest.raises(ValueError, match="Question must not be empty"):
        pipeline.answer("")

    with pytest.raises(ValueError, match="Question must not be empty"):
        pipeline.answer("   ")


def test_relevant_question_returns_answer_and_sources(pipeline):
    result = pipeline.answer("What is predictive learning?")

    assert "Predictive learning" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["retrieval_seconds"] >= 0
    assert result["generation_seconds"] >= 0
    assert result["total_seconds"] >= 0


def test_unsupported_question_returns_fallback(pipeline):
    pipeline.vector_store.search.return_value = {
        "ids": [["chunk-1"]],
        "documents": [["An unrelated article about an ancient city."]],
        "metadatas": [[{"title": "Ancient city"}]],
        "distances": [[0.90]],
    }

    result = pipeline.answer("What is the capital of Mars?")

    assert "couldn't find sufficiently relevant information" in (
        result["answer"]
    )
    assert result["sources"] == []
    assert result["generation_seconds"] == 0.0

    pipeline.client.chat.completions.create.assert_not_called()


def test_empty_database_returns_fallback(pipeline):
    pipeline.vector_store.search.return_value = {
        "ids": [],
        "documents": [],
        "metadatas": [],
        "distances": [],
    }

    result = pipeline.answer("What is predictive learning?")

    assert result["sources"] == []
    assert result["generation_seconds"] == 0.0
    pipeline.client.chat.completions.create.assert_not_called()


def test_groq_failure_is_logged_and_raised(pipeline):
    pipeline.client.chat.completions.create.side_effect = RuntimeError(
        "Simulated API failure"
    )

    with pytest.raises(RuntimeError, match="Simulated API failure"):
        pipeline.answer("What is predictive learning?")
