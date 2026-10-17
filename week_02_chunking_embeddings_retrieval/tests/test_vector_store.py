import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
import numpy as np
import pytest

from src.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "test_db")


def test_empty_database(store):
    assert store.collection.count() == 0

    results = store.search(np.array([0.1, 0.2, 0.3]))

    assert results["documents"] == []
    assert results["ids"] == []


def test_add_and_search_chunks(store):
    chunks = [
        "Cats are small domestic animals.",
        "Python is a programming language."
    ]

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])

    metadatas = [
        {"title": "Cats"},
        {"title": "Python"}
    ]

    ids = ["chunk_1", "chunk_2"]

    store.add_chunks(chunks, embeddings, metadatas, ids)

    assert store.collection.count() == 2

    results = store.search(
        np.array([1.0, 0.0, 0.0]),
        n_results=1
    )

    assert len(results["documents"][0]) == 1
    assert results["documents"][0][0] == chunks[0]


def test_mismatched_lengths(store):
    with pytest.raises(ValueError):
        store.add_chunks(
            ["Hello"],
            np.array([[0.1, 0.2, 0.3]]),
            [],
            ["chunk_1"]
        )


def test_invalid_result_count(store):
    with pytest.raises(ValueError):
        store.search(np.array([0.1, 0.2, 0.3]), n_results=0)
