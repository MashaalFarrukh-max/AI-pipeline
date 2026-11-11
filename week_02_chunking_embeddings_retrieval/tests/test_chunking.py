
"""Unit tests for the text chunking functionality."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.chunking import split_text


def test_split_text_returns_chunks() -> None:
    """Check that long text is divided into chunks of the expected size."""
    text = "Hello world. " * 100
    chunks = split_text(text, chunk_size=100, overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunks_have_overlap() -> None:
    """Check that consecutive chunks share the specified overlap."""
    text = "abcdefghijklmnopqrstuvwxyz" * 5
    chunks = split_text(text, chunk_size=50, overlap=10)

    assert chunks[0][-10:] == chunks[1][:10]


def test_empty_text() -> None:
    """Check that empty text produces an empty list."""
    assert split_text("") == []


def test_invalid_input() -> None:
    """Check that non-string input produces an empty list."""
    assert split_text(None) == []


def test_invalid_chunk_size() -> None:
    """Check that a zero chunk size raises a ValueError."""
    try:
        split_text("Hello world", chunk_size=0)
        assert False
    except ValueError:
        assert True


def test_invalid_overlap() -> None:
    """Check that overlap equal to chunk size raises a ValueError."""
    try:
        split_text("Hello world", chunk_size=10, overlap=10)
        assert False
    except ValueError:
        assert True
