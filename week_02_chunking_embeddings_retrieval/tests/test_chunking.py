
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.chunking import split_text


def test_split_text_returns_chunks():
    text = "Hello world. " * 100
    chunks = split_text(text, chunk_size=100, overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunks_have_overlap():
    text = "abcdefghijklmnopqrstuvwxyz" * 5
    chunks = split_text(text, chunk_size=50, overlap=10)

    assert chunks[0][-10:] == chunks[1][:10]


def test_empty_text():
    assert split_text("") == []


def test_invalid_input():
    assert split_text(None) == []


def test_invalid_chunk_size():
    try:
        split_text("Hello world", chunk_size=0)
        assert False
    except ValueError:
        assert True


def test_invalid_overlap():
    try:
        split_text("Hello world", chunk_size=10, overlap=10)
        assert False
    except ValueError:
        assert True
