
"""
Unit tests for the Week 1 text preprocessing functions.

Tests HTML removal, Unicode normalization, whitespace cleaning,
English language detection, and the complete text-cleaning pipeline.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.preprocessing import (
    strip_html,
    normalize_unicode,
    clean_whitespace,
    is_english,
    clean_text
)


def test_strip_html() -> None:
    """Verify that HTML tags are removed from input text."""
    result = strip_html("<p>Hello <b>world</b></p>")
    assert result.strip() == "Hello world"


def test_normalize_unicode() -> None:
    """Verify that equivalent Unicode characters are normalized."""
    result = normalize_unicode("cafe\u0301")
    assert result == "café"


def test_clean_whitespace() -> None:
    """Verify that repeated whitespace is replaced with single spaces."""
    result = clean_whitespace("  Hello   world \n")
    assert result == "Hello world"


def test_is_english() -> None:
    """Verify that English text is detected and empty text is rejected."""
    assert is_english("This is a simple English sentence for testing.")
    assert not is_english("")


def test_clean_text() -> None:
    """Verify that the complete text-cleaning pipeline works correctly."""
    result = clean_text("<p>  Hello   world! </p>")
    assert result == "Hello world!"