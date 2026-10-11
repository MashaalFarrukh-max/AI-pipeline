
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


def test_strip_html():
    result = strip_html("<p>Hello <b>world</b></p>")
    assert result.strip() == "Hello world"


def test_normalize_unicode():
    result = normalize_unicode("cafe\u0301")
    assert result == "café"


def test_clean_whitespace():
    result = clean_whitespace("  Hello   world \n")
    assert result == "Hello world"


def test_is_english():
    assert is_english("This is a simple English sentence for testing.")
    assert not is_english("")


def test_clean_text():
    result = clean_text("<p>  Hello   world! </p>")
    assert result == "Hello world!"
