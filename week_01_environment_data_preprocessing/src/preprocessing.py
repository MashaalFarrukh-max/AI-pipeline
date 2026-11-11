
import re
import unicodedata

from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException


def strip_html(text: str) -> str:
    """
    Remove HTML tags from text and clean extra whitespace.

    Args:
        text: Input text that may contain HTML markup.

    Returns:
        Text without HTML tags and unnecessary whitespace.
        Returns an empty string if the input is not a string.
    """
    if not isinstance(text, str):
        return ""

    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return clean_whitespace(text)


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters using NFKC normalization.

    Args:
        text: Input text to normalize.

    Returns:
        Unicode-normalized text, or an empty string for invalid input.
    """
    if not isinstance(text, str):
        return ""

    return unicodedata.normalize("NFKC", text)


def clean_whitespace(text: str) -> str:
    """
    Replace repeated whitespace with single spaces and trim the text.

    Args:
        text: Input text to clean.

    Returns:
        Text with normalized whitespace, or an empty string for invalid input.
    """
    if not isinstance(text, str):
        return ""

    return re.sub(r"\s+", " ", text).strip()


def is_english(text: str) -> bool:
    """
    Detect whether the input text is written in English.

    Text shorter than 20 characters is rejected to reduce unreliable
    language detection results.

    Args:
        text: Text whose language should be detected.

    Returns:
        True if the detected language is English; otherwise, False.
    """
    if not isinstance(text, str) or len(text.strip()) < 20:
        return False

    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def clean_text(text: str) -> str:
    """
    Apply the complete text-cleaning pipeline.

    The pipeline removes HTML, normalizes Unicode, and cleans whitespace.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text ready for downstream processing.
    """
    text = strip_html(text)
    text = normalize_unicode(text)
    text = clean_whitespace(text)

    return text
