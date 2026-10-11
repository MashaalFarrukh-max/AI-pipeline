
import re
import unicodedata
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException


def strip_html(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return clean_whitespace(text)


def normalize_unicode(text):
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


def clean_whitespace(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def is_english(text):
    if not isinstance(text, str) or len(text.strip()) < 20:
        return False

    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def clean_text(text):
    text = strip_html(text)
    text = normalize_unicode(text)
    text = clean_whitespace(text)
    return text
