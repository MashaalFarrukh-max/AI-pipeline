
"""
Process raw Wikipedia articles into a cleaned English-language corpus.

The script removes HTML and unnecessary whitespace, normalizes Unicode,
filters short and non-English documents, and removes duplicate text.
The cleaned documents are saved in JSONL format.
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.preprocessing import clean_text, is_english


RAW_FILE = PROJECT_DIR / "data" / "raw" / "wikipedia_articles.jsonl"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "clean_corpus.jsonl"


def process_data() -> None:
    """
    Clean and filter the raw Wikipedia dataset.

    Reads raw JSONL documents, cleans their text, keeps English documents
    with at least 100 characters, and removes duplicate cleaned text.

    The resulting documents are saved as JSONL using pandas.

    Returns:
        None. Saves the processed corpus and prints processing statistics.

    Raises:
        FileNotFoundError: If the raw input dataset does not exist.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_documents: list[dict] = []
    seen_texts: set[str] = set()
    total = 0

    with RAW_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                document = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            text = clean_text(document.get("text", ""))

            if len(text) < 100 or not is_english(text):
                continue

            if text in seen_texts:
                continue

            seen_texts.add(text)

            cleaned_documents.append({
                "id": document.get("id"),
                "title": document.get("title", ""),
                "text": text,
                "source": document.get("source", "Wikipedia"),
                "url": document.get("url", "")
            })

    df = pd.DataFrame(cleaned_documents)

    df.to_json(
        OUTPUT_FILE,
        orient="records",
        lines=True,
        force_ascii=False
    )

    print("Preprocessing completed!")
    print("Raw documents read:", total)
    print("Clean documents saved:", len(df))
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    process_data()
