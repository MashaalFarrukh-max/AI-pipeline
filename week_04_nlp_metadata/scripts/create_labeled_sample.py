
"""Create a reproducible sample of articles for manual entity labeling."""

import json
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "week_01_environment_data_preprocessing"
    / "data"
    / "processed"
    / "clean_corpus.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "week_04_nlp_metadata"
    / "data"
    / "labeled_samples.json"
)


def main() -> None:
    """Select 10 random articles and save them for manual labeling."""
    random.seed(42)

    articles: list[dict[str, Any]] = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            try:
                articles.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    sample = random.sample(articles, 10)

    results: list[dict[str, Any]] = []

    for article in sample:
        results.append({
            "id": article["id"],
            "title": article["title"],
            "text": article["text"][:600],
            "manual_entities": []
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("Created 10 labeled samples.")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
