
"""Train the LDA topic model and save article topics and keywords."""

import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_PATH = PROJECT_ROOT / "week_04_nlp_metadata"
sys.path.insert(0, str(SRC_PATH))

from src.topic_modeling import load_corpus, train_topic_model


CORPUS_PATH = (
    PROJECT_ROOT
    / "week_01_environment_data_preprocessing"
    / "data"
    / "processed"
    / "clean_corpus.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "week_04_nlp_metadata" / "data"


def main() -> None:
    """Train the LDA model and save topic modeling results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cleaned corpus...")
    documents = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(documents)} valid articles.")

    print("\nTraining LDA topic model...")
    model, vectorizer, results, topic_keywords = train_topic_model(
        documents,
        n_topics=8
    )

    csv_path = OUTPUT_DIR / "topic_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "title",
                "topic_id",
                "topic_probability",
                "topic_keywords"
            ]
        )
        writer.writeheader()

        for result in results:
            writer.writerow({
                "id": result["id"],
                "title": result["title"],
                "topic_id": result["topic_id"],
                "topic_probability": result["topic_probability"],
                "topic_keywords": result["topic_keywords"]
            })

    topics_path = OUTPUT_DIR / "topic_keywords.json"

    with open(topics_path, "w", encoding="utf-8") as file:
        json.dump(topic_keywords, file, indent=4, ensure_ascii=False)

    print("\nDiscovered Topics")
    print("=" * 60)

    for topic_id, keywords in topic_keywords.items():
        print(f"\nTopic {topic_id}:")
        print(", ".join(keywords))

    print("\nTopic modeling completed.")
    print(f"Article results saved to: {csv_path}")
    print(f"Topic keywords saved to: {topics_path}")


if __name__ == "__main__":
    main()
