
"""Evaluate NER predictions against manually labeled entities."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "week_04_nlp_metadata")
)

from src.ner import NamedEntityExtractor


SAMPLE_PATH = (
    PROJECT_ROOT
    / "week_04_nlp_metadata"
    / "data"
    / "labeled_samples.json"
)


def normalize(entity_text: str) -> str:
    """Normalize entity text for consistent comparison.

    Args:
        entity_text: Entity text to normalize.

    Returns:
        Case-insensitive text with normalized whitespace.
    """
    return " ".join(entity_text.casefold().split())


def main() -> None:
    """Calculate precision, recall, and F1-score for NER predictions."""
    with open(SAMPLE_PATH, "r", encoding="utf-8") as file:
        samples = json.load(file)

    extractor = NamedEntityExtractor()

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for article in samples:
        text = article["title"] + ". " + article["text"]

        predictions = extractor.extract_entities(text)

        predicted_set = {
            (normalize(entity["text"]), entity["label"])
            for entity in predictions
        }

        manual_set = {
            (normalize(entity["text"]), entity["label"])
            for entity in article["manual_entities"]
        }

        true_positives += len(predicted_set & manual_set)
        false_positives += len(predicted_set - manual_set)
        false_negatives += len(manual_set - predicted_set)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives else 0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives else 0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print("\nNER Evaluation Results")
    print("=" * 40)
    print(f"Manually labeled entities matched: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-score:  {f1_score:.4f} ({f1_score * 100:.2f}%)")


if __name__ == "__main__":
    main()
