
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "week_04_nlp_metadata")
)

from src.ner import NamedEntityExtractor


CORPUS_PATH = (
    PROJECT_ROOT
    / "week_01_environment_data_preprocessing"
    / "data"
    / "processed"
    / "clean_corpus.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "week_04_nlp_metadata"
    / "data"
    / "ner_results.jsonl"
)


def load_documents(file_path):
    documents = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                article = json.loads(line)
            except json.JSONDecodeError:
                continue

            article_id = article.get("id")
            title = article.get("title", "")
            text = article.get("text", "")

            if article_id is not None:
                documents.append({
                    "id": str(article_id),
                    "title": title,
                    "text": f"{title}. {text}"
                })

    return documents


def main():
    print("Loading corpus...")
    documents = load_documents(CORPUS_PATH)
    print(f"Loaded {len(documents)} articles.")

    print("\nLoading spaCy NER model...")
    extractor = NamedEntityExtractor()

    print("Extracting named entities...")
    entity_batches = extractor.extract_batch(documents)

    total_entities = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        for document, entities in zip(documents, entity_batches):
            result = {
                "id": document["id"],
                "title": document["title"],
                "entities": entities
            }

            total_entities += len(entities)

            file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

    print("\nNER completed successfully.")
    print(f"Articles processed: {len(documents)}")
    print(f"Total entities detected: {total_entities}")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
