
"""Enrich ChromaDB chunks with topic modeling and NER metadata."""

import json
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEEK2_DIR = PROJECT_ROOT / "week_02_chunking_embeddings_retrieval"
WEEK4_DIR = PROJECT_ROOT / "week_04_nlp_metadata"

DB_PATH = WEEK2_DIR / "chroma_db"
TOPIC_RESULTS_PATH = WEEK4_DIR / "data" / "topic_results.csv"
TOPIC_LABELS_PATH = WEEK4_DIR / "data" / "topic_labels.json"
NER_RESULTS_PATH = WEEK4_DIR / "data" / "ner_results.jsonl"


def load_topic_metadata() -> dict[str, dict[str, Any]]:
    """Load topic assignments and labels for each article.

    Returns:
        A dictionary mapping article IDs to their topic metadata.
    """
    topic_df = pd.read_csv(TOPIC_RESULTS_PATH)

    with open(TOPIC_LABELS_PATH, "r", encoding="utf-8") as file:
        topic_labels = json.load(file)

    topic_metadata: dict[str, dict[str, Any]] = {}

    for _, row in topic_df.iterrows():
        article_id = str(row["id"])
        topic_id = int(row["topic_id"])

        topic_metadata[article_id] = {
            "topic_id": topic_id,
            "topic_label": topic_labels.get(str(topic_id), "Unknown"),
            "topic_probability": float(row["topic_probability"]),
            "topic_keywords": str(row["topic_keywords"])
        }

    return topic_metadata


def load_ner_metadata() -> dict[str, dict[str, str]]:
    """Load named entities and entity types for each article.

    Returns:
        A dictionary mapping article IDs to their NER metadata.
    """
    ner_metadata: dict[str, dict[str, str]] = {}

    with open(NER_RESULTS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            article_id = str(record["id"])

            entities = record.get("entities", [])

            entity_texts = sorted({
                entity["text"].strip()
                for entity in entities
                if entity.get("text", "").strip()
            })

            entity_types = sorted({
                entity["label"]
                for entity in entities
                if entity.get("label")
            })

            ner_metadata[article_id] = {
                "entities": " | ".join(entity_texts[:30]),
                "entity_types": " | ".join(entity_types)
            }

    return ner_metadata


def main() -> None:
    """Combine topic and NER metadata and update ChromaDB in batches."""
    print("Loading topic metadata...")
    topic_metadata = load_topic_metadata()

    print("Loading NER metadata...")
    ner_metadata = load_ner_metadata()

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection("wikipedia_chunks")

    total_chunks = collection.count()
    batch_size = 500
    updated_chunks = 0
    skipped_chunks = 0

    print(f"Total chunks: {total_chunks}")
    print("Preparing metadata updates...")

    for offset in range(0, total_chunks, batch_size):
        records = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"]
        )

        ids = []
        updated_metadatas = []

        for chunk_id, metadata in zip(
            records["ids"],
            records["metadatas"]
        ):
            article_id = str(metadata.get("document_id", ""))

            topic_info = topic_metadata.get(article_id)
            ner_info = ner_metadata.get(article_id)

            if topic_info is None or ner_info is None:
                skipped_chunks += 1
                continue

            new_metadata = {
                **metadata,
                **topic_info,
                **ner_info
            }

            ids.append(chunk_id)
            updated_metadatas.append(new_metadata)

        if ids:
            collection.update(
                ids=ids,
                metadatas=updated_metadatas
            )
            updated_chunks += len(ids)

        print(
            f"Processed {min(offset + batch_size, total_chunks)}"
            f"/{total_chunks} chunks"
        )

    print("\nMetadata enrichment completed.")
    print(f"Updated chunks: {updated_chunks}")
    print(f"Skipped chunks: {skipped_chunks}")


if __name__ == "__main__":
    main()
