
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEK2_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(WEEK2_DIR))

from src.chunking import split_text
from src.embeddings import load_model, generate_embeddings
from src.vector_store import VectorStore


INPUT_FILE = (
    PROJECT_ROOT
    / "week_01_environment_data_preprocessing"
    / "data"
    / "processed"
    / "clean_corpus.jsonl"
)

DATABASE_DIR = WEEK2_DIR / "chroma_db"
LOG_FILE = WEEK2_DIR / "ingestion_log.txt"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Clean dataset not found: " + str(INPUT_FILE)
        )

    all_chunks = []
    all_metadatas = []
    all_ids = []

    print("Reading cleaned Wikipedia dataset...")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            try:
                document = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = document.get("text", "")
            chunks = split_text(text, chunk_size=500, overlap=50)

            for chunk_number, chunk in enumerate(chunks):
                chunk_id = str(document.get("id", line_number))
                chunk_id = chunk_id + "_" + str(chunk_number)

                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadatas.append({
                    "source": str(document.get("source", "Wikipedia")),
                    "title": str(document.get("title", ""))[:500],
                    "document_id": str(document.get("id", line_number))
                })

    if not all_chunks:
        raise ValueError("No text chunks were created from the dataset.")

    print("Total chunks:", len(all_chunks))

    model = load_model()
    embeddings, embedding_time = generate_embeddings(
        all_chunks,
        model=model,
        batch_size=32
    )

    store = VectorStore(DATABASE_DIR)

    store.add_chunks(
        all_chunks,
        embeddings,
        all_metadatas,
        all_ids
    )

    log_text = (
        "Total chunks: " + str(len(all_chunks)) + "\n"
        + "Embedding time: " + str(round(embedding_time, 2))
        + " seconds\n"
        + "Database count: " + str(store.collection.count()) + "\n"
    )

    LOG_FILE.write_text(log_text, encoding="utf-8")

    print("Database count:", store.collection.count())
    print("Log saved to:", LOG_FILE)
    print("Ingestion completed successfully!")


if __name__ == "__main__":
    main()
