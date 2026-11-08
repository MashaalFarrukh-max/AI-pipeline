
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEEK2_PATH = (
    PROJECT_ROOT / "week_02_chunking_embeddings_retrieval"
)

DATABASE_PATH = WEEK2_PATH / "chroma_db"

sys.path.insert(0, str(WEEK2_PATH))

from src.embeddings import load_model
from src.vector_store import VectorStore


def main():
    print("Loading embedding model...")
    model = load_model()

    print("Connecting to existing ChromaDB...")
    vector_store = VectorStore(DATABASE_PATH)

    questions = [
        "What is ISO 10007?",
        "What is machine learning?",
        "What is geography and population?",
    ]

    for question in questions:

        print("\n" + "=" * 80)
        print("QUESTION:", question)
        print("=" * 80)

        query_embedding = model.encode(
            question,
            convert_to_numpy=True
        )

        results = vector_store.search(
            query_embedding,
            n_results=10
        )

        retrieved_ids = results["ids"][0]

        if not retrieved_ids:
            print("No results found.")
            continue

        for index, chunk_id in enumerate(retrieved_ids):

            print(f"\nResult {index + 1}")

            print("Chunk ID:", chunk_id)

            print(
                "Distance:",
                round(results["distances"][0][index], 4)
            )

            print(
                "Text:",
                results["documents"][0][index][:500]
            )

            print(
                "Metadata:",
                results["metadatas"][0][index]
            )


if __name__ == "__main__":
    main()
