
import sys
from pathlib import Path

WEEK2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK2_DIR))

from src.embeddings import load_model
from src.vector_store import VectorStore


DATABASE_DIR = WEEK2_DIR / "chroma_db"


def main():
    store = VectorStore(DATABASE_DIR)

    if store.collection.count() == 0:
        print("The database is empty. Run ingest_data.py first.")
        return

    model = load_model()

    print("Semantic Search")
    print("Type 'exit' to stop.")

    while True:
        query = input("\nEnter your search query: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a search query.")
            continue

        query_embedding = model.encode(
            query,
            convert_to_numpy=True
        )

        results = store.search(query_embedding, n_results=5)

        print("\nSearch Results:")

        for index, document in enumerate(results["documents"][0], start=1):
            metadata = results["metadatas"][0][index - 1]
            distance = results["distances"][0][index - 1]

            print("\nResult", index)
            print("Title:", metadata.get("title", "Unknown"))
            print("Distance:", round(distance, 4))
            print("Text:", document[:500])


if __name__ == "__main__":
    main()
