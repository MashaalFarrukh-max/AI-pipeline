import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEEK2_DIR = PROJECT_ROOT / "week_02_chunking_embeddings_retrieval"

sys.path.insert(0, str(WEEK2_DIR))

import chromadb
from src.embeddings import load_model


DB_PATH = WEEK2_DIR / "chroma_db"
COLLECTION_NAME = "wikipedia_chunks"


def search_by_topic(query, topic_id, model, collection, n_results=5):
    """Search ChromaDB using semantic similarity and a topic filter."""

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        where={"topic_id": topic_id},
        include=["documents", "metadatas", "distances"]
    )

    return results


def display_results(results):
    """Display retrieved chunks with NLP metadata."""

    if not results["ids"] or not results["ids"][0]:
        print("\nNo matching results found.")
        return

    print("\nSearch Results")
    print("=" * 60)

    for index, chunk_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][index]
        document = results["documents"][0][index]
        distance = results["distances"][0][index]

        print(f"\nResult {index + 1}")
        print("-" * 60)
        print("Chunk ID:", chunk_id)
        print("Article:", metadata.get("title", "Unknown"))
        print("Topic:", metadata.get("topic_label", "Unknown"))
        print("Topic probability:", round(
            metadata.get("topic_probability", 0), 4
        ))
        print("Distance:", round(distance, 4))
        print("Entity types:", metadata.get("entity_types", "None"))
        print("Entities:", metadata.get("entities", "None")[:300])
        print("\nText:")
        print(document[:700])


def main():
    print("Loading ChromaDB...")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    print("Loading embedding model...")
    model = load_model()

    print("\nAvailable topics:")
    print("0 - Music")
    print("1 - Geography and Population")
    print("2 - Film and Entertainment")
    print("3 - Technology and General Concepts")
    print("4 - Military and History")
    print("5 - Science, Nature and Culture")
    print("6 - People, Education and Sports")
    print("7 - Places and Urban Geography")

    try:
        topic_id = int(input("\nEnter topic ID (0-7): "))

        if topic_id not in range(8):
            print("Please enter a topic ID between 0 and 7.")
            return

        query = input("Enter your search query: ").strip()

        if not query:
            print("Search query cannot be empty.")
            return

    except ValueError:
        print("Invalid topic ID. Please enter a number.")
        return

    print("\nSearching...")
    results = search_by_topic(
        query=query,
        topic_id=topic_id,
        model=model,
        collection=collection,
        n_results=5
    )

    display_results(results)


if __name__ == "__main__":
    main()