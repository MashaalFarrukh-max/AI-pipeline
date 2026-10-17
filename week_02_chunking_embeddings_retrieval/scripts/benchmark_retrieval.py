
import csv
import sys
import time
from pathlib import Path

WEEK2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK2_DIR))

from src.embeddings import load_model
from src.vector_store import VectorStore


DATABASE_DIR = WEEK2_DIR / "chroma_db"
RESULTS_FILE = WEEK2_DIR / "retrieval_benchmark.csv"

QUERIES = [
    "What is artificial intelligence?",
    "How does climate change affect the environment?",
    "History of computers and technology",
    "What are the benefits of education?",
    "How do plants grow?"
]


def main():
    store = VectorStore(DATABASE_DIR)

    if store.collection.count() == 0:
        print("Database is empty. Run ingest_data.py first.")
        return

    model = load_model()
    rows = []

    for query in QUERIES:
        query_embedding = model.encode(
            query,
            convert_to_numpy=True
        )

        start_time = time.perf_counter()
        results = store.search(query_embedding, n_results=5)
        latency_ms = (time.perf_counter() - start_time) * 1000

        result_count = len(results["documents"][0])

        rows.append({
            "query": query,
            "latency_ms": round(latency_ms, 3),
            "results_returned": result_count
        })

        print("Query:", query)
        print("Retrieval latency:", round(latency_ms, 3), "ms")
        print("Results returned:", result_count)
        print("-" * 40)

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["query", "latency_ms", "results_returned"]
        )
        writer.writeheader()
        writer.writerows(rows)

    average_latency = sum(row["latency_ms"] for row in rows) / len(rows)

    print("Average retrieval latency:", round(average_latency, 3), "ms")
    print("Benchmark results saved to:", RESULTS_FILE)


if __name__ == "__main__":
    main()
