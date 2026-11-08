
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEEK2_PATH = (
    PROJECT_ROOT / "week_02_chunking_embeddings_retrieval"
)

DATABASE_PATH = WEEK2_PATH / "chroma_db"

EVALUATION_PATH = (
    PROJECT_ROOT
    / "week_05_fastapi_evaluation"
    / "evaluation"
)

TEST_SET_PATH = EVALUATION_PATH / "retrieval_test_set.json"
REPORT_PATH = EVALUATION_PATH / "retrieval_report.json"

sys.path.insert(0, str(WEEK2_PATH))

from src.embeddings import load_model
from src.vector_store import VectorStore


def calculate_precision(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant_retrieved = len(
        set(top_k) & set(relevant_ids)
    )

    return relevant_retrieved / k


def calculate_recall(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    relevant_retrieved = len(
        set(top_k) & set(relevant_ids)
    )

    return relevant_retrieved / len(relevant_ids)


def main():
    print("Loading evaluation dataset...")

    with open(TEST_SET_PATH, "r", encoding="utf-8") as file:
        test_data = json.load(file)

    print("Loading embedding model...")
    model = load_model()

    print("Connecting to existing ChromaDB...")
    vector_store = VectorStore(DATABASE_PATH)

    report = []

    for item in test_data:
        question = item["question"]
        relevant_ids = item["relevant_ids"]

        print("\n" + "=" * 60)
        print("Question:", question)

        query_embedding = model.encode(
            question,
            convert_to_numpy=True
        )

        results = vector_store.search(
            query_embedding,
            n_results=10
        )

        retrieved_ids = results["ids"][0]

        metrics = {
            "question": question,
            "precision_at_5": calculate_precision(
                retrieved_ids, relevant_ids, 5
            ),
            "recall_at_5": calculate_recall(
                retrieved_ids, relevant_ids, 5
            ),
            "precision_at_10": calculate_precision(
                retrieved_ids, relevant_ids, 10
            ),
            "recall_at_10": calculate_recall(
                retrieved_ids, relevant_ids, 10
            )
        }

        report.append(metrics)

        print("Precision@5:", round(metrics["precision_at_5"], 4))
        print("Recall@5:", round(metrics["recall_at_5"], 4))
        print("Precision@10:", round(metrics["precision_at_10"], 4))
        print("Recall@10:", round(metrics["recall_at_10"], 4))

    summary = {}

    for metric in [
        "precision_at_5",
        "recall_at_5",
        "precision_at_10",
        "recall_at_10"
    ]:
        summary[metric] = round(
            sum(item[metric] for item in report) / len(report),
            4
        )

    final_report = {
        "number_of_questions": len(test_data),
        "results": report,
        "average_metrics": summary
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(final_report, file, indent=4)

    print("\n" + "=" * 60)
    print("AVERAGE RETRIEVAL METRICS")
    print("=" * 60)

    for metric, value in summary.items():
        print(f"{metric}: {value}")

    print("\nReport saved to:", REPORT_PATH)


if __name__ == "__main__":
    main()
