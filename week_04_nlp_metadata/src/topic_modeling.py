
import re
from pathlib import Path

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def clean_text(text):
    """Clean text before topic modeling."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\u00e2\u20ac\u201c", "-")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_corpus(file_path):
    """Load JSONL articles from the cleaned corpus."""
    import json

    documents = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                article = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_number}")
                continue

            article_id = article.get("id")
            title = clean_text(article.get("title", ""))
            content = clean_text(article.get("text", ""))

            combined_text = f"{title}. {content}".strip()

            if article_id is not None and len(combined_text.split()) >= 5:
                documents.append({
                    "id": str(article_id),
                    "title": title,
                    "text": combined_text,
                    "source": article.get("source", "Unknown"),
                    "url": article.get("url", "")
                })

    return documents


def train_topic_model(documents, n_topics=8):
    """Train an LDA model and return topic assignments."""
    if len(documents) < 2:
        raise ValueError("At least two valid documents are required.")

    texts = [document["text"] for document in documents]

    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=2,
        max_features=10000,
        ngram_range=(1, 2)
    )

    document_term_matrix = vectorizer.fit_transform(texts)

    if document_term_matrix.shape[1] == 0:
        raise ValueError("No usable vocabulary found in the corpus.")

    actual_topics = min(n_topics, len(documents))

    model = LatentDirichletAllocation(
        n_components=actual_topics,
        random_state=42,
        max_iter=10,
        learning_method="batch"
    )

    topic_distributions = model.fit_transform(document_term_matrix)
    feature_names = vectorizer.get_feature_names_out()

    topic_keywords = {}

    for topic_id, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-10:][::-1]
        topic_keywords[topic_id] = [
            feature_names[index] for index in top_indices
        ]

    results = []

    for index, document in enumerate(documents):
        topic_id = int(topic_distributions[index].argmax())
        confidence = float(topic_distributions[index][topic_id])

        results.append({
            **document,
            "topic_id": topic_id,
            "topic_probability": confidence,
            "topic_keywords": ", ".join(topic_keywords[topic_id])
        })

    return model, vectorizer, results, topic_keywords
