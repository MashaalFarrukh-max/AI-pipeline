
import time
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_model():
    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(texts, model=None, batch_size=32):
    if not texts:
        return [], 0.0

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if model is None:
        model = load_model()

    start_time = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    elapsed_time = time.perf_counter() - start_time

    print("Texts embedded:", len(texts))
    print("Embedding time:", round(elapsed_time, 2), "seconds")

    return embeddings, elapsed_time
