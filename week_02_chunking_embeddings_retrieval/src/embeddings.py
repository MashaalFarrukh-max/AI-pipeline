
"""Load the sentence-transformer model and generate text embeddings."""

import time

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_model() -> SentenceTransformer:
    """Load and return the configured sentence-transformer model."""
    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(
    texts: list[str],
    model: SentenceTransformer | None = None,
    batch_size: int = 32
) -> tuple:
    """
    Generate vector embeddings for a list of texts.

    Args:
        texts: Text strings to convert into embeddings.
        model: An optional preloaded sentence-transformer model.
        batch_size: Number of texts processed in each batch.

    Returns:
        A tuple containing the generated embeddings and elapsed time
        in seconds. Returns an empty list and 0.0 for empty input.

    Raises:
        ValueError: If batch_size is zero or negative.
    """
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
