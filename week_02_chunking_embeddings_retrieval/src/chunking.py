
"""
Text chunking utilities for the Week 2 retrieval pipeline.

Splits long documents into smaller overlapping text chunks
for embedding generation and vector database storage.
"""


def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[str]:
    """
    Split text into smaller chunks with configurable overlap.

    Each chunk contains at most chunk_size characters. Consecutive
    chunks share overlap characters to preserve context between chunks.

    Args:
        text: Input document text.
        chunk_size: Maximum number of characters in each chunk.
            Defaults to 500.
        overlap: Number of characters shared between consecutive
            chunks. Defaults to 50.

    Returns:
        A list of non-empty text chunks. Returns an empty list
        when the input is not a string or contains only whitespace.

    Raises:
        ValueError: If chunk_size is not positive or overlap is
            negative or greater than or equal to chunk_size.
    """
    if not isinstance(text, str):
        return []

    text = text.strip()

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks
