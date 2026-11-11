
"""Manage persistent ChromaDB storage and similarity search."""

from pathlib import Path
from typing import Any

import chromadb


class VectorStore:
    """Store text chunks and their embeddings in a persistent ChromaDB collection."""

    def __init__(self, persist_directory: str | Path) -> None:
        """
        Initialize the persistent ChromaDB client and collection.

        Args:
            persist_directory: Directory where ChromaDB stores its database.
        """
        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name="wikipedia_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        chunks: list[str],
        embeddings: Any,
        metadatas: list[dict],
        ids: list[str]
    ) -> None:
        """
        Add or update text chunks in the vector database.

        Data is inserted in batches of 500 records.

        Args:
            chunks: Text content of the chunks.
            embeddings: Vector embeddings corresponding to each chunk.
            metadatas: Metadata dictionaries for each chunk.
            ids: Unique identifiers for the chunks.

        Raises:
            ValueError: If the input lists have different lengths.
        """
        if not (len(chunks) == len(embeddings) == len(metadatas) == len(ids)):
            raise ValueError("Chunks, embeddings, metadata, and IDs must have equal lengths")

        if not chunks:
            return

        for start in range(0, len(chunks), 500):
            end = start + 500

            self.collection.upsert(
                ids=ids[start:end],
                documents=chunks[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=metadatas[start:end]
            )

    def search(
        self,
        query_embedding: Any,
        n_results: int = 5
    ) -> dict[str, list]:
        """
        Find the most similar stored chunks for a query embedding.

        Args:
            query_embedding: Vector representation of the search query.
            n_results: Maximum number of matching chunks to return.

        Returns:
            A dictionary containing matching IDs, documents, metadata,
            and similarity distances. Returns empty lists if the
            collection contains no records.

        Raises:
            ValueError: If n_results is zero or negative.
        """
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero")

        if self.collection.count() == 0:
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
