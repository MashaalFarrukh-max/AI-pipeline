
import chromadb


class VectorStore:
    def __init__(self, persist_directory):
        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name="wikipedia_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks, embeddings, metadatas, ids):
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

    def search(self, query_embedding, n_results=5):
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero")

        if self.collection.count() == 0:
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
