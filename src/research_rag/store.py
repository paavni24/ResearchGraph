from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from research_rag.models import Chunk


class VectorStore:
    def __init__(self, path: Path, collection_name: str, embedding_model: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(path))
        self.collection: Collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
        )
        stored_model = (
            self.collection.metadata.get("embedding_model") if self.collection.metadata else None
        )
        if stored_model and stored_model != embedding_model:
            raise RuntimeError(
                f"Collection uses {stored_model}, but RAG_EMBEDDING_MODEL is {embedding_model}. "
                "Use a new RAG_COLLECTION name or restore the original model."
            )

    def replace_source(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have one embedding")
        self.collection.delete(where={"source": chunks[0].source})
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            embeddings=embeddings,
        )

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            {"id": chunk_id, "text": text, "metadata": metadata, "distance": distance}
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0], documents, metadatas, distances, strict=True
            )
        ]

    def all_chunks(self) -> list[dict[str, Any]]:
        result = self.collection.get(include=["documents", "metadatas"])
        documents = result["documents"] or []
        metadatas = result["metadatas"] or []
        return [
            {"id": chunk_id, "text": text, "metadata": metadata}
            for chunk_id, text, metadata in zip(
                result["ids"], documents, metadatas, strict=True
            )
        ]

    def count(self) -> int:
        return self.collection.count()
