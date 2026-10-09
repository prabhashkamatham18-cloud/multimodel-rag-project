from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb

from src.embeddings import EmbeddingService
from src.models import RetrievedChunk, TextChunk


class VectorStore:
    def __init__(self, path: Path, embedding_model: str):
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            "visionrag_chunks", metadata={"hnsw:space": "cosine"}
        )
        self.embeddings = EmbeddingService(embedding_model)

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            return
        vectors = self.embeddings.embed([chunk.text for chunk in chunks])
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            embeddings=vectors,
        )

    def query(
        self, question: str, top_k: int, document_id: str | None = None
    ) -> list[RetrievedChunk]:
        vectors = self.embeddings.embed([question])
        where = {"document_id": document_id} if document_id else None
        results = self.collection.query(
            query_embeddings=vectors,
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        return [
            RetrievedChunk(text, metadata, float(distance))
            for text, metadata, distance in zip(documents, metadatas, distances, strict=True)
        ]

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    @staticmethod
    def make_chunk_id(document_id: str, page_number: int, kind: str, text: str) -> str:
        digest = hashlib.sha256(
            f"{document_id}:{page_number}:{kind}:{text}".encode("utf-8")
        ).hexdigest()
        return digest
