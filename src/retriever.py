from __future__ import annotations

from src.models import Citation, RetrievedChunk
from src.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, top_k: int, relevance_threshold: float):
        self.vector_store = vector_store
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    def retrieve(self, question: str, document_id: str | None = None) -> list[RetrievedChunk]:
        chunks = self.vector_store.query(question, self.top_k, document_id)
        return [chunk for chunk in chunks if chunk.distance <= 1 - self.relevance_threshold]

    @staticmethod
    def citations(chunks: list[RetrievedChunk]) -> list[Citation]:
        seen: set[tuple[str, int]] = set()
        citations: list[Citation] = []
        for chunk in chunks:
            metadata = chunk.metadata
            key = (str(metadata["document_id"]), int(metadata["page_number"]))
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    document_id=key[0],
                    document_name=str(metadata["document_name"]),
                    page_number=key[1],
                    excerpt=chunk.text[:360],
                    image_path=str(metadata.get("image_path", "")),
                )
            )
        return citations
