"""
Runtime retriever. Loaded once at pipeline startup, reused across turns.
"""

from model2vec import StaticModel
from qdrant_client import QdrantClient

from backend.config import settings


class BookRetriever:
    def __init__(self) -> None:
        self._model = StaticModel.from_pretrained(settings.embedding_model_name)
        if settings.qdrant_mode == "local":
            self._client = QdrantClient(url=settings.qdrant_url)
        else:
            self._client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        top_k = top_k or settings.retrieval_top_k
        vector = self._model.encode([query])[0].tolist()
        hits = self._client.query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            limit=top_k,
        ).points
        return [hit.payload["text"] for hit in hits]

    def build_context_block(self, query: str, top_k: int | None = None) -> str:
        passages = self.retrieve(query, top_k)
        if not passages:
            return ""
        joined = "\n\n---\n\n".join(passages)
        return (
            "Relevant passages from Pride and Prejudice:\n\n"
            f"{joined}\n\n"
            "Use these passages to ground your answer. If they don't contain "
            "the answer, say so honestly rather than inventing plot details."
        )
