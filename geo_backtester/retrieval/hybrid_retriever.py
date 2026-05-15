from __future__ import annotations

from geo_backtester.models import Query, RetrievalResult
from geo_backtester.retrieval.bm25_retriever import BM25Retriever
from geo_backtester.retrieval.embedding_retriever import EmbeddingRetriever


def minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    min_value = min(values.values())
    max_value = max(values.values())
    if max_value == min_value:
        return {key: 1.0 if max_value > 0 else 0.0 for key in values}
    return {key: (value - min_value) / (max_value - min_value) for key, value in values.items()}


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        embedding: EmbeddingRetriever,
        alpha: float = 0.45,
    ) -> None:
        self.bm25 = bm25
        self.embedding = embedding
        self.alpha = alpha

    def search(self, query: Query, top_k: int = 5) -> list[RetrievalResult]:
        depth = max(top_k, len(self.bm25.chunks))
        bm25_results = self.bm25.search(query, top_k=depth)
        embedding_results = self.embedding.search(query, top_k=depth)

        bm25_scores = {result.chunk_id: result.score for result in bm25_results}
        embedding_scores = {result.chunk_id: result.score for result in embedding_results}
        bm25_norm = minmax(bm25_scores)
        embedding_norm = minmax(embedding_scores)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in self.bm25.chunks}

        all_ids = set(bm25_scores) | set(embedding_scores)
        scored = []
        for chunk_id in all_ids:
            score = self.alpha * bm25_norm.get(chunk_id, 0.0) + (1 - self.alpha) * embedding_norm.get(chunk_id, 0.0)
            scored.append((chunk_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)

        results: list[RetrievalResult] = []
        for rank, (chunk_id, score) in enumerate(scored[:top_k], start=1):
            chunk = chunk_by_id[chunk_id]
            results.append(
                RetrievalResult(
                    query_id=query.query_id,
                    query=query.query,
                    retriever="hybrid",
                    article_version=chunk.article_version,
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    score=float(score),
                    text=chunk.text,
                    article_id=chunk.article_id,
                    source_type=chunk.source_type,
                    title=chunk.title,
                    heading_path=chunk.heading_path,
                )
            )
        return results
