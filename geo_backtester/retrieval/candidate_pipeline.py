from __future__ import annotations

from dataclasses import replace

from geo_backtester.config import BacktestConfig
from geo_backtester.evaluation.core_term_eval import CoreTermConfig, chunk_core_term_context_score
from geo_backtester.evaluation.relevance import RelevanceResolver, apply_relevance
from geo_backtester.models import Chunk, Query, RetrievalResult
from geo_backtester.retrieval.bm25_retriever import BM25Retriever
from geo_backtester.retrieval.embedding_retriever import EmbeddingRetriever
from geo_backtester.retrieval.hybrid_retriever import HybridRetriever
from geo_backtester.retrieval.reranker import LocalReranker


def run_realistic_retrieval(
    chunks: list[Chunk],
    queries: list[Query],
    config: BacktestConfig,
    resolver: RelevanceResolver,
    core_term_config: CoreTermConfig | None = None,
    candidate_depth: int = 20,
    rerank_depth: int = 10,
    final_depth: int = 5,
) -> tuple[list[RetrievalResult], dict[str, list[RetrievalResult]], dict[str, list[RetrievalResult]]]:
    bm25 = BM25Retriever(chunks)
    embedding = EmbeddingRetriever(chunks, config)
    hybrid = HybridRetriever(bm25, embedding, alpha=config.hybrid_alpha)
    core_term_config = core_term_config or CoreTermConfig()
    reranker = LocalReranker(
        use_openai=config.use_openai_reranker,
        openai_model=config.openai_rerank_model,
        use_core_terms=core_term_config.has_terms,
    )

    all_rows: list[RetrievalResult] = []
    candidates_by_query: dict[str, list[RetrievalResult]] = {}
    reranked_by_query: dict[str, list[RetrievalResult]] = {}

    for query in queries:
        candidates = []
        for result in hybrid.search(query, top_k=min(candidate_depth, len(chunks))):
            candidate = replace(result)
            candidate.retriever = "realistic_candidate"
            candidate.candidate_rank = candidate.rank
            candidate.core_term_context_score = chunk_core_term_context_score(
                _result_as_chunk_text(candidate),
                core_term_config,
                query,
            )
            apply_relevance(candidate, query, resolver)
            candidates.append(candidate)
        candidates_by_query[query.query_id] = candidates

        reranked = reranker.rerank(query, candidates, top_k=min(rerank_depth, len(candidates)))
        for result in reranked:
            apply_relevance(result, query, resolver)
            if result.rerank_rank and result.rerank_rank <= final_depth:
                result.final_rank = result.rerank_rank
            result.hit_at_1 = int(any(item.is_relevant for item in reranked[:1]))
            result.hit_at_3 = int(any(item.is_relevant for item in reranked[:3]))
            result.hit_at_5 = int(any(item.is_relevant for item in reranked[:5]))
        rr = _mrr(reranked)
        for result in reranked:
            result.mrr = rr
        reranked_by_query[query.query_id] = reranked
        all_rows.extend(reranked)

    return all_rows, candidates_by_query, reranked_by_query


def version_metric_view(results_by_query: dict[str, list[RetrievalResult]], version: str) -> dict[str, list[RetrievalResult]]:
    view: dict[str, list[RetrievalResult]] = {}
    for query_id, results in results_by_query.items():
        version_rows: list[RetrievalResult] = []
        for result in results:
            row = replace(result)
            if row.article_version != version:
                row.is_relevant = False
                row.relevance_grade = 0
                row.answer_support_grade = None
                row.citation_worthy = None
            version_rows.append(row)
        view[query_id] = version_rows
    return view


def _result_as_chunk_text(result: RetrievalResult) -> object:
    class _ChunkLike:
        text = result.text
        heading_path: list[str] = []

    return _ChunkLike()


def _mrr(results: list[RetrievalResult]) -> float:
    for result in results:
        if result.is_relevant:
            return 1.0 / result.rank
    return 0.0
