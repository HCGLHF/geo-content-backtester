from __future__ import annotations

from geo_backtester.models import RetrievalResult


def answer_contexts_by_version(
    reranked_by_query: dict[str, list[RetrievalResult]],
) -> dict[str, dict[str, list[RetrievalResult]]]:
    contexts = {"old": {}, "new": {}}
    for query_id, results in reranked_by_query.items():
        for version in ["old", "new"]:
            contexts[version][query_id] = [
                result
                for result in results
                if result.article_version == version or result.source_type == "background"
            ][:5]
    return contexts


def display_by_version(
    reranked_by_query: dict[str, list[RetrievalResult]],
    version: str,
) -> dict[str, list[RetrievalResult]]:
    return {
        query_id: [result for result in results if result.article_version == version]
        for query_id, results in reranked_by_query.items()
    }


def core_term_score(core_term_summary: dict[str, object], version: str) -> float | None:
    if not core_term_summary:
        return None
    return float(core_term_summary.get(version, {}).get("core_term_score", 0.0))


def stuffing_risk(core_term_summary: dict[str, object], version: str) -> float | None:
    if not core_term_summary:
        return None
    return float(core_term_summary.get(version, {}).get("stuffing_risk_score", 0.0))


def config_summary(config: object) -> dict[str, object]:
    return {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "hybrid_alpha": config.hybrid_alpha,
        "use_openai_embeddings": config.use_openai_embeddings,
        "use_openai_reranker": config.use_openai_reranker,
        "answer_eval_enabled": config.has_openai_api_key,
    }


def failure_type_counts(failure_rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in failure_rows:
        failure_type = str(row["failure_type"])
        counts[failure_type] = counts.get(failure_type, 0) + 1
    return counts
