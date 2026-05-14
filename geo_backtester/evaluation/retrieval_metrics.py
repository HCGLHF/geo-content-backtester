from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean

from geo_backtester.evaluation.relevance import RelevanceResolver, apply_relevance, keyword_overlap
from geo_backtester.models import Query, RetrievalResult


def is_relevant_result(result: RetrievalResult, query: Query, overlap_threshold: float = 0.22) -> bool:
    return result.article_id == query.target_article or keyword_overlap(query.expected_answer_points, result.text) >= overlap_threshold


def hit_at_k(results: list[RetrievalResult], k: int) -> int:
    return int(any(result.is_relevant for result in results[:k]))


def mrr(results: list[RetrievalResult]) -> float:
    for result in results:
        if result.is_relevant:
            return 1.0 / result.rank
    return 0.0


def precision_at_k(results: list[RetrievalResult], k: int) -> float:
    if k <= 0:
        return 0.0
    window = results[:k]
    if not window:
        return 0.0
    return sum(1 for result in window if result.is_relevant) / k


def recall_at_k(results: list[RetrievalResult], k: int, total_relevant: int | None = None) -> float:
    if total_relevant is None:
        total_relevant = sum(1 for result in results if result.is_relevant)
    if total_relevant <= 0:
        return 0.0
    return sum(1 for result in results[:k] if result.is_relevant) / total_relevant


def ndcg_at_k(results: list[RetrievalResult], k: int) -> float:
    grades = [int(result.relevance_grade or (3 if result.is_relevant else 0)) for result in results]
    if not grades:
        return 0.0

    def dcg(values: list[int]) -> float:
        return sum((2**grade - 1) / math.log2(idx + 2) for idx, grade in enumerate(values[:k]))

    ideal = dcg(sorted(grades, reverse=True))
    if ideal == 0:
        return 0.0
    return dcg(grades) / ideal


def first_relevant_rank(results: list[RetrievalResult]) -> int | None:
    for result in results:
        if result.is_relevant:
            return result.rank
    return None


def average_rank(results_by_query: dict[str, list[RetrievalResult]]) -> float | None:
    ranks = [first_relevant_rank(results) for results in results_by_query.values()]
    found_ranks = [rank for rank in ranks if rank is not None]
    if not found_ranks:
        return None
    return round(mean(found_ranks), 2)


def aggregate_retrieval_metrics(results_by_query: dict[str, list[RetrievalResult]]) -> dict[str, float | None]:
    if not results_by_query:
        return {
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "mrr": 0.0,
            "average_rank": None,
        }
    return {
        "hit_at_1": round(mean(hit_at_k(results, 1) for results in results_by_query.values()), 4),
        "hit_at_3": round(mean(hit_at_k(results, 3) for results in results_by_query.values()), 4),
        "hit_at_5": round(mean(hit_at_k(results, 5) for results in results_by_query.values()), 4),
        "mrr": round(mean(mrr(results) for results in results_by_query.values()), 4),
        "average_rank": average_rank(results_by_query),
    }


def aggregate_retrieval_metrics_graded(
    results_by_query: dict[str, list[RetrievalResult]],
    queries: list[Query],
    recall_k: int = 10,
) -> dict[str, float | None]:
    if not results_by_query:
        return {
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "precision_at_3": 0.0,
            "recall_at_10": 0.0,
            "ndcg_at_5": 0.0,
            "mrr": 0.0,
            "average_rank": None,
        }
    query_by_id = {query.query_id: query for query in queries}
    values = list(results_by_query.values())
    weights = [_priority_weight(query_by_id.get(query_id)) for query_id in results_by_query]
    return {
        "hit_at_1": round(_weighted_mean([hit_at_k(results, 1) for results in values], weights), 4),
        "hit_at_3": round(_weighted_mean([hit_at_k(results, 3) for results in values], weights), 4),
        "hit_at_5": round(_weighted_mean([hit_at_k(results, 5) for results in values], weights), 4),
        "precision_at_3": round(_weighted_mean([precision_at_k(results, 3) for results in values], weights), 4),
        "recall_at_10": round(_weighted_mean([recall_at_k(results, recall_k) for results in values], weights), 4),
        "ndcg_at_5": round(_weighted_mean([ndcg_at_k(results, 5) for results in values], weights), 4),
        "mrr": round(_weighted_mean([mrr(results) for results in values], weights), 4),
        "average_rank": average_rank(results_by_query),
    }


def annotate_results(
    results: list[RetrievalResult],
    query: Query,
    resolver: RelevanceResolver | None = None,
) -> list[RetrievalResult]:
    resolver = resolver or RelevanceResolver(allow_article_id_fallback=True)
    for result in results:
        apply_relevance(result, query, resolver)
    h1 = hit_at_k(results, 1)
    h3 = hit_at_k(results, 3)
    h5 = hit_at_k(results, 5)
    rr = mrr(results)
    for result in results:
        result.hit_at_1 = h1
        result.hit_at_3 = h3
        result.hit_at_5 = h5
        result.mrr = rr
    return results


def query_comparison(
    hybrid_results_by_version: dict[str, dict[str, list[RetrievalResult]]],
    queries: list[Query],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for query in queries:
        old_results = hybrid_results_by_version.get("old", {}).get(query.query_id, [])
        new_results = hybrid_results_by_version.get("new", {}).get(query.query_id, [])
        old_rank = first_relevant_rank(old_results)
        new_rank = first_relevant_rank(new_results)
        old_mrr = mrr(old_results)
        new_mrr = mrr(new_results)

        if old_rank is None and new_rank is None:
            winner = "tie"
            rank_delta = 0
        elif old_rank is None:
            winner = "new"
            rank_delta = 999
        elif new_rank is None:
            winner = "old"
            rank_delta = -999
        else:
            rank_delta = old_rank - new_rank
            winner = "new" if rank_delta > 0 else "old" if rank_delta < 0 else "tie"

        rows.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "old_rank": old_rank,
                "new_rank": new_rank,
                "rank_delta": rank_delta,
                "old_hit_at_1": hit_at_k(old_results, 1),
                "new_hit_at_1": hit_at_k(new_results, 1),
                "old_hit_at_3": hit_at_k(old_results, 3),
                "new_hit_at_3": hit_at_k(new_results, 3),
                "old_hit_at_5": hit_at_k(old_results, 5),
                "new_hit_at_5": hit_at_k(new_results, 5),
                "old_mrr": round(old_mrr, 4),
                "new_mrr": round(new_mrr, 4),
                "winner": winner,
            }
        )
    return rows


def retrieval_score(results_by_query: dict[str, list[RetrievalResult]]) -> float:
    if not results_by_query:
        return 0.0
    hit3_values = [hit_at_k(results, 3) for results in results_by_query.values()]
    hit5_values = [hit_at_k(results, 5) for results in results_by_query.values()]
    mrr_values = [mrr(results) for results in results_by_query.values()]
    ranks = [first_relevant_rank(results) for results in results_by_query.values()]
    rank_scores = [max(0.0, 1.0 - ((rank or 6) - 1) / 5) for rank in ranks]
    score = 35 * mean(hit3_values) + 20 * mean(hit5_values) + 30 * mean(mrr_values) + 15 * mean(rank_scores)
    return round(score, 2)


def retrieval_score_graded(results_by_query: dict[str, list[RetrievalResult]], queries: list[Query]) -> float:
    metrics = aggregate_retrieval_metrics_graded(results_by_query, queries)
    score = (
        float(metrics["ndcg_at_5"] or 0) * 35
        + float(metrics["recall_at_10"] or 0) * 25
        + float(metrics["precision_at_3"] or 0) * 20
        + float(metrics["mrr"] or 0) * 20
    )
    return round(score, 2)


def group_results_by_query(results: list[RetrievalResult]) -> dict[str, list[RetrievalResult]]:
    grouped: dict[str, list[RetrievalResult]] = defaultdict(list)
    for result in results:
        grouped[result.query_id].append(result)
    for query_results in grouped.values():
        query_results.sort(key=lambda item: item.rank)
    return dict(grouped)


def _priority_weight(query: Query | None) -> float:
    if not query:
        return 1.0
    return {"high": 3.0, "medium": 2.0, "low": 1.0}.get(query.priority.lower(), 2.0)


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    total_weight = sum(weights) or len(values)
    return sum(value * weight for value, weight in zip(values, weights, strict=False)) / total_weight
