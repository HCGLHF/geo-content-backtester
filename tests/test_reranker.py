from geo_backtester.models import Query, RetrievalResult
from geo_backtester.retrieval.reranker import LocalReranker


def test_reranker_outputs_consecutive_stable_ranks() -> None:
    query = Query("q1", "AI citation readiness", "how_to", "geo_intro", "citation readiness", "high")
    candidates = [
        RetrievalResult("q1", query.query, "candidate", "new", 1, "c1", 0.1, "generic text", "geo_intro"),
        RetrievalResult("q1", query.query, "candidate", "new", 2, "c2", 0.2, "AI citation readiness checklist", "geo_intro"),
    ]
    for idx, candidate in enumerate(candidates, start=1):
        candidate.candidate_rank = idx
    reranked = LocalReranker(use_cross_encoder=False).rerank(query, candidates, top_k=2)
    assert [result.rerank_rank for result in reranked] == [1, 2]
    assert [result.rank for result in reranked] == [1, 2]
    assert reranked[0].chunk_id == "c2"


def test_core_term_boost_does_not_override_strong_semantic_score() -> None:
    query = Query("q1", "AI citation readiness", "how_to", "geo_intro", "citation readiness", "high")
    strong = RetrievalResult("q1", query.query, "candidate", "new", 1, "strong", 0.9, "AI citation readiness guide", "geo_intro")
    weak_stuffed = RetrievalResult("q1", query.query, "candidate", "new", 2, "stuffed", 0.1, "citation readiness citation readiness", "geo_intro")
    weak_stuffed.core_term_context_score = 1.0
    reranked = LocalReranker(use_cross_encoder=False, use_core_terms=True).rerank(query, [weak_stuffed, strong], top_k=2)
    assert reranked[0].chunk_id == "strong"
