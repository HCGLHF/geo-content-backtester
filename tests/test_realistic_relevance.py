from geo_backtester.evaluation.relevance import RelevanceResolver, apply_relevance
from geo_backtester.evaluation.retrieval_metrics import ndcg_at_k, precision_at_k, recall_at_k
from geo_backtester.models import Query, RelevanceLabel, RetrievalResult


def make_query() -> Query:
    return Query("q1", "What is GEO?", "definition", "geo_intro", "discover retrieve cite summarize", "high")


def make_result(version: str = "new", chunk_id: str = "new_chunk_001", text: str = "unrelated") -> RetrievalResult:
    return RetrievalResult("q1", "What is GEO?", "hybrid", version, 1, chunk_id, 1.0, text, "geo_intro")


def test_manual_label_overrides_heuristic_relevance() -> None:
    query = make_query()
    result = make_result(text="discover retrieve cite summarize")
    resolver = RelevanceResolver(
        [RelevanceLabel("q1", "new", "new_chunk_001", 0, 0, False, "manual negative")],
        allow_article_id_fallback=False,
    )
    apply_relevance(result, query, resolver)
    assert result.is_relevant is False
    assert result.relevance_grade == 0
    assert result.label_source == "manual"


def test_no_label_uses_keyword_fallback() -> None:
    query = make_query()
    result = make_result(text="GEO helps AI systems discover retrieve cite and summarize content.")
    apply_relevance(result, query, RelevanceResolver(allow_article_id_fallback=False))
    assert result.is_relevant is True
    assert result.label_source == "heuristic"


def test_background_is_not_relevant_by_target_article_when_article_fallback_disabled() -> None:
    query = make_query()
    result = make_result(version="background_001", chunk_id="background_001_chunk_001", text="generic text")
    result.source_type = "background"
    result.article_id = "geo_intro"
    apply_relevance(result, query, RelevanceResolver(allow_article_id_fallback=False))
    assert result.is_relevant is False


def test_graded_metrics_calculate_precision_recall_ndcg() -> None:
    results = [
        make_result(chunk_id="c1"),
        make_result(chunk_id="c2"),
        make_result(chunk_id="c3"),
    ]
    for rank, result in enumerate(results, start=1):
        result.rank = rank
        result.relevance_grade = 3 if rank == 2 else 0
        result.is_relevant = rank == 2
    assert precision_at_k(results, 3) == 1 / 3
    assert recall_at_k(results, 3, total_relevant=1) == 1
    assert 0 < ndcg_at_k(results, 3) < 1
