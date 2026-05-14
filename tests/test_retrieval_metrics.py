from geo_backtester.evaluation.retrieval_metrics import hit_at_k, mrr
from geo_backtester.models import RetrievalResult


def make_result(rank: int, relevant: bool) -> RetrievalResult:
    return RetrievalResult("q1", "query", "hybrid", "new", rank, f"c{rank}", 1.0, "text", "geo_intro", relevant)


def test_hit_at_k_calculation() -> None:
    results = [make_result(1, False), make_result(2, True)]
    assert hit_at_k(results, 1) == 0
    assert hit_at_k(results, 3) == 1


def test_mrr_calculation() -> None:
    results = [make_result(1, False), make_result(2, True)]
    assert mrr(results) == 0.5


def test_no_relevant_result_returns_zero_mrr() -> None:
    results = [make_result(1, False), make_result(2, False)]
    assert mrr(results) == 0
