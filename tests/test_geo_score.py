from geo_backtester.evaluation.geo_score import calculate_total_score


def test_score_weighting_with_answer_score_available() -> None:
    score = calculate_total_score(80, 70, 60, 50, answer_score=90)
    assert score == 74.5


def test_score_weighting_without_answer_score() -> None:
    score = calculate_total_score(80, 70, 60, 50, answer_score=None)
    assert score == 70.75
