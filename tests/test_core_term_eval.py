from geo_backtester.evaluation.core_term_eval import (
    CoreTerm,
    CoreTermConfig,
    StuffingLimits,
    chunk_core_term_context_score,
    evaluate_chunk_core_terms,
    evaluate_core_terms_by_version,
    load_core_terms,
)
from geo_backtester.models import Chunk


def make_chunk(text: str, heading_path: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id="new_chunk_001",
        article_version="new",
        article_id="geo_intro",
        title="Title",
        heading_path=heading_path or ["What is GEO?"],
        text=text,
        token_count=100,
        start_index=0,
        end_index=len(text),
        source_type="new_article",
    )


def test_alias_matches_canonical_term(tmp_path) -> None:
    path = tmp_path / "core_terms.json"
    path.write_text(
        '{"must_have_terms":[{"term":"Generative Engine Optimization","aliases":["GEO"],"weight":10}]}',
        encoding="utf-8",
    )
    config = load_core_terms(path)
    row = evaluate_chunk_core_terms(make_chunk("GEO is the process of improving AI retrieval."), config)
    assert row["covered_core_terms"] == ["Generative Engine Optimization"]


def test_context_quality_beats_bare_stuffing() -> None:
    config = CoreTermConfig(
        [CoreTerm("citation readiness", weight=10, must_explain=True)],
        StuffingLimits(max_repeated_credit_per_term=2, penalty_threshold=4),
    )
    good = make_chunk("Citation readiness is the process of making content clear enough for AI systems to cite.")
    stuffed = make_chunk("citation readiness citation readiness citation readiness citation readiness")
    assert chunk_core_term_context_score(good, config) > chunk_core_term_context_score(stuffed, config)


def test_high_value_section_scores_higher() -> None:
    config = CoreTermConfig([CoreTerm("AI search visibility", weight=10, preferred_sections=["summary"])])
    summary = make_chunk("AI search visibility helps teams measure retrieval.", ["Summary"])
    body = make_chunk("AI search visibility helps teams measure retrieval.", ["Background"])
    assert evaluate_chunk_core_terms(summary, config)["high_value_section_terms"] == ["AI search visibility"]
    assert chunk_core_term_context_score(summary, config) >= chunk_core_term_context_score(body, config)


def test_repetition_has_penalty_and_saturation() -> None:
    config = CoreTermConfig(
        [CoreTerm("GEO", weight=10, must_explain=True)],
        StuffingLimits(max_repeated_credit_per_term=2, penalty_threshold=4),
    )
    clean = make_chunk("GEO is the process of improving how AI systems retrieve content.")
    stuffed = make_chunk("GEO GEO GEO GEO GEO GEO")
    _, clean_summary = evaluate_core_terms_by_version({"new": [clean]}, config)
    _, stuffed_summary = evaluate_core_terms_by_version({"new": [stuffed]}, config)
    assert stuffed_summary["new"]["stuffing_risk_score"] > clean_summary["new"]["stuffing_risk_score"]
    assert stuffed_summary["new"]["core_term_score"] <= clean_summary["new"]["core_term_score"]
