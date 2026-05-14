from pathlib import Path

from geo_backtester.cli import run_backtest


def test_realistic_demo_runs_without_openai_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "realistic"
    result = run_backtest(
        str(root / "data/articles/old/old_article.md"),
        str(root / "data/articles/new/new_article.md"),
        str(root / "data/queries/queries.csv"),
        str(root / "data/entities/entity_list.json"),
        str(output),
        mode="realistic",
        labels_path=str(root / "data/labels/relevance_labels.csv"),
        background_corpus=str(root / "data/corpus/background"),
    )
    assert Path(result["report_path"]).exists()
    assert (output / "retrieval_results.csv").exists()
    assert (output / "failure_analysis.csv").exists()
    assert result["score_summary"]["mode"] == "realistic"
    assert result["score_summary"]["label_coverage"]["label_count"] > 0


def test_realistic_demo_with_core_terms_writes_core_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "realistic_core"
    result = run_backtest(
        str(root / "data/articles/old/old_article.md"),
        str(root / "data/articles/new/new_article.md"),
        str(root / "data/queries/queries.csv"),
        str(root / "data/entities/entity_list.json"),
        str(output),
        mode="realistic",
        labels_path=str(root / "data/labels/relevance_labels.csv"),
        background_corpus=str(root / "data/corpus/background"),
        core_terms_path=str(root / "data/terms/core_terms.json"),
    )
    assert (output / "core_term_results.csv").exists()
    assert (output / "core_term_summary.json").exists()
    assert result["score_summary"]["core_term_enabled"] is True
    assert result["score_summary"]["new"]["core_term_score"] is not None
