from __future__ import annotations

from geo_backtester.pipelines.mvp import run_mvp_backtest
from geo_backtester.pipelines.realistic import run_realistic_backtest


def run_backtest(
    old_path: str,
    new_path: str,
    queries_path: str,
    entities_path: str | None,
    output_dir: str,
    mode: str = "mvp",
    labels_path: str | None = None,
    background_corpus: str | None = None,
    core_terms_path: str | None = None,
) -> dict[str, object]:
    if mode == "realistic":
        return run_realistic_backtest(
            old_path,
            new_path,
            queries_path,
            entities_path,
            output_dir,
            labels_path,
            background_corpus,
            core_terms_path,
        )
    return run_mvp_backtest(old_path, new_path, queries_path, entities_path, output_dir)
