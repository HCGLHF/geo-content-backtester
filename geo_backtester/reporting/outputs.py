from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from geo_backtester.models import RetrievalResult


def write_run_outputs(
    output: Path,
    run_id: str,
    all_results: list[RetrievalResult],
    citation_rows: list[dict[str, object]],
    answer_rows: list[dict[str, object]],
    score_summary: dict[str, object],
    entity_results: dict[str, object],
    structure_results: dict[str, object],
    failure_rows: list[dict[str, object]] | None = None,
    core_term_rows: list[dict[str, object]] | None = None,
    core_term_summary: dict[str, object] | None = None,
) -> None:
    retrieval_rows = []
    for result in all_results:
        row = result.to_dict()
        row["run_id"] = run_id
        row["chunk_text_preview"] = " ".join(result.text.split())[:260]
        row.pop("text", None)
        retrieval_rows.append(row)

    pd.DataFrame(retrieval_rows).to_csv(output / "retrieval_results.csv", index=False)
    pd.DataFrame(citation_rows).to_csv(output / "citation_results.csv", index=False)
    pd.DataFrame(answer_rows).to_csv(output / "answer_results.csv", index=False)
    pd.DataFrame(failure_rows or []).to_csv(output / "failure_analysis.csv", index=False)
    pd.DataFrame(core_term_rows or []).to_csv(output / "core_term_results.csv", index=False)
    (output / "score_summary.json").write_text(json.dumps(score_summary, indent=2), encoding="utf-8")
    (output / "entity_results.json").write_text(json.dumps(entity_results, indent=2), encoding="utf-8")
    (output / "structure_results.json").write_text(json.dumps(structure_results, indent=2), encoding="utf-8")
    (output / "core_term_summary.json").write_text(json.dumps(core_term_summary or {}, indent=2), encoding="utf-8")
