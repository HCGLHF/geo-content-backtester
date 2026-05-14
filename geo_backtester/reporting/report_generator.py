from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


def generate_report(
    output_dir: str | Path,
    score_summary: dict[str, object],
    query_rows: list[dict[str, object]],
    citation_rows: list[dict[str, object]],
    entity_results: dict[str, object],
    structure_results: dict[str, object],
    hybrid_results_by_version: dict[str, dict[str, list[object]]],
    failure_rows: list[dict[str, object]] | None = None,
    core_term_summary: dict[str, object] | None = None,
) -> Path:
    output_path = Path(output_dir)
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")
    html = template.render(
        score_summary=score_summary,
        query_rows=query_rows,
        citation_rows=sorted(citation_rows, key=lambda row: row["citation_score"], reverse=True),
        entity_results=entity_results,
        structure_results=structure_results,
        hybrid_results_by_version=hybrid_results_by_version,
        failure_rows=failure_rows or [],
        core_term_summary=core_term_summary or {},
    )
    report_path = output_path / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
