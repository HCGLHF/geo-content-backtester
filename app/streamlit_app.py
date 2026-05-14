from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from geo_backtester.cli import run_backtest


st.set_page_config(page_title="GEO Content Backtester", layout="wide")
st.title("GEO Content Backtester")

old_file = st.file_uploader("Old article markdown", type=["md", "html", "txt"])
new_file = st.file_uploader("New article markdown", type=["md", "html", "txt"])
queries_file = st.file_uploader("queries.csv", type=["csv"])
entities_file = st.file_uploader("entity_list.json", type=["json"])

if st.button("Run backtest", disabled=not (old_file and new_file and queries_file)):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        old_path = tmp_path / "old_article.md"
        new_path = tmp_path / "new_article.md"
        queries_path = tmp_path / "queries.csv"
        entities_path = tmp_path / "entity_list.json"
        output_path = tmp_path / "run"

        old_path.write_bytes(old_file.getvalue())
        new_path.write_bytes(new_file.getvalue())
        queries_path.write_bytes(queries_file.getvalue())
        entity_arg = None
        if entities_file:
            entities_path.write_bytes(entities_file.getvalue())
            entity_arg = str(entities_path)

        result = run_backtest(str(old_path), str(new_path), str(queries_path), entity_arg, str(output_path))
        summary = result["score_summary"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Old GEO Score", summary["old"]["total_geo_score"])
        col2.metric("New GEO Score", summary["new"]["total_geo_score"])
        col3.metric("Winner", summary["improvement"]["winner"], summary["improvement"]["absolute_delta"])

        st.subheader("Score Summary")
        st.json(summary)

        retrieval_csv = (output_path / "retrieval_results.csv").read_bytes()
        report_html = (output_path / "report.html").read_bytes()
        st.download_button("Download retrieval_results.csv", retrieval_csv, "retrieval_results.csv")
        st.download_button("Download report.html", report_html, "report.html")
