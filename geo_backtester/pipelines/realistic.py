from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from geo_backtester.chunking.chunker import chunk_article
from geo_backtester.config import config_from_env
from geo_backtester.evaluation.answer_eval import evaluate_answers_with_openai, skip_answer_evaluation
from geo_backtester.evaluation.citation_eval import evaluate_citations
from geo_backtester.evaluation.core_term_eval import evaluate_core_terms_by_version, load_core_terms
from geo_backtester.evaluation.entity_eval import evaluate_entities, flatten_entities, load_entity_list
from geo_backtester.evaluation.failure_analysis import build_failure_analysis
from geo_backtester.evaluation.geo_score import build_score_summary, calculate_total_score
from geo_backtester.evaluation.relevance import RelevanceResolver
from geo_backtester.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics_graded,
    query_comparison,
    retrieval_score_graded,
)
from geo_backtester.evaluation.structure_eval import evaluate_structure
from geo_backtester.ingestion.loader import (
    load_article,
    load_background_articles,
    load_queries,
    load_relevance_labels,
)
from geo_backtester.pipelines.common import (
    answer_contexts_by_version,
    config_summary,
    core_term_score,
    display_by_version,
    failure_type_counts,
    stuffing_risk,
)
from geo_backtester.reporting.outputs import write_run_outputs
from geo_backtester.reporting.report_generator import generate_report
from geo_backtester.retrieval.candidate_pipeline import run_realistic_retrieval, version_metric_view


def run_realistic_backtest(
    old_path: str,
    new_path: str,
    queries_path: str,
    entities_path: str | None,
    output_dir: str,
    labels_path: str | None,
    background_corpus: str | None,
    core_terms_path: str | None,
) -> dict[str, object]:
    config = config_from_env()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    old_article = load_article(old_path, "old")
    new_article = load_article(new_path, "new")
    background_articles = load_background_articles(background_corpus)
    queries = load_queries(queries_path)
    labels = load_relevance_labels(labels_path)
    core_term_config = load_core_terms(core_terms_path)
    entity_list = load_entity_list(entities_path)
    entity_terms = flatten_entities(entity_list)
    resolver = RelevanceResolver(labels, allow_article_id_fallback=False)

    old_chunks = chunk_article(old_article, config.chunk_size, config.chunk_overlap, "old_article")
    new_chunks = chunk_article(new_article, config.chunk_size, config.chunk_overlap, "new_article")
    background_chunks = [
        chunk
        for article in background_articles
        for chunk in chunk_article(article, config.chunk_size, config.chunk_overlap, "background")
    ]
    all_chunks = old_chunks + new_chunks + background_chunks

    all_results, candidates_by_query, reranked_by_query = run_realistic_retrieval(
        all_chunks,
        queries,
        config,
        resolver,
        core_term_config,
        candidate_depth=20,
        rerank_depth=10,
        final_depth=5,
    )

    old_view = version_metric_view(reranked_by_query, "old")
    new_view = version_metric_view(reranked_by_query, "new")
    metric_views = {"old": old_view, "new": new_view}
    query_rows = query_comparison(metric_views, queries)
    old_retrieval_score = retrieval_score_graded(old_view, queries)
    new_retrieval_score = retrieval_score_graded(new_view, queries)

    old_citation_rows, old_citation_score = evaluate_citations(old_chunks, entity_terms)
    new_citation_rows, new_citation_score = evaluate_citations(new_chunks, entity_terms)
    citation_rows = old_citation_rows + new_citation_rows
    core_term_rows, core_term_summary = evaluate_core_terms_by_version(
        {"old": old_chunks, "new": new_chunks},
        core_term_config,
    )

    entity_results = {
        "old": evaluate_entities(old_article.text, entity_list),
        "new": evaluate_entities(new_article.text, entity_list),
    }
    structure_results = {
        "old": evaluate_structure(old_article.text),
        "new": evaluate_structure(new_article.text),
    }

    answer_contexts = answer_contexts_by_version(reranked_by_query)
    if config.has_openai_api_key:
        answer_rows, answer_scores = evaluate_answers_with_openai(queries, answer_contexts)
    else:
        answer_rows, answer_scores = skip_answer_evaluation(queries, ["old", "new"])

    failure_rows = build_failure_analysis(
        queries,
        candidates_by_query,
        reranked_by_query,
        citation_rows,
        answer_rows,
        versions=["old", "new"],
    )

    old_scores = {
        "retrieval_score": old_retrieval_score,
        "citation_score": old_citation_score,
        "answer_score": answer_scores.get("old"),
        "core_term_score": core_term_score(core_term_summary, "old"),
        "entity_score": entity_results["old"]["entity_score"],
        "structure_score": structure_results["old"]["structure_score"],
    }
    old_scores["total_geo_score"] = calculate_total_score(**old_scores, mode="realistic")
    new_scores = {
        "retrieval_score": new_retrieval_score,
        "citation_score": new_citation_score,
        "answer_score": answer_scores.get("new"),
        "core_term_score": core_term_score(core_term_summary, "new"),
        "entity_score": entity_results["new"]["entity_score"],
        "structure_score": structure_results["new"]["structure_score"],
    }
    new_scores["total_geo_score"] = calculate_total_score(**new_scores, mode="realistic")

    score_summary = build_score_summary(old_scores, new_scores)
    score_summary.update(
        {
            "mode": "realistic",
            "query_count": len(queries),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "retrieval_metrics": {
                "old": aggregate_retrieval_metrics_graded(old_view, queries),
                "new": aggregate_retrieval_metrics_graded(new_view, queries),
            },
            "label_coverage": resolver.label_coverage(queries),
            "background_corpus_size": len(background_articles),
            "background_chunk_count": len(background_chunks),
            "failure_type_counts": failure_type_counts(failure_rows),
            "core_term_enabled": core_term_config.has_terms,
            "core_term_summary": core_term_summary,
            "stuffing_risk_score": {
                "old": stuffing_risk(core_term_summary, "old"),
                "new": stuffing_risk(core_term_summary, "new"),
            },
            "config_used": {
                **config_summary(config),
                "candidate_depth": 20,
                "rerank_depth": 10,
                "final_depth": 5,
            },
        }
    )

    write_run_outputs(
        output,
        output.name,
        all_results,
        citation_rows,
        answer_rows,
        score_summary,
        entity_results,
        structure_results,
        failure_rows,
        core_term_rows,
        core_term_summary,
    )
    report_path = generate_report(
        output,
        score_summary,
        query_rows,
        citation_rows,
        entity_results,
        structure_results,
        {
            "old": display_by_version(reranked_by_query, "old"),
            "new": display_by_version(reranked_by_query, "new"),
        },
        failure_rows,
        core_term_summary,
    )
    return {"output_dir": str(output), "report_path": str(report_path), "score_summary": score_summary}
