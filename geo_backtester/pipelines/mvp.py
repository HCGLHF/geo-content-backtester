from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from geo_backtester.chunking.chunker import chunk_article
from geo_backtester.config import config_from_env
from geo_backtester.evaluation.answer_eval import evaluate_answers_with_openai, skip_answer_evaluation
from geo_backtester.evaluation.citation_eval import evaluate_citations
from geo_backtester.evaluation.entity_eval import evaluate_entities, flatten_entities, load_entity_list
from geo_backtester.evaluation.geo_score import build_score_summary, calculate_total_score
from geo_backtester.evaluation.relevance import RelevanceResolver
from geo_backtester.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    annotate_results,
    query_comparison,
    retrieval_score,
)
from geo_backtester.evaluation.structure_eval import evaluate_structure
from geo_backtester.ingestion.loader import load_article, load_queries
from geo_backtester.models import RetrievalResult
from geo_backtester.pipelines.common import config_summary
from geo_backtester.reporting.outputs import write_run_outputs
from geo_backtester.reporting.report_generator import generate_report
from geo_backtester.retrieval.bm25_retriever import BM25Retriever
from geo_backtester.retrieval.embedding_retriever import EmbeddingRetriever
from geo_backtester.retrieval.hybrid_retriever import HybridRetriever


def run_mvp_backtest(
    old_path: str,
    new_path: str,
    queries_path: str,
    entities_path: str | None,
    output_dir: str,
) -> dict[str, object]:
    config = config_from_env()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    old_article = load_article(old_path, "old")
    new_article = load_article(new_path, "new")
    queries = load_queries(queries_path)
    entity_list = load_entity_list(entities_path)
    entity_terms = flatten_entities(entity_list)

    chunks_by_version = {
        "old": chunk_article(old_article, config.chunk_size, config.chunk_overlap, "old_article"),
        "new": chunk_article(new_article, config.chunk_size, config.chunk_overlap, "new_article"),
    }

    all_results: list[RetrievalResult] = []
    hybrid_results_by_version: dict[str, dict[str, list[RetrievalResult]]] = {"old": {}, "new": {}}
    resolver = RelevanceResolver(allow_article_id_fallback=True)

    for version, chunks in chunks_by_version.items():
        bm25 = BM25Retriever(chunks)
        embedding = EmbeddingRetriever(chunks, config)
        hybrid = HybridRetriever(bm25, embedding, alpha=config.hybrid_alpha)
        for query in queries:
            for retriever_results in [
                bm25.search(query, config.top_k),
                embedding.search(query, config.top_k),
                hybrid.search(query, config.top_k),
            ]:
                annotated = annotate_results(retriever_results, query, resolver)
                all_results.extend(annotated)
                if annotated and annotated[0].retriever == "hybrid":
                    hybrid_results_by_version[version][query.query_id] = annotated

    query_rows = query_comparison(hybrid_results_by_version, queries)
    old_retrieval_score = retrieval_score(hybrid_results_by_version["old"])
    new_retrieval_score = retrieval_score(hybrid_results_by_version["new"])

    old_citation_rows, old_citation_score = evaluate_citations(chunks_by_version["old"], entity_terms)
    new_citation_rows, new_citation_score = evaluate_citations(chunks_by_version["new"], entity_terms)
    citation_rows = old_citation_rows + new_citation_rows

    entity_results = {
        "old": evaluate_entities(old_article.text, entity_list),
        "new": evaluate_entities(new_article.text, entity_list),
    }
    structure_results = {
        "old": evaluate_structure(old_article.text),
        "new": evaluate_structure(new_article.text),
    }

    if config.has_openai_api_key:
        answer_rows, answer_scores = evaluate_answers_with_openai(queries, hybrid_results_by_version)
    else:
        answer_rows, answer_scores = skip_answer_evaluation(queries, ["old", "new"])

    old_scores = {
        "retrieval_score": old_retrieval_score,
        "citation_score": old_citation_score,
        "answer_score": answer_scores.get("old"),
        "entity_score": entity_results["old"]["entity_score"],
        "structure_score": structure_results["old"]["structure_score"],
    }
    old_scores["total_geo_score"] = calculate_total_score(**old_scores)
    new_scores = {
        "retrieval_score": new_retrieval_score,
        "citation_score": new_citation_score,
        "answer_score": answer_scores.get("new"),
        "entity_score": entity_results["new"]["entity_score"],
        "structure_score": structure_results["new"]["structure_score"],
    }
    new_scores["total_geo_score"] = calculate_total_score(**new_scores)

    score_summary = build_score_summary(old_scores, new_scores)
    score_summary.update(
        {
            "mode": "mvp",
            "query_count": len(queries),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "retrieval_metrics": {
                "old": aggregate_retrieval_metrics(hybrid_results_by_version["old"]),
                "new": aggregate_retrieval_metrics(hybrid_results_by_version["new"]),
            },
            "config_used": config_summary(config),
        }
    )

    write_run_outputs(output, output.name, all_results, citation_rows, answer_rows, score_summary, entity_results, structure_results)
    report_path = generate_report(
        output,
        score_summary,
        query_rows,
        citation_rows,
        entity_results,
        structure_results,
        hybrid_results_by_version,
    )
    return {"output_dir": str(output), "report_path": str(report_path), "score_summary": score_summary}
