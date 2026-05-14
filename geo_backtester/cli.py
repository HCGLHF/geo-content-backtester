from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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
    aggregate_retrieval_metrics,
    aggregate_retrieval_metrics_graded,
    annotate_results,
    query_comparison,
    retrieval_score,
    retrieval_score_graded,
)
from geo_backtester.evaluation.structure_eval import evaluate_structure
from geo_backtester.ingestion.loader import (
    load_article,
    load_background_articles,
    load_queries,
    load_relevance_labels,
)
from geo_backtester.models import Article, RetrievalResult
from geo_backtester.reporting.report_generator import generate_report
from geo_backtester.retrieval.bm25_retriever import BM25Retriever
from geo_backtester.retrieval.candidate_pipeline import run_realistic_retrieval, version_metric_view
from geo_backtester.retrieval.embedding_retriever import EmbeddingRetriever
from geo_backtester.retrieval.hybrid_retriever import HybridRetriever


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
        return _run_realistic_backtest(
            old_path,
            new_path,
            queries_path,
            entities_path,
            output_dir,
            labels_path,
            background_corpus,
            core_terms_path,
        )
    return _run_mvp_backtest(old_path, new_path, queries_path, entities_path, output_dir)


def _run_mvp_backtest(
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
            "config_used": _config_summary(config),
        }
    )

    _write_outputs(output, output.name, all_results, citation_rows, answer_rows, score_summary, entity_results, structure_results)
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


def _run_realistic_backtest(
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
    old_core_term_score = _core_term_score(core_term_summary, "old")
    new_core_term_score = _core_term_score(core_term_summary, "new")

    entity_results = {
        "old": evaluate_entities(old_article.text, entity_list),
        "new": evaluate_entities(new_article.text, entity_list),
    }
    structure_results = {
        "old": evaluate_structure(old_article.text),
        "new": evaluate_structure(new_article.text),
    }

    answer_contexts = _answer_contexts_by_version(reranked_by_query)
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
        "core_term_score": old_core_term_score,
        "entity_score": entity_results["old"]["entity_score"],
        "structure_score": structure_results["old"]["structure_score"],
    }
    old_scores["total_geo_score"] = calculate_total_score(**old_scores, mode="realistic")
    new_scores = {
        "retrieval_score": new_retrieval_score,
        "citation_score": new_citation_score,
        "answer_score": answer_scores.get("new"),
        "core_term_score": new_core_term_score,
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
            "failure_type_counts": _failure_type_counts(failure_rows),
            "core_term_enabled": core_term_config.has_terms,
            "core_term_summary": core_term_summary,
            "stuffing_risk_score": {
                "old": _stuffing_risk(core_term_summary, "old"),
                "new": _stuffing_risk(core_term_summary, "new"),
            },
            "config_used": {
                **_config_summary(config),
                "candidate_depth": 20,
                "rerank_depth": 10,
                "final_depth": 5,
            },
        }
    )

    _write_outputs(
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
            "old": _display_by_version(reranked_by_query, "old"),
            "new": _display_by_version(reranked_by_query, "new"),
        },
        failure_rows,
        core_term_summary,
    )
    return {"output_dir": str(output), "report_path": str(report_path), "score_summary": score_summary}


def _answer_contexts_by_version(
    reranked_by_query: dict[str, list[RetrievalResult]],
) -> dict[str, dict[str, list[RetrievalResult]]]:
    contexts = {"old": {}, "new": {}}
    for query_id, results in reranked_by_query.items():
        for version in ["old", "new"]:
            contexts[version][query_id] = [
                result
                for result in results
                if result.article_version == version or result.source_type == "background"
            ][:5]
    return contexts


def _display_by_version(
    reranked_by_query: dict[str, list[RetrievalResult]],
    version: str,
) -> dict[str, list[RetrievalResult]]:
    return {
        query_id: [result for result in results if result.article_version == version]
        for query_id, results in reranked_by_query.items()
    }


def _write_outputs(
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


def _core_term_score(core_term_summary: dict[str, object], version: str) -> float | None:
    if not core_term_summary:
        return None
    return float(core_term_summary.get(version, {}).get("core_term_score", 0.0))


def _stuffing_risk(core_term_summary: dict[str, object], version: str) -> float | None:
    if not core_term_summary:
        return None
    return float(core_term_summary.get(version, {}).get("stuffing_risk_score", 0.0))


def _config_summary(config: object) -> dict[str, object]:
    return {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "hybrid_alpha": config.hybrid_alpha,
        "use_openai_embeddings": config.use_openai_embeddings,
        "use_openai_reranker": config.use_openai_reranker,
        "answer_eval_enabled": config.has_openai_api_key,
    }


def _failure_type_counts(failure_rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in failure_rows:
        failure_type = str(row["failure_type"])
        counts[failure_type] = counts.get(failure_type, 0) + 1
    return counts


def inspect_chunks(article_path: str) -> None:
    config = config_from_env()
    article = load_article(article_path, "inspect")
    chunks = chunk_article(article, config.chunk_size, config.chunk_overlap)
    for chunk in chunks:
        print(f"\n{chunk.chunk_id} | tokens={chunk.token_count} | section={' > '.join(chunk.heading_path)}")
        print("-" * 80)
        print(chunk.text[:1000])


def init_demo(base_dir: str = ".") -> None:
    base = Path(base_dir)
    for directory in [
        "data/articles/old",
        "data/articles/new",
        "data/queries",
        "data/entities",
        "data/labels",
        "data/terms",
        "data/corpus/background",
    ]:
        (base / directory).mkdir(parents=True, exist_ok=True)
    _write_demo_files(base)
    print(f"Demo files created under {base.resolve() / 'data'}")


def _write_demo_files(base: Path) -> None:
    old_md = """---
article_id: geo_intro
version: v1
url: /learn/what-is-geo
title: What is Generative Engine Optimization?
brand: GEO-ALPHA
date: 2026-05-14
---

# What is GEO?

GEO is a future-ready way for brands to grow in the AI era. It helps companies unlock their potential when people ask AI tools questions.

## GEO and SEO

SEO helps websites show up in search results. GEO optimization is about AI tools, but teams often treat it like another content channel. Businesses should publish helpful articles and keep improving them.

## Measuring AI Visibility

Teams can review whether their brand is mentioned by AI systems. They can also look at rankings and answer quality over time.

## Conclusion

GEO-ALPHA helps brands grow as AI changes discovery.
"""
    new_md = """---
article_id: geo_intro
version: v2
url: /learn/what-is-geo
title: What is Generative Engine Optimization?
brand: GEO-ALPHA
date: 2026-05-14
---

# What is Generative Engine Optimization?

**Definition:** Generative Engine Optimization is the process of improving how AI systems discover, retrieve, cite, and summarize brand content in answers.

GEO-ALPHA helps businesses improve AI search visibility through retrieval analysis, citation readiness, entity optimization, and AI visibility scoring.

## GEO vs SEO

SEO targets traditional search engine ranking pages. GEO targets AI retrieval, citation support, and answer generation in systems such as ChatGPT, Gemini, Perplexity, Claude, and Google AI Overviews.

## How GEO-ALPHA Measures AI Visibility

GEO-ALPHA measures AI visibility with four signals:

1. Retrieval ranking analysis for priority user queries.
2. Citation readiness of chunks that contain definitions, comparisons, examples, and measurable claims.
3. Answer grounding, which checks whether retrieved context supports an accurate LLM answer.
4. Entity coverage across brand entities, platform entities, and core topic entities.

## How Businesses Improve Citation Readiness

Businesses can improve citation readiness by adding direct definitions, structured claims, schema markup, FAQPage content, and consistent entity naming. A strong GEO page explains the topic, names the relevant platforms, and provides concise sections that can stand alone as retrieved context.

## FAQ

### How is GEO different from SEO?

GEO is different from SEO because SEO optimizes pages for search engines, while GEO optimizes content for AI retrieval, citation, and grounded answer generation.

### Which platforms matter for GEO?

GEO-ALPHA focuses on ChatGPT, Claude, Gemini, Perplexity, and Google AI Overviews.

## Summary

GEO turns content into retrieval-ready evidence for AI answers. The best pages connect GEO-ALPHA with AI search visibility, retrieval ranking analysis, citation readiness, and entity optimization.
"""
    queries_csv = """query_id,query,intent,target_article,expected_answer_points,priority
q001,What is Generative Engine Optimization?,definition,geo_intro,"GEO improves how AI systems discover retrieve cite and summarize brand content",high
q002,How is GEO different from SEO?,comparison,geo_intro,"SEO targets search engines while GEO targets AI retrieval citation and answer generation",high
q003,How can businesses improve AI citation readiness?,how_to,geo_intro,"clear definitions schema structured claims entity consistency",high
q004,Which platforms does GEO-ALPHA focus on?,platform,geo_intro,"ChatGPT Gemini Perplexity Google AI Overviews",medium
q005,How does GEO-ALPHA measure AI visibility?,measurement,geo_intro,"retrieval ranking citation tracking answer grounding entity coverage",high
"""
    labels_csv = """query_id,article_version,chunk_id_or_heading,relevance_grade,answer_support_grade,citation_worthy,notes
q001,old,old_chunk_001,1,1,false,Old version is too vague for definition.
q001,new,Definition,3,3,true,New definition directly answers the query.
q002,old,GEO and SEO,1,1,false,Old comparison is shallow.
q002,new,GEO vs SEO,3,3,true,New comparison names retrieval citation and answer generation.
q003,old,old_chunk_001,0,0,false,Old article lacks citation readiness details.
q003,new,citation readiness,3,3,true,New article explains definitions schema and entities.
q004,old,old_chunk_001,0,0,false,Old version does not name platforms.
q004,new,Which platforms matter,3,3,true,New FAQ lists target platforms.
q005,old,Measuring AI Visibility,1,1,false,Old measurement is generic.
q005,new,How GEO-ALPHA Measures AI Visibility,3,3,true,New measurement section lists concrete signals.
"""
    entities_json = {
        "brand_entities": ["GEO-ALPHA", "alphaxxxx.com"],
        "core_topic_entities": [
            "Generative Engine Optimization",
            "GEO",
            "AI search",
            "LLM retrieval",
            "citation readiness",
            "AI visibility scoring",
            "retrieval ranking analysis",
            "structured data",
            "schema markup",
        ],
        "platform_entities": ["ChatGPT", "Claude", "Gemini", "Perplexity", "Google AI Overviews"],
    }
    core_terms_json = {
        "must_have_terms": [
            {
                "term": "Generative Engine Optimization",
                "aliases": ["GEO"],
                "weight": 10,
                "type": "core_topic",
                "must_explain": True,
                "preferred_sections": ["h1", "definition", "summary"],
            },
            {
                "term": "AI search visibility",
                "weight": 9,
                "type": "capability",
                "must_explain": True,
                "preferred_sections": ["definition", "summary"],
            },
            {
                "term": "citation readiness",
                "weight": 9,
                "type": "capability",
                "must_explain": True,
                "preferred_sections": ["h2", "faq"],
            },
            {
                "term": "retrieval ranking analysis",
                "weight": 8,
                "type": "capability",
                "must_explain": True,
                "preferred_sections": ["h2", "summary"],
            },
        ],
        "should_have_terms": [
            {"term": "entity optimization", "weight": 7, "type": "capability", "must_explain": True},
            {"term": "AI visibility scoring", "weight": 7, "type": "capability", "must_explain": True},
            {"term": "ChatGPT", "weight": 5, "type": "platform", "must_explain": False},
            {"term": "Gemini", "weight": 5, "type": "platform", "must_explain": False},
            {"term": "Perplexity", "weight": 5, "type": "platform", "must_explain": False},
            {
                "term": "Google AI Overviews",
                "aliases": ["AI Overviews", "Google AIO"],
                "weight": 5,
                "type": "platform",
                "must_explain": False,
            },
        ],
        "stuffing_limits": {
            "max_repeated_credit_per_term": 2,
            "penalty_threshold": 4,
        },
    }
    background_faq = """---
article_id: ai_visibility_faq
title: AI Visibility FAQ
---

# AI Visibility FAQ

## What is AI visibility?

AI visibility measures whether a brand appears in AI-generated answers across systems such as ChatGPT, Gemini, Perplexity, Claude, and Google AI Overviews.

## How should teams track AI mentions?

Teams can track brand mentions, citation frequency, answer accuracy, and query-level retrieval rank across a repeatable query set.
"""
    background_blog = """---
article_id: seo_basics
title: SEO Basics for Content Teams
---

# SEO Basics for Content Teams

SEO improves how pages rank in traditional search engines. It uses keyword research, internal links, technical performance, and structured data to help search crawlers understand pages.
"""
    (base / "data/articles/old/old_article.md").write_text(old_md, encoding="utf-8")
    (base / "data/articles/new/new_article.md").write_text(new_md, encoding="utf-8")
    (base / "data/queries/queries.csv").write_text(queries_csv, encoding="utf-8")
    (base / "data/entities/entity_list.json").write_text(json.dumps(entities_json, indent=2), encoding="utf-8")
    (base / "data/labels/relevance_labels.csv").write_text(labels_csv, encoding="utf-8")
    (base / "data/terms/core_terms.json").write_text(json.dumps(core_terms_json, indent=2), encoding="utf-8")
    (base / "data/corpus/background/ai_visibility_faq.md").write_text(background_faq, encoding="utf-8")
    (base / "data/corpus/background/seo_basics.md").write_text(background_blog, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geo-backtester", description="GEO Content Backtester")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run old vs new article backtest")
    run_parser.add_argument("--old", required=True)
    run_parser.add_argument("--new", required=True)
    run_parser.add_argument("--queries", required=True)
    run_parser.add_argument("--entities")
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--mode", choices=["mvp", "realistic"], default="mvp")
    run_parser.add_argument("--labels")
    run_parser.add_argument("--background-corpus")
    run_parser.add_argument("--core-terms")

    subparsers.add_parser("init-demo", help="Create demo article, query, entity, label, and background files")

    inspect_parser = subparsers.add_parser("inspect-chunks", help="Print chunks for one article")
    inspect_parser.add_argument("--article", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        result = run_backtest(
            args.old,
            args.new,
            args.queries,
            args.entities,
            args.output,
            mode=args.mode,
            labels_path=args.labels,
            background_corpus=args.background_corpus,
            core_terms_path=args.core_terms,
        )
        summary = result["score_summary"]
        print(f"Report written to: {result['report_path']}")
        print(
            f"Mode: {summary.get('mode', 'mvp')} | "
            f"Old score: {summary['old']['total_geo_score']} | "
            f"New score: {summary['new']['total_geo_score']} | "
            f"Winner: {summary['improvement']['winner']}"
        )
    elif args.command == "init-demo":
        init_demo()
    elif args.command == "inspect-chunks":
        inspect_chunks(args.article)


if __name__ == "__main__":
    main()
