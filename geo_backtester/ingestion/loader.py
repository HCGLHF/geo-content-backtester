from __future__ import annotations

from pathlib import Path

import pandas as pd

from geo_backtester.ingestion.cleaner import clean_text
from geo_backtester.ingestion.parser import infer_article_id, infer_title, parse_front_matter
from geo_backtester.models import Article, Query, RelevanceLabel


REQUIRED_QUERY_COLUMNS = {
    "query_id",
    "query",
    "intent",
    "target_article",
    "expected_answer_points",
}
REQUIRED_LABEL_COLUMNS = {
    "query_id",
    "article_version",
    "chunk_id_or_heading",
    "relevance_grade",
    "answer_support_grade",
    "citation_worthy",
}


def load_article(path: str | Path, version: str) -> Article:
    article_path = Path(path)
    if not article_path.exists():
        raise FileNotFoundError(f"Article file not found: {article_path}")

    raw_text = article_path.read_text(encoding="utf-8")
    metadata, body = parse_front_matter(raw_text)
    cleaned = clean_text(body)
    article_id = infer_article_id(article_path, metadata)
    title = infer_title(cleaned, metadata, fallback=article_path.stem.replace("_", " ").title())
    return Article(
        article_id=article_id,
        version=version,
        title=title,
        text=cleaned,
        metadata=metadata,
        source_path=str(article_path),
    )


def load_queries(path: str | Path) -> list[Query]:
    query_path = Path(path)
    if not query_path.exists():
        raise FileNotFoundError(f"Query CSV not found: {query_path}")

    df = pd.read_csv(query_path)
    missing = REQUIRED_QUERY_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"queries.csv is missing required columns: {', '.join(sorted(missing))}")

    if "priority" not in df.columns:
        df["priority"] = "medium"

    return [
        Query(
            query_id=str(row.query_id),
            query=str(row.query),
            intent=str(row.intent),
            target_article=str(row.target_article),
            expected_answer_points=str(row.expected_answer_points),
            priority=str(row.priority),
        )
        for row in df.itertuples(index=False)
    ]


def load_relevance_labels(path: str | Path | None) -> list[RelevanceLabel]:
    if not path:
        return []
    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(f"Relevance label CSV not found: {label_path}")

    df = pd.read_csv(label_path)
    missing = REQUIRED_LABEL_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"relevance_labels.csv is missing required columns: {', '.join(sorted(missing))}")
    if "notes" not in df.columns:
        df["notes"] = ""

    labels: list[RelevanceLabel] = []
    for row in df.itertuples(index=False):
        labels.append(
            RelevanceLabel(
                query_id=str(row.query_id),
                article_version=str(row.article_version),
                chunk_id_or_heading=str(row.chunk_id_or_heading),
                relevance_grade=int(row.relevance_grade),
                answer_support_grade=int(row.answer_support_grade),
                citation_worthy=_parse_bool(row.citation_worthy),
                notes=str(row.notes) if not pd.isna(row.notes) else "",
            )
        )
    return labels


def load_background_articles(path: str | Path | None) -> list[Article]:
    if not path:
        return []
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Background corpus path not found: {corpus_path}")

    article_paths = [corpus_path] if corpus_path.is_file() else sorted(
        file
        for file in corpus_path.rglob("*")
        if file.is_file() and file.suffix.lower() in {".md", ".markdown", ".html", ".htm", ".txt"}
    )
    articles: list[Article] = []
    for idx, article_path in enumerate(article_paths, start=1):
        articles.append(load_article(article_path, f"background_{idx:03d}"))
    return articles


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
