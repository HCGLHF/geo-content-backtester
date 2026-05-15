from __future__ import annotations

import json
import logging
import os
from statistics import mean

from geo_backtester.evaluation.failure_analysis import expected_terms_missing, unsupported_claim_count
from geo_backtester.models import Query, RetrievalResult


logger = logging.getLogger(__name__)


def openai_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def skip_answer_evaluation(
    queries: list[Query],
    versions: list[str],
    reason: str = "Skipped because OPENAI_API_KEY is not available.",
) -> tuple[list[dict[str, object]], dict[str, None]]:
    rows = [
        {
            "query_id": query.query_id,
            "article_version": version,
            "answer": None,
            "overall_answer_score": None,
            "unsupported_claim_count": None,
            "missing_expected_points": None,
            "explanation": reason,
        }
        for query in queries
        for version in versions
    ]
    return rows, {version: None for version in versions}


def evaluate_answers_with_openai(
    queries: list[Query],
    hybrid_results_by_version: dict[str, dict[str, list[RetrievalResult]]],
) -> tuple[list[dict[str, object]], dict[str, float | None]]:
    if not openai_available():
        return skip_answer_evaluation(queries, list(hybrid_results_by_version))

    try:
        from openai import OpenAI
    except Exception as exc:
        logger.warning("OpenAI package unavailable; skipping answer evaluation: %s", exc)
        return skip_answer_evaluation(
            queries,
            list(hybrid_results_by_version),
            f"Skipped because OpenAI client is unavailable: {exc}",
        )

    client = OpenAI()
    rows: list[dict[str, object]] = []
    scores_by_version: dict[str, list[float]] = {version: [] for version in hybrid_results_by_version}

    for version, by_query in hybrid_results_by_version.items():
        for query in queries:
            chunks = by_query.get(query.query_id, [])[:5]
            context = "\n\n".join(f"[{result.chunk_id}] {result.text}" for result in chunks)
            try:
                answer = _generate_answer(client, query.query, context)
                judge = _judge_answer(client, query, context, answer)
            except Exception as exc:
                logger.warning("OpenAI answer evaluation failed for %s/%s: %s", version, query.query_id, exc)
                rows.append(
                    {
                        "query_id": query.query_id,
                        "article_version": version,
                        "answer": None,
                        "overall_answer_score": None,
                        "unsupported_claim_count": None,
                        "missing_expected_points": None,
                        "explanation": f"OpenAI answer evaluation failed: {exc}",
                    }
                )
                continue
            judge.setdefault("unsupported_claim_count", unsupported_claim_count(answer))
            judge.setdefault("missing_expected_points", expected_terms_missing(query.expected_answer_points, answer))
            score = float(judge.get("overall_answer_score", 0))
            scores_by_version[version].append(score)
            rows.append(
                {
                    "query_id": query.query_id,
                    "article_version": version,
                    "answer": answer,
                    **judge,
                }
            )

    aggregate = {
        version: round(mean(values), 2) if values else None
        for version, values in scores_by_version.items()
    }
    return rows, aggregate


def _generate_answer(client: object, query: str, context: str) -> str:
    response = client.chat.completions.create(
        model=os.getenv("GEO_OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "You are a retrieval-grounded answer generator. Answer only using the provided context. If the answer is not supported by the context, say \"Not found in the provided context.\"",
            },
            {
                "role": "user",
                "content": f"Question:\n{query}\n\nContext:\n{context}\n\nInstructions:\n- Answer in 3 to 6 sentences.\n- Cite the supporting chunk IDs in brackets.\n- Do not use outside knowledge.",
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def _judge_answer(client: object, query: Query, context: str, answer: str) -> dict[str, object]:
    response = client.chat.completions.create(
        model=os.getenv("GEO_OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "user",
                "content": f"""You are evaluating whether an answer is grounded in the retrieved context.

Question:
{query.query}

Expected answer points:
{query.expected_answer_points}

Retrieved context:
{context}

Answer:
{answer}

Score the answer from 0 to 100 on:
1. Faithfulness to context
2. Relevance to question
3. Completeness
4. Citation support
5. No hallucination

Return JSON only:
{{
  "faithfulness": 0,
  "relevance": 0,
  "completeness": 0,
  "citation_support": 0,
  "hallucination_risk": 0,
  "unsupported_claim_count": 0,
  "missing_expected_points": [],
  "overall_answer_score": 0,
  "explanation": "..."
}}""",
            }
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "faithfulness": 0,
            "relevance": 0,
            "completeness": 0,
            "citation_support": 0,
            "hallucination_risk": 100,
            "unsupported_claim_count": unsupported_claim_count(answer),
            "missing_expected_points": expected_terms_missing(query.expected_answer_points, answer),
            "overall_answer_score": 0,
            "explanation": f"Judge returned invalid JSON: {content[:200]}",
        }
