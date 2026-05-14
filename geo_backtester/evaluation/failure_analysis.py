from __future__ import annotations

from geo_backtester.models import Query, RetrievalResult
from geo_backtester.retrieval.tokenize import simple_tokenize


def build_failure_analysis(
    queries: list[Query],
    candidates_by_query: dict[str, list[RetrievalResult]],
    reranked_by_query: dict[str, list[RetrievalResult]],
    citation_rows: list[dict[str, object]],
    answer_rows: list[dict[str, object]] | None = None,
    versions: list[str] | None = None,
) -> list[dict[str, object]]:
    versions = versions or ["old", "new"]
    citation_by_chunk = {str(row["chunk_id"]): float(row["citation_score"]) for row in citation_rows}
    answer_by_key = {
        (str(row.get("query_id")), str(row.get("article_version"))): row
        for row in answer_rows or []
    }

    rows: list[dict[str, object]] = []
    for query in queries:
        candidates = candidates_by_query.get(query.query_id, [])
        reranked = reranked_by_query.get(query.query_id, [])
        final = reranked[:5]
        for version in versions:
            version_candidates = [row for row in candidates if row.article_version == version and row.is_relevant]
            version_reranked = [row for row in reranked if row.article_version == version and row.is_relevant]
            version_final = [row for row in final if row.article_version == version and row.is_relevant]
            failure_type = "ok"
            reason = "Relevant evidence reached the final context."
            action = "Keep monitoring this query and preserve the winning evidence structure."

            if not version_candidates:
                failure_type = "recall_failure"
                reason = "No relevant chunk from this article version reached the candidate set."
                action = "Add direct answer language, exact entities, and expected answer terms for this query."
            elif not version_reranked:
                failure_type = "rerank_failure"
                reason = "A relevant chunk was recalled but did not survive reranking."
                action = "Strengthen the chunk with clearer query wording, headings, and concise evidence."
            elif not version_final:
                failure_type = "rerank_failure"
                reason = "A relevant chunk was reranked, but it did not reach the final top 5 context."
                action = "Make the best relevant chunk more answer-like so it outranks competing context."
            else:
                best_final = version_final[0]
                citation_score = citation_by_chunk.get(best_final.chunk_id, 0)
                answer = answer_by_key.get((query.query_id, version))
                answer_score = answer.get("overall_answer_score") if answer else None
                if citation_score < 55:
                    failure_type = "citation_failure"
                    reason = f"Relevant final chunk is weak for citation (citation_score={citation_score:g})."
                    action = "Rewrite the chunk as a citable claim with a clear subject, entities, and direct definition."
                elif answer_score is not None and float(answer_score) < 60:
                    failure_type = "answer_failure"
                    reason = f"Retrieved context reached final top 5, but answer score is low ({answer_score})."
                    action = "Add missing expected answer points and remove ambiguity that causes unsupported claims."

            rows.append(
                {
                    "query_id": query.query_id,
                    "query": query.query,
                    "article_version": version,
                    "failure_type": failure_type,
                    "reason": reason,
                    "best_candidate_rank": _best_rank(version_candidates, "candidate_rank"),
                    "best_rerank_rank": _best_rank(version_reranked, "rerank_rank"),
                    "best_final_rank": _best_rank(version_final, "final_rank"),
                    "recommended_action": action,
                }
            )
    return rows


def expected_terms_missing(expected_answer_points: str, answer: str) -> list[str]:
    answer_tokens = set(simple_tokenize(answer))
    terms = [token for token in simple_tokenize(expected_answer_points) if len(token) > 2]
    seen: set[str] = set()
    missing: list[str] = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        if term not in answer_tokens:
            missing.append(term)
    return missing


def unsupported_claim_count(answer: str) -> int:
    if not answer:
        return 0
    sentence_like = [part.strip() for part in answer.replace("\n", " ").split(".") if part.strip()]
    return sum(1 for sentence in sentence_like if "[" not in sentence or "]" not in sentence)


def _best_rank(results: list[RetrievalResult], attr: str) -> int | None:
    ranks = [getattr(result, attr) for result in results if getattr(result, attr) is not None]
    return min(ranks) if ranks else None
