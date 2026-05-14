from __future__ import annotations

from dataclasses import dataclass

from geo_backtester.models import Query, RelevanceLabel, RetrievalResult
from geo_backtester.retrieval.tokenize import simple_tokenize


@dataclass(frozen=True)
class ResolvedRelevance:
    is_relevant: bool
    relevance_grade: int
    answer_support_grade: int | None
    citation_worthy: bool | None
    label_source: str


class RelevanceResolver:
    def __init__(self, labels: list[RelevanceLabel] | None = None, allow_article_id_fallback: bool = True) -> None:
        self.labels = labels or []
        self.allow_article_id_fallback = allow_article_id_fallback
        self._exact: dict[tuple[str, str, str], RelevanceLabel] = {}
        self._by_query: dict[str, list[RelevanceLabel]] = {}
        for label in self.labels:
            key = (
                label.query_id,
                label.article_version.lower(),
                label.chunk_id_or_heading.lower(),
            )
            self._exact[key] = label
            self._by_query.setdefault(label.query_id, []).append(label)

    @property
    def label_count(self) -> int:
        return len(self.labels)

    def resolve(self, result: RetrievalResult, query: Query) -> ResolvedRelevance:
        label = self._find_label(result, query.query_id)
        if label:
            return ResolvedRelevance(
                is_relevant=label.relevance_grade >= 2,
                relevance_grade=label.relevance_grade,
                answer_support_grade=label.answer_support_grade,
                citation_worthy=label.citation_worthy,
                label_source="manual",
            )

        grade = self._fallback_grade(result, query)
        return ResolvedRelevance(
            is_relevant=grade >= 2,
            relevance_grade=grade,
            answer_support_grade=None,
            citation_worthy=None,
            label_source="heuristic",
        )

    def label_coverage(self, queries: list[Query]) -> dict[str, object]:
        query_ids = {query.query_id for query in queries}
        labeled_queries = {label.query_id for label in self.labels if label.query_id in query_ids}
        return {
            "label_count": len(self.labels),
            "labeled_query_count": len(labeled_queries),
            "query_count": len(query_ids),
            "coverage_percent": round(100 * len(labeled_queries) / len(query_ids), 2) if query_ids else 0.0,
        }

    def _find_label(self, result: RetrievalResult, query_id: str) -> RelevanceLabel | None:
        version_keys = {result.article_version.lower(), result.source_type.lower()}
        for version in version_keys:
            exact = self._exact.get((query_id, version, result.chunk_id.lower()))
            if exact:
                return exact

        for label in self._by_query.get(query_id, []):
            label_version = label.article_version.lower()
            if label_version not in version_keys and label_version not in {"*", "any", "all"}:
                continue
            needle = label.chunk_id_or_heading.strip().lower()
            if not needle:
                continue
            if needle == result.chunk_id.lower() or needle in result.text.lower():
                return label
        return None

    def _fallback_grade(self, result: RetrievalResult, query: Query) -> int:
        overlap = keyword_overlap(query.expected_answer_points, result.text)
        if self.allow_article_id_fallback and result.article_id == query.target_article:
            return 2 if overlap < 0.45 else 3
        if overlap >= 0.45:
            return 3
        if overlap >= 0.22:
            return 2
        if overlap >= 0.12:
            return 1
        return 0


def keyword_overlap(expected_answer_points: str, chunk_text: str) -> float:
    expected = {token for token in simple_tokenize(expected_answer_points) if len(token) > 2}
    if not expected:
        return 0.0
    chunk_tokens = set(simple_tokenize(chunk_text))
    return len(expected & chunk_tokens) / len(expected)


def apply_relevance(result: RetrievalResult, query: Query, resolver: RelevanceResolver) -> RetrievalResult:
    resolved = resolver.resolve(result, query)
    result.is_relevant = resolved.is_relevant
    result.relevance_grade = resolved.relevance_grade
    result.answer_support_grade = resolved.answer_support_grade
    result.citation_worthy = resolved.citation_worthy
    result.label_source = resolved.label_source
    return result
