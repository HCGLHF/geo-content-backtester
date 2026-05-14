from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Article:
    article_id: str
    version: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""


@dataclass
class Chunk:
    chunk_id: str
    article_version: str
    article_id: str
    title: str
    heading_path: list[str]
    text: str
    token_count: int
    start_index: int
    end_index: int
    source_type: str = "article"

    def retrieval_text(self) -> str:
        section = " > ".join(self.heading_path) if self.heading_path else self.title
        return f"Title: {self.title}\nSection: {section}\nContent: {self.text}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Query:
    query_id: str
    query: str
    intent: str
    target_article: str
    expected_answer_points: str
    priority: str = "medium"


@dataclass(frozen=True)
class RelevanceLabel:
    query_id: str
    article_version: str
    chunk_id_or_heading: str
    relevance_grade: int
    answer_support_grade: int
    citation_worthy: bool
    notes: str = ""


@dataclass
class RetrievalResult:
    query_id: str
    query: str
    retriever: str
    article_version: str
    rank: int
    chunk_id: str
    score: float
    text: str
    article_id: str
    is_relevant: bool = False
    hit_at_1: int = 0
    hit_at_3: int = 0
    hit_at_5: int = 0
    mrr: float = 0.0
    source_type: str = "article"
    candidate_rank: int | None = None
    rerank_rank: int | None = None
    final_rank: int | None = None
    relevance_grade: int | None = None
    answer_support_grade: int | None = None
    citation_worthy: bool | None = None
    label_source: str = "heuristic"
    core_term_context_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
