from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from geo_backtester.models import Chunk, Query
from geo_backtester.retrieval.tokenize import simple_tokenize


CONTEXT_SIGNALS = [
    "is the process of",
    "refers to",
    "means",
    "is defined as",
    "helps",
    "improves",
    "measures",
    "through",
    "because",
    "includes",
    "with",
    "by",
]
RELATION_TERMS = [
    "GEO-ALPHA",
    "retrieval",
    "citation",
    "visibility",
    "AI search",
    "answer",
    "entity",
    "ranking",
]
HIGH_VALUE_SECTIONS = {"h1", "h2", "h3", "faq", "definition", "summary", "conclusion", "key takeaways"}


@dataclass(frozen=True)
class CoreTerm:
    term: str
    aliases: list[str] = field(default_factory=list)
    weight: float = 1.0
    level: str = "should"
    term_type: str = "core"
    must_explain: bool = True
    preferred_sections: list[str] = field(default_factory=list)

    @property
    def variants(self) -> list[str]:
        return [self.term, *self.aliases]


@dataclass(frozen=True)
class StuffingLimits:
    max_repeated_credit_per_term: int = 2
    penalty_threshold: int = 4


@dataclass(frozen=True)
class CoreTermConfig:
    terms: list[CoreTerm] = field(default_factory=list)
    stuffing_limits: StuffingLimits = field(default_factory=StuffingLimits)

    @property
    def has_terms(self) -> bool:
        return bool(self.terms)


@dataclass(frozen=True)
class TermOccurrence:
    term: str
    variant: str
    context_quality: float
    high_value_position: bool
    entity_relation: bool
    context_preview: str


def load_core_terms(path: str | Path | None) -> CoreTermConfig:
    if not path:
        return CoreTermConfig()
    term_path = Path(path)
    if not term_path.exists():
        raise FileNotFoundError(f"Core terms JSON not found: {term_path}")
    data = json.loads(term_path.read_text(encoding="utf-8"))

    aliases_by_term = {str(key): list(value) for key, value in data.get("aliases", {}).items()}
    terms: list[CoreTerm] = []
    for level, key in [("must", "must_have_terms"), ("should", "should_have_terms"), ("should", "global_terms"), ("should", "platform_terms")]:
        for entry in data.get(key, []):
            terms.append(_parse_term_entry(entry, level, aliases_by_term))
    limits = data.get("stuffing_limits", {})
    return CoreTermConfig(
        terms=terms,
        stuffing_limits=StuffingLimits(
            max_repeated_credit_per_term=int(limits.get("max_repeated_credit_per_term", 2)),
            penalty_threshold=int(limits.get("penalty_threshold", 4)),
        ),
    )


def evaluate_core_terms_by_version(
    chunks_by_version: dict[str, list[Chunk]],
    config: CoreTermConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not config.has_terms:
        return [], {}
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for version, chunks in chunks_by_version.items():
        version_rows = [evaluate_chunk_core_terms(chunk, config) for chunk in chunks]
        rows.extend(version_rows)
        summary[version] = summarize_core_terms(version_rows, chunks, config)
    return rows, summary


def evaluate_chunk_core_terms(chunk: Chunk, config: CoreTermConfig, query: Query | None = None) -> dict[str, object]:
    occurrences = find_term_occurrences(chunk, config, query)
    terms = sorted({occurrence.term for occurrence in occurrences})
    term_counts = {term: sum(1 for occurrence in occurrences if occurrence.term == term) for term in terms}
    weak_terms = sorted(
        {
            occurrence.term
            for occurrence in occurrences
            if occurrence.context_quality < 0.55
        }
    )
    high_value_terms = sorted({occurrence.term for occurrence in occurrences if occurrence.high_value_position})
    context_score = chunk_core_term_context_score(chunk, config, query)
    return {
        "article_version": chunk.article_version,
        "chunk_id": chunk.chunk_id,
        "source_type": chunk.source_type,
        "core_term_context_score": round(context_score * 100, 2),
        "covered_core_terms": terms,
        "term_counts": term_counts,
        "weak_context_terms": weak_terms,
        "high_value_section_terms": high_value_terms,
        "term_occurrence_count": len(occurrences),
        "text_preview": re.sub(r"\s+", " ", chunk.text)[:260],
    }


def summarize_core_terms(rows: list[dict[str, object]], chunks: list[Chunk], config: CoreTermConfig) -> dict[str, object]:
    occurrences_by_term: dict[str, list[dict[str, object]]] = {term.term: [] for term in config.terms}
    chunk_hits_by_term: dict[str, set[str]] = {term.term: set() for term in config.terms}
    for row in rows:
        for term in row["covered_core_terms"]:
            occurrences_by_term[str(term)].append(row)
            chunk_hits_by_term[str(term)].add(str(row["chunk_id"]))

    total_weight = sum(term.weight for term in config.terms) or 1.0
    coverage_points = 0.0
    placement_values: list[float] = []
    context_values: list[float] = []
    relation_values: list[float] = []
    overused_terms: list[str] = []
    weak_context_terms: set[str] = set()

    for term in config.terms:
        term_rows = occurrences_by_term[term.term]
        occurrence_count = sum(int(row.get("term_counts", {}).get(term.term, 0)) for row in term_rows)
        if occurrence_count:
            credit_count = min(occurrence_count, config.stuffing_limits.max_repeated_credit_per_term)
            coverage_points += term.weight * min(1.0, 0.8 + 0.2 * (credit_count - 1))
        if occurrence_count >= config.stuffing_limits.penalty_threshold:
            overused_terms.append(term.term)

        term_context_scores = [
            float(row["core_term_context_score"]) / 100
            for row in term_rows
            if term.term in row["covered_core_terms"]
        ]
        if term_context_scores:
            context_values.append(max(term_context_scores))
            if max(term_context_scores) < 0.55:
                weak_context_terms.add(term.term)
        placement_values.append(1.0 if any(term.term in row["high_value_section_terms"] for row in term_rows) else 0.0)
        relation_values.append(_term_relation_score(term, chunks))

    covered_terms = sorted(term for term, term_rows in occurrences_by_term.items() if term_rows)
    missing_must = sorted(term.term for term in config.terms if term.level == "must" and not occurrences_by_term[term.term])
    coverage_score = 45 * coverage_points / total_weight
    placement_score = 15 * (mean(placement_values) if placement_values else 0)
    context_score = 20 * (mean(context_values) if context_values else 0)
    relation_score = 10 * (mean(relation_values) if relation_values else 0)
    distribution_score = 10 * _distribution_score(chunk_hits_by_term, chunks)
    stuffing_risk = stuffing_risk_score(rows, config)
    total = max(0.0, coverage_score + placement_score + context_score + relation_score + distribution_score - stuffing_risk * 0.25)

    return {
        "core_term_score": round(min(100.0, total), 2),
        "stuffing_risk_score": round(stuffing_risk, 2),
        "covered_core_terms": covered_terms,
        "missing_must_have_terms": missing_must,
        "weak_context_terms": sorted(weak_context_terms),
        "overused_terms": sorted(overused_terms),
        "high_value_section_terms": sorted(
            {
                str(term)
                for row in rows
                for term in row["high_value_section_terms"]
            }
        ),
        "recommended_safe_insertions": _recommended_insertions(missing_must, sorted(weak_context_terms)),
        "score_breakdown": {
            "weighted_term_coverage": round(coverage_score, 2),
            "high_value_placement": round(placement_score, 2),
            "context_quality": round(context_score, 2),
            "entity_relation": round(relation_score, 2),
            "distribution_quality": round(distribution_score, 2),
            "stuffing_penalty": round(stuffing_risk * 0.25, 2),
        },
    }


def chunk_core_term_context_score(chunk: Chunk, config: CoreTermConfig, query: Query | None = None) -> float:
    occurrences = find_term_occurrences(chunk, config, query)
    if not occurrences:
        return 0.0
    weights_by_term = {term.term: term.weight for term in config.terms}
    best_by_term: dict[str, float] = {}
    for occurrence in occurrences:
        quality = occurrence.context_quality
        if occurrence.high_value_position:
            quality += 0.12
        if occurrence.entity_relation:
            quality += 0.10
        best_by_term[occurrence.term] = max(best_by_term.get(occurrence.term, 0.0), min(1.0, quality))
    weighted = sum(best_by_term[term] * weights_by_term.get(term, 1.0) for term in best_by_term)
    total_weight = sum(weights_by_term.get(term, 1.0) for term in best_by_term) or 1.0
    return max(0.0, min(1.0, weighted / total_weight))


def find_term_occurrences(chunk: Chunk, config: CoreTermConfig, query: Query | None = None) -> list[TermOccurrence]:
    occurrences: list[TermOccurrence] = []
    lowered = chunk.text.lower()
    for term in config.terms:
        for variant in term.variants:
            pattern = _term_pattern(variant)
            for match in re.finditer(pattern, lowered, re.I):
                context = _context_window(chunk.text, match.start(), match.end())
                occurrences.append(
                    TermOccurrence(
                        term=term.term,
                        variant=variant,
                        context_quality=_context_quality(context, term, query),
                        high_value_position=_high_value_position(chunk, context, term),
                        entity_relation=_entity_relation(context, term),
                        context_preview=re.sub(r"\s+", " ", context)[:220],
                    )
                )
    return occurrences


def stuffing_risk_score(rows: list[dict[str, object]], config: CoreTermConfig) -> float:
    if not rows:
        return 0.0
    risk = 0.0
    term_counts: dict[str, int] = {term.term: 0 for term in config.terms}
    weak_terms: set[str] = set()
    total_tokens = 0
    for row in rows:
        total_tokens += max(1, len(simple_tokenize(str(row.get("text_preview", "")))))
        for term in row["covered_core_terms"]:
            term_counts[str(term)] = term_counts.get(str(term), 0) + int(row.get("term_counts", {}).get(str(term), 0))
        weak_terms.update(str(term) for term in row["weak_context_terms"])
    for term, count in term_counts.items():
        if count >= config.stuffing_limits.penalty_threshold:
            risk += min(35, (count - config.stuffing_limits.penalty_threshold + 1) * 8)
            if term in weak_terms:
                risk += 12
    return min(100.0, risk)


def _parse_term_entry(entry: str | dict[str, Any], level: str, aliases_by_term: dict[str, list[str]]) -> CoreTerm:
    if isinstance(entry, str):
        return CoreTerm(term=entry, aliases=aliases_by_term.get(entry, []), level=level)
    term = str(entry["term"])
    return CoreTerm(
        term=term,
        aliases=list(entry.get("aliases", aliases_by_term.get(term, []))),
        weight=float(entry.get("weight", 1.0)),
        level=str(entry.get("level", level)),
        term_type=str(entry.get("type", "core")),
        must_explain=bool(entry.get("must_explain", True)),
        preferred_sections=list(entry.get("preferred_sections", [])),
    )


def _term_pattern(term: str) -> str:
    return rf"(?<![\w-]){re.escape(term.lower())}(?![\w-])"


def _context_window(text: str, start: int, end: int, radius: int = 520) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)]


def _context_quality(context: str, term: CoreTerm, query: Query | None = None) -> float:
    lowered = context.lower()
    quality = 0.15
    if any(signal in lowered for signal in CONTEXT_SIGNALS):
        quality += 0.35
    if re.search(r"\b(is|are|helps|improves|measures|uses|targets|supports|enables)\b", lowered):
        quality += 0.20
    if _entity_relation(context, term):
        quality += 0.20
    if query and _expected_overlap(query.expected_answer_points, context) >= 0.2:
        quality += 0.15
    if term.must_explain and quality < 0.45:
        quality *= 0.7
    return min(1.0, quality)


def _expected_overlap(expected_answer_points: str, context: str) -> float:
    expected = {token for token in simple_tokenize(expected_answer_points) if len(token) > 2}
    if not expected:
        return 0.0
    context_tokens = set(simple_tokenize(context))
    return len(expected & context_tokens) / len(expected)


def _high_value_position(chunk: Chunk, context: str, term: CoreTerm) -> bool:
    text_start = chunk.text[:700].lower()
    context_lowered = context.lower()
    sections = {section.lower() for section in chunk.heading_path}
    preferred = {section.lower() for section in term.preferred_sections}
    high_value = HIGH_VALUE_SECTIONS | preferred
    return (
        any(section in " ".join(sections) for section in high_value)
        or "definition" in context_lowered
        or "faq" in context_lowered
        or "summary" in context_lowered
        or any(variant.lower() in text_start for variant in term.variants)
    )


def _entity_relation(context: str, term: CoreTerm) -> bool:
    lowered = context.lower()
    if term.term_type == "platform":
        return any(word.lower() in lowered for word in ["focus", "platform", "system", "answer", "overview"])
    return any(word.lower() in lowered for word in RELATION_TERMS)


def _term_relation_score(term: CoreTerm, chunks: list[Chunk]) -> float:
    for chunk in chunks:
        for variant in term.variants:
            if re.search(_term_pattern(variant), chunk.text, re.I) and _entity_relation(chunk.text, term):
                return 1.0
    return 0.0


def _distribution_score(chunk_hits_by_term: dict[str, set[str]], chunks: list[Chunk]) -> float:
    covered_chunks = set().union(*chunk_hits_by_term.values()) if chunk_hits_by_term else set()
    covered_terms = [term for term, hits in chunk_hits_by_term.items() if hits]
    if not covered_terms:
        return 0.0
    ideal = min(len(covered_terms), max(1, len(chunks)))
    return min(1.0, len(covered_chunks) / ideal)


def _recommended_insertions(missing_must: list[str], weak_context_terms: list[str]) -> list[str]:
    recommendations: list[str] = []
    for term in missing_must[:5]:
        recommendations.append(f"Add a direct definition or FAQ sentence that explains '{term}' in relation to GEO-ALPHA.")
    for term in weak_context_terms[:5]:
        recommendations.append(f"Rewrite the sentence around '{term}' with a clear subject, action, and evidence claim.")
    return recommendations
