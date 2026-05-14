from __future__ import annotations

import re
from statistics import mean

from geo_backtester.models import Chunk


DEFINITION_SIGNALS = [
    "is the process of",
    "refers to",
    "means",
    "is defined as",
    " is a ",
    " are ",
]
VAGUE_PHRASES = [
    "unlock your potential",
    "future-ready",
    "transform your business",
    "revolutionary",
    "seamless experience",
    "new era",
]
ANSWER_STRUCTURE_SIGNALS = [r"\n\s*\d+\.", r"\n\s*- ", "for example", "compared with", "steps", "because"]
WEAK_REFERENCE_RE = re.compile(r"\b(this|it|they|these|those)\b", re.I)


def evaluate_chunk_citation(chunk: Chunk, entities: list[str] | None = None) -> dict[str, object]:
    text = chunk.text
    lowered = text.lower()
    score = 35
    reasons: list[str] = []

    if any(signal in lowered for signal in DEFINITION_SIGNALS):
        score += 20
        reasons.append("Contains clear definition sentence")

    vague_found = [phrase for phrase in VAGUE_PHRASES if phrase in lowered]
    if vague_found:
        score -= min(20, len(vague_found) * 8)
        reasons.append(f"Vague marketing phrases detected: {', '.join(vague_found)}")
    else:
        score += 10
        reasons.append("Low marketing vagueness")

    entity_hits = [entity for entity in entities or [] if entity.lower() in lowered]
    if entity_hits:
        score += min(20, 5 + len(entity_hits) * 3)
        reasons.append("Mentions core GEO entities")

    if any(re.search(signal, text, re.I) for signal in ANSWER_STRUCTURE_SIGNALS):
        score += 12
        reasons.append("Uses answer-like structure")

    weak_refs = len(WEAK_REFERENCE_RE.findall(text[:300]))
    if weak_refs >= 5:
        score -= 12
        reasons.append("Opening relies on unclear pronouns")
    else:
        score += 5
        reasons.append("Chunk can mostly stand alone")

    score = max(0, min(100, score))
    strength = "strong" if score >= 75 else "moderate" if score >= 55 else "weak"
    return {
        "article_version": chunk.article_version,
        "chunk_id": chunk.chunk_id,
        "citation_score": score,
        "citation_strength": strength,
        "reasons": reasons,
        "text_preview": re.sub(r"\s+", " ", text)[:260],
    }


def evaluate_citations(chunks: list[Chunk], entities: list[str] | None = None) -> tuple[list[dict[str, object]], float]:
    rows = [evaluate_chunk_citation(chunk, entities) for chunk in chunks]
    score = round(mean([float(row["citation_score"]) for row in rows]), 2) if rows else 0.0
    return rows, score
