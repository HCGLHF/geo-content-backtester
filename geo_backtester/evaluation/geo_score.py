from __future__ import annotations


def calculate_total_score(
    retrieval_score: float,
    citation_score: float,
    entity_score: float,
    structure_score: float,
    answer_score: float | None = None,
    mode: str = "mvp",
    core_term_score: float | None = None,
) -> float:
    if mode == "realistic" and core_term_score is not None and answer_score is not None:
        total = (
            retrieval_score * 0.38
            + answer_score * 0.28
            + citation_score * 0.12
            + core_term_score * 0.12
            + entity_score * 0.05
            + structure_score * 0.05
        )
    elif mode == "realistic" and core_term_score is not None:
        total = (
            retrieval_score * 0.48
            + citation_score * 0.16
            + core_term_score * 0.14
            + entity_score * 0.10
            + structure_score * 0.12
        )
    elif mode == "realistic" and answer_score is not None:
        total = (
            retrieval_score * 0.45
            + answer_score * 0.30
            + citation_score * 0.15
            + entity_score * 0.05
            + structure_score * 0.05
        )
    elif mode == "realistic":
        total = (
            retrieval_score * 0.55
            + citation_score * 0.20
            + entity_score * 0.125
            + structure_score * 0.125
        )
    elif answer_score is None:
        total = (
            retrieval_score * 0.45
            + citation_score * 0.30
            + entity_score * 0.125
            + structure_score * 0.125
        )
    else:
        total = (
            retrieval_score * 0.35
            + citation_score * 0.25
            + answer_score * 0.20
            + entity_score * 0.10
            + structure_score * 0.10
        )
    return round(total, 2)


def build_score_summary(old_scores: dict[str, float | None], new_scores: dict[str, float | None]) -> dict[str, object]:
    old_total = float(old_scores["total_geo_score"] or 0)
    new_total = float(new_scores["total_geo_score"] or 0)
    delta = round(new_total - old_total, 2)
    relative = round((delta / old_total) * 100, 2) if old_total else None
    return {
        "old": old_scores,
        "new": new_scores,
        "improvement": {
            "absolute_delta": delta,
            "relative_delta_percent": relative,
            "winner": "new" if delta > 0 else "old" if delta < 0 else "tie",
        },
    }
