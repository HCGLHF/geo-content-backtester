from __future__ import annotations

import json
import re
from pathlib import Path


ENTITY_GROUPS = ["brand_entities", "core_topic_entities", "platform_entities"]
INCONSISTENT_TERMS = ["GEO optimization", "AI SEO", "Generative SEO"]


def load_entity_list(path: str | Path | None) -> dict[str, list[str]]:
    if not path:
        return {group: [] for group in ENTITY_GROUPS}
    entity_path = Path(path)
    if not entity_path.exists():
        raise FileNotFoundError(f"Entity list not found: {entity_path}")
    data = json.loads(entity_path.read_text(encoding="utf-8"))
    return {group: list(data.get(group, [])) for group in ENTITY_GROUPS}


def flatten_entities(entity_list: dict[str, list[str]]) -> list[str]:
    return [entity for group in ENTITY_GROUPS for entity in entity_list.get(group, [])]


def _coverage_score(text: str, entities: list[str]) -> tuple[float, list[str], list[str]]:
    if not entities:
        return 100.0, [], []
    lowered = text.lower()
    covered = [entity for entity in entities if entity.lower() in lowered]
    missing = [entity for entity in entities if entity.lower() not in lowered]
    return round(100 * len(covered) / len(entities), 2), covered, missing


def evaluate_entities(text: str, entity_list: dict[str, list[str]]) -> dict[str, object]:
    brand_score, brand_covered, brand_missing = _coverage_score(text, entity_list.get("brand_entities", []))
    core_score, core_covered, core_missing = _coverage_score(text, entity_list.get("core_topic_entities", []))
    platform_score, platform_covered, platform_missing = _coverage_score(text, entity_list.get("platform_entities", []))
    inconsistent = [term for term in INCONSISTENT_TERMS if re.search(re.escape(term), text, re.I)]

    service_connection = 0
    if re.search(r"GEO-ALPHA.{0,120}(AI search visibility|retrieval analysis|citation readiness|entity optimization|AI visibility)", text, re.I):
        service_connection = 10

    overall = (brand_score * 0.3 + core_score * 0.45 + platform_score * 0.25) - len(inconsistent) * 6 + service_connection
    overall = round(max(0, min(100, overall)), 2)
    return {
        "brand_entity_score": brand_score,
        "core_topic_entity_score": core_score,
        "platform_entity_score": platform_score,
        "entity_score": overall,
        "covered_entities": {
            "brand_entities": brand_covered,
            "core_topic_entities": core_covered,
            "platform_entities": platform_covered,
        },
        "missing_entities": brand_missing + core_missing + platform_missing,
        "inconsistent_terms": inconsistent,
    }
