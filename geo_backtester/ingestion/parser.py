from __future__ import annotations

import re
from pathlib import Path


FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_front_matter(raw_text: str) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER_RE.match(raw_text)
    if not match:
        return {}, raw_text

    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, raw_text[match.end() :]


def infer_article_id(path: str | Path, metadata: dict[str, str]) -> str:
    if metadata.get("article_id"):
        return metadata["article_id"]
    stem = Path(path).stem
    return re.sub(r"[_-]v?\d+$", "", stem)


def infer_title(text: str, metadata: dict[str, str], fallback: str) -> str:
    if metadata.get("title"):
        return metadata["title"]
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            return line.lstrip("#").strip()
    return fallback
