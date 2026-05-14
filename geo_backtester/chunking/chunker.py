from __future__ import annotations

import re

from geo_backtester.models import Article, Chunk


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(re.findall(r"\w+|[^\w\s]", text))


def _split_units(text: str) -> list[tuple[str, int, int, list[str]]]:
    units: list[tuple[str, int, int, list[str]]] = []
    heading_stack: list[str] = []
    cursor = 0
    blocks = re.split(r"\n\s*\n", text)

    for block in blocks:
        start = text.find(block, cursor)
        end = start + len(block)
        cursor = end
        stripped = block.strip()
        if not stripped:
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(heading)
            units.append((stripped, start, end, heading_stack.copy()))
            continue

        units.append((stripped, start, end, heading_stack.copy()))
    return units


def chunk_article(
    article: Article,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    source_type: str | None = None,
) -> list[Chunk]:
    units = _split_units(article.text)
    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_start = 0
    current_end = 0
    current_headings: list[str] = []

    def flush() -> None:
        nonlocal current_parts, current_start, current_end, current_headings
        if not current_parts:
            return
        text = "\n\n".join(current_parts).strip()
        if not text:
            return
        chunk_num = len(chunks) + 1
        heading_path = current_headings.copy()
        prefix_lines = [f"Title: {article.title}"]
        if heading_path:
            prefix_lines.append(f"Section: {' > '.join(heading_path)}")
        retrieval_text = "\n".join(prefix_lines + [text])
        chunks.append(
            Chunk(
                chunk_id=f"{article.version}_chunk_{chunk_num:03d}",
                article_version=article.version,
                article_id=article.article_id,
                title=article.title,
                heading_path=heading_path,
                text=retrieval_text,
                token_count=estimate_tokens(retrieval_text),
                start_index=current_start,
                end_index=current_end,
                source_type=source_type or f"{article.version}_article",
            )
        )

        if chunk_overlap > 0:
            words = re.findall(r"\S+", text)
            overlap_words = words[-chunk_overlap:]
            current_parts = [" ".join(overlap_words)] if overlap_words else []
            current_start = max(current_end - len(current_parts[0]), current_start) if current_parts else 0
        else:
            current_parts = []
        current_headings = heading_path

    for unit_text, start, end, headings in units:
        unit_tokens = estimate_tokens(unit_text)
        current_tokens = estimate_tokens("\n\n".join(current_parts)) if current_parts else 0
        if current_parts and current_tokens + unit_tokens > chunk_size:
            flush()

        if not current_parts:
            current_start = start
            current_headings = headings
        elif headings:
            current_headings = headings
        current_parts.append(unit_text)
        current_end = end

        if estimate_tokens("\n\n".join(current_parts)) >= chunk_size:
            flush()

    flush()
    return chunks
