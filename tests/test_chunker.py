from geo_backtester.chunking.chunker import chunk_article
from geo_backtester.models import Article


def test_chunker_returns_non_empty_chunks() -> None:
    article = Article("geo_intro", "new", "What is GEO?", "# What is GEO?\n\nGEO is about AI retrieval.")
    chunks = chunk_article(article, chunk_size=50, chunk_overlap=5)
    assert chunks


def test_chunk_size_approximately_respects_target() -> None:
    text = "# Title\n\n" + "\n\n".join(f"## Section {i}\n\n" + "word " * 80 for i in range(6))
    article = Article("geo_intro", "new", "Title", text)
    chunks = chunk_article(article, chunk_size=120, chunk_overlap=10)
    assert len(chunks) > 1
    assert max(chunk.token_count for chunk in chunks) <= 180


def test_heading_context_is_preserved() -> None:
    article = Article("geo_intro", "new", "Title", "# Title\n\n## GEO vs SEO\n\nGEO targets AI retrieval.")
    chunks = chunk_article(article, chunk_size=80, chunk_overlap=0)
    assert any("GEO vs SEO" in chunk.heading_path for chunk in chunks)
    assert any("Section:" in chunk.text and "GEO vs SEO" in chunk.text for chunk in chunks)
