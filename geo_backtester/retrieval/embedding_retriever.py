from __future__ import annotations

import os
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from geo_backtester.config import BacktestConfig
from geo_backtester.models import Chunk, Query, RetrievalResult


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        ...


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self, corpus: list[str]) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.vectorizer.fit(corpus or [""])

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray()


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model_name: str) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return np.asarray([item.embedding for item in response.data])


def build_embedder(chunks: list[Chunk], config: BacktestConfig) -> Embedder:
    corpus = [chunk.text for chunk in chunks]
    if config.use_openai_embeddings and os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIEmbedder(config.openai_embedding_model)
        except Exception as exc:
            print(f"OpenAI embeddings unavailable, falling back locally: {exc}")
    try:
        return SentenceTransformerEmbedder(config.sentence_transformer_model)
    except Exception as exc:
        print(f"Sentence-transformers unavailable, falling back to TF-IDF embeddings: {exc}")
        return TfidfEmbedder(corpus)


class EmbeddingRetriever:
    def __init__(self, chunks: list[Chunk], config: BacktestConfig) -> None:
        self.chunks = chunks
        self.embedder = build_embedder(chunks, config)
        self.chunk_embeddings = self.embedder.encode([chunk.text for chunk in chunks]) if chunks else np.zeros((0, 1))

    def search(self, query: Query, top_k: int = 5) -> list[RetrievalResult]:
        if not self.chunks:
            return []
        query_embedding = self.embedder.encode([query.query])
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        order = np.argsort(similarities)[::-1][:top_k]
        results: list[RetrievalResult] = []
        for rank, idx in enumerate(order, start=1):
            chunk = self.chunks[int(idx)]
            results.append(
                RetrievalResult(
                    query_id=query.query_id,
                    query=query.query,
                    retriever=f"embedding:{self.embedder.name}",
                    article_version=chunk.article_version,
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    score=float(similarities[int(idx)]),
                    text=chunk.text,
                    article_id=chunk.article_id,
                    source_type=chunk.source_type,
                )
            )
        return results
