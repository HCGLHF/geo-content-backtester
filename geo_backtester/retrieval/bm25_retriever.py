from __future__ import annotations

import math
from collections import Counter, defaultdict

from geo_backtester.models import Chunk, Query, RetrievalResult
from geo_backtester.retrieval.tokenize import simple_tokenize


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.docs = [simple_tokenize(chunk.text) for chunk in chunks]
        self.doc_freq: dict[str, int] = defaultdict(int)
        for doc in self.docs:
            for token in set(doc):
                self.doc_freq[token] += 1
        self.avgdl = sum(len(doc) for doc in self.docs) / len(self.docs) if self.docs else 0.0

    def _score_doc(self, query_tokens: list[str], doc: list[str]) -> float:
        if not doc or not self.avgdl:
            return 0.0
        counts = Counter(doc)
        score = 0.0
        total_docs = len(self.docs)
        for token in query_tokens:
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            freq = counts[token]
            denom = freq + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
            score += idf * (freq * (self.k1 + 1)) / denom
        return float(score)

    def search(self, query: Query, top_k: int = 5) -> list[RetrievalResult]:
        query_tokens = simple_tokenize(query.query)
        scored = [
            (chunk, self._score_doc(query_tokens, doc))
            for chunk, doc in zip(self.chunks, self.docs, strict=False)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            RetrievalResult(
                query_id=query.query_id,
                query=query.query,
                retriever="bm25",
                article_version=chunk.article_version,
                rank=rank,
                chunk_id=chunk.chunk_id,
                score=score,
                text=chunk.text,
                article_id=chunk.article_id,
                source_type=chunk.source_type,
                title=chunk.title,
                heading_path=chunk.heading_path,
            )
            for rank, (chunk, score) in enumerate(scored[:top_k], start=1)
        ]
