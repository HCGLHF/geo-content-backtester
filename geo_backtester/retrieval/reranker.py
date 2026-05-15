from __future__ import annotations

import json
import logging
import os
from dataclasses import replace

from geo_backtester.models import Query, RetrievalResult
from geo_backtester.retrieval.tokenize import simple_tokenize


logger = logging.getLogger(__name__)


class LocalReranker:
    def __init__(
        self,
        use_cross_encoder: bool = True,
        use_openai: bool = False,
        openai_model: str = "gpt-4o-mini",
        use_core_terms: bool = False,
    ) -> None:
        self.cross_encoder = None
        self.openai_client = None
        self.openai_model = openai_model
        self.use_core_terms = use_core_terms
        if use_openai and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI

                self.openai_client = OpenAI()
                return
            except Exception as exc:
                logger.warning("OpenAI reranker unavailable, falling back locally: %s", exc)
        if not use_cross_encoder:
            return
        try:
            from sentence_transformers import CrossEncoder

            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as exc:
            logger.warning("Cross-encoder reranker unavailable, falling back to local heuristic rerank: %s", exc)

    def rerank(self, query: Query, candidates: list[RetrievalResult], top_k: int = 10) -> list[RetrievalResult]:
        if not candidates:
            return []
        if self.openai_client is not None:
            scored = self._openai_score(query, candidates)
        elif self.cross_encoder is not None:
            scores = self.cross_encoder.predict([(query.query, candidate.text) for candidate in candidates])
            scored = [
                (candidate, float(score) + candidate.score * 0.15)
                for candidate, score in zip(candidates, scores, strict=False)
            ]
        else:
            scored = [(candidate, self._heuristic_score(query, candidate)) for candidate in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)

        reranked: list[RetrievalResult] = []
        for rank, (candidate, score) in enumerate(scored[:top_k], start=1):
            result = replace(candidate)
            result.retriever = "realistic_rerank"
            result.score = float(score)
            result.rank = rank
            result.rerank_rank = rank
            reranked.append(result)
        return reranked

    def _heuristic_score(self, query: Query, candidate: RetrievalResult) -> float:
        query_tokens = {token for token in simple_tokenize(query.query) if len(token) > 2}
        chunk_tokens = set(simple_tokenize(candidate.text))
        query_overlap = len(query_tokens & chunk_tokens) / len(query_tokens) if query_tokens else 0.0
        if not self.use_core_terms:
            return candidate.score * 0.75 + query_overlap * 0.25
        return (
            candidate.score * 0.62
            + query_overlap * 0.15
            + candidate.core_term_context_score * 0.15
            + self._citation_quality_score(candidate.text) * 0.08
        )

    def _openai_score(
        self,
        query: Query,
        candidates: list[RetrievalResult],
    ) -> list[tuple[RetrievalResult, float]]:
        try:
            candidate_text = "\n\n".join(
                f"{idx}. chunk_id={candidate.chunk_id}\n{candidate.text[:900]}"
                for idx, candidate in enumerate(candidates, start=1)
            )
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Rerank these candidate chunks for a retrieval-grounded RAG answer. "
                            "Return JSON only as an array of chunk_id strings from best to worst.\n\n"
                            f"Question:\n{query.query}\n\nCandidates:\n{candidate_text}"
                        ),
                    }
                ],
                temperature=0,
            )
            ordered_ids = json.loads(response.choices[0].message.content or "[]")
            order = {str(chunk_id): idx for idx, chunk_id in enumerate(ordered_ids)}
            max_score = len(candidates)
            return sorted(
                [
                    (
                        candidate,
                        float(max_score - order.get(candidate.chunk_id, max_score)) + candidate.score * 0.01,
                    )
                    for candidate in candidates
                ],
                key=lambda item: item[1],
                reverse=True,
            )
        except Exception as exc:
            logger.warning("OpenAI rerank failed, falling back to local heuristic: %s", exc)
            return [(candidate, self._heuristic_score(query, candidate)) for candidate in candidates]

    def _citation_quality_score(self, text: str) -> float:
        lowered = text.lower()
        score = 0.25
        if any(signal in lowered for signal in ["is the process of", "refers to", "means", "is defined as", "definition"]):
            score += 0.25
        if any(signal in lowered for signal in ["for example", "compared", "steps", "because", "measures", "signals"]):
            score += 0.20
        if any(signal in lowered for signal in ["geo-alpha", "chatgpt", "gemini", "perplexity", "google ai overviews"]):
            score += 0.20
        if any(signal in lowered for signal in ["unlock your potential", "future-ready", "revolutionary", "new era"]):
            score -= 0.25
        return max(0.0, min(1.0, score))
