from __future__ import annotations

import sys
from types import SimpleNamespace

from geo_backtester.evaluation.answer_eval import evaluate_answers_with_openai
from geo_backtester.models import Query


def test_openai_answer_failure_does_not_abort(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class BrokenCompletions:
        def create(self, **kwargs):
            raise RuntimeError("provider unavailable")

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=BrokenCompletions()))
    fake_openai = SimpleNamespace(OpenAI=lambda: fake_client)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    query = Query(
        query_id="q1",
        query="What is GEO?",
        intent="definition",
        target_article="geo_intro",
        expected_answer_points="retrieval citation answer generation",
        priority="high",
    )
    rows, scores = evaluate_answers_with_openai([query], {"new": {"q1": []}})

    assert scores["new"] is None
    assert rows[0]["answer"] is None
    assert "OpenAI answer evaluation failed" in rows[0]["explanation"]
