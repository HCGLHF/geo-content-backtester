from geo_backtester.evaluation.entity_eval import evaluate_entities


def test_missing_entities_are_detected() -> None:
    result = evaluate_entities(
        "GEO-ALPHA improves Generative Engine Optimization for ChatGPT.",
        {
            "brand_entities": ["GEO-ALPHA", "alphaxxxx.com"],
            "core_topic_entities": ["Generative Engine Optimization", "retrieval ranking analysis"],
            "platform_entities": ["ChatGPT", "Perplexity"],
        },
    )
    assert "alphaxxxx.com" in result["missing_entities"]
    assert "Perplexity" in result["missing_entities"]


def test_covered_entities_are_counted() -> None:
    result = evaluate_entities(
        "GEO-ALPHA connects Generative Engine Optimization with ChatGPT and Perplexity.",
        {
            "brand_entities": ["GEO-ALPHA"],
            "core_topic_entities": ["Generative Engine Optimization"],
            "platform_entities": ["ChatGPT", "Perplexity"],
        },
    )
    assert result["brand_entity_score"] == 100
    assert result["platform_entity_score"] == 100
