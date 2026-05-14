# GEO Content Backtester

Local MVP for comparing old and new article versions before publication. It simulates a simplified AI search / RAG retrieval pipeline and produces retrieval, citation, entity, structure, answer, and total GEO scores.

## Quick Start

```bash
python -m geo_backtester.cli init-demo
python -m geo_backtester.cli run \
  --old data/articles/old/old_article.md \
  --new data/articles/new/new_article.md \
  --queries data/queries/queries.csv \
  --entities data/entities/entity_list.json \
  --output outputs/runs/demo
```

Outputs:

- `outputs/runs/demo/retrieval_results.csv`
- `outputs/runs/demo/citation_results.csv`
- `outputs/runs/demo/answer_results.csv`
- `outputs/runs/demo/score_summary.json`
- `outputs/runs/demo/entity_results.json`
- `outputs/runs/demo/structure_results.json`
- `outputs/runs/demo/report.html`

## Commands

Inspect article chunks:

```bash
python -m geo_backtester.cli inspect-chunks --article data/articles/new/new_article.md
```

Run tests:

```bash
pytest
```

Run the more realistic RAG-style simulation with manual labels and background distractor corpus:

```bash
python -m geo_backtester.cli run \
  --mode realistic \
  --old data/articles/old/old_article.md \
  --new data/articles/new/new_article.md \
  --queries data/queries/queries.csv \
  --entities data/entities/entity_list.json \
  --labels data/labels/relevance_labels.csv \
  --background-corpus data/corpus/background \
  --core-terms data/terms/core_terms.json \
  --output outputs/runs/realistic_demo
```

Realistic mode adds:

- `relevance_labels.csv` support with manual labels taking priority over heuristic relevance.
- Background corpus distractors so old/new chunks compete against surrounding site content.
- Candidate, rerank, and final rank fields in `retrieval_results.csv`.
- Graded metrics: `NDCG@5`, `Recall@10`, and `Precision@3`.
- `failure_analysis.csv` with `recall_failure`, `rerank_failure`, `citation_failure`, and `answer_failure`.
- `core_term_results.csv` and `core_term_summary.json` when `--core-terms` is provided.
- Core term rerank boost with stuffing-risk diagnostics, so important terms improve exposure only when they appear in useful context.

Optional dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Configuration

The system works without `OPENAI_API_KEY`. If no key is present, answer evaluation is skipped and the answer-score weight is redistributed across retrieval, citation, entity, and structure scores.

Optional environment variables:

- `OPENAI_API_KEY`
- `GEO_USE_OPENAI_EMBEDDINGS=true`
- `GEO_USE_OPENAI_RERANKER=true`
- `GEO_CHUNK_SIZE=500`
- `GEO_CHUNK_OVERLAP=80`
- `GEO_TOP_K=5`
- `GEO_HYBRID_ALPHA=0.45`
- `GEO_OPENAI_CHAT_MODEL=gpt-4o-mini`

The embedding retriever tries OpenAI embeddings only when configured, then `sentence-transformers`, then local TF-IDF cosine similarity as a no-download fallback.

## Public Demo Safety

This repository is designed to be safe as a public demo:

- Do not commit real `.env` files or API keys.
- Use `.env.example` only as a template.
- Generated run artifacts under `outputs/runs/` are ignored by git.
- Demo data is synthetic/sample content for local backtesting workflows.
