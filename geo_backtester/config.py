from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BacktestConfig:
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 5
    hybrid_alpha: float = 0.45
    use_openai_embeddings: bool = False
    use_openai_reranker: bool = False
    openai_embedding_model: str = "text-embedding-3-small"
    openai_rerank_model: str = "gpt-4o-mini"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"

    @property
    def has_openai_api_key(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def config_from_env() -> BacktestConfig:
    load_env_file()
    return BacktestConfig(
        chunk_size=int(os.getenv("GEO_CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("GEO_CHUNK_OVERLAP", "80")),
        top_k=int(os.getenv("GEO_TOP_K", "5")),
        hybrid_alpha=float(os.getenv("GEO_HYBRID_ALPHA", "0.45")),
        use_openai_embeddings=os.getenv("GEO_USE_OPENAI_EMBEDDINGS", "false").lower()
        in {"1", "true", "yes"},
        use_openai_reranker=os.getenv("GEO_USE_OPENAI_RERANKER", "false").lower()
        in {"1", "true", "yes"},
    )
