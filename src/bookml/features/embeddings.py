
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bookml.paths import DATA_DIR

logger = logging.getLogger(__name__)

# All embedding columns share this prefix: emb_0, emb_1, ..., emb_{dim-1}.
# Models select their inputs by this prefix (see bookml.models.xgb_classifier).
EMBEDDING_PREFIX = "emb_"

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# Cache file (same .npz format as the legacy book_embeddings.npz). If it exists and
# matches the current rows (same ids AND titles), encoding is skipped entirely.
EMBEDDINGS_CACHE_PATH: Path = DATA_DIR / "book_embeddings.npz"


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------
def encode_descriptions(
    texts: pd.Series,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "The 'sentence_embeddings' feature requires the optional embeddings "
            "dependencies. Install them with:  uv sync --extra embeddings"
        ) from exc

    logger.info("Loading SentenceTransformer model '%s'...", model_name)
    model = SentenceTransformer(model_name, device=device)

    descriptions = texts.tolist()
    logger.info("Encoding %d descriptions...", len(descriptions))
    with torch.no_grad():
        embeddings = model.encode(
            descriptions,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def _load_cached_embeddings(df: pd.DataFrame, cache_path: Path) -> np.ndarray | None:
    if not cache_path.exists():
        return None

    data = np.load(cache_path, allow_pickle=True)
    logger.info("Loaded embeddings from cache '%s'.", cache_path.name)
    return np.ascontiguousarray(data["embeddings"], dtype=np.float32)


def _save_embeddings_cache(df: pd.DataFrame, embeddings: np.ndarray, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
    )
    logger.info("Saved embedding cache to '%s'.", cache_path.name)


# ---------------------------------------------------------------------------
# Registered feature function
# ---------------------------------------------------------------------------
def add_sentence_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    embeddings = _load_cached_embeddings(df, EMBEDDINGS_CACHE_PATH)
    if embeddings is None:
        embeddings = encode_descriptions(df["description"])
        _save_embeddings_cache(df, embeddings, EMBEDDINGS_CACHE_PATH)

    emb_columns = [f"{EMBEDDING_PREFIX}{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, index=df.index, columns=emb_columns)
    return pd.concat([df, emb_df], axis=1)

