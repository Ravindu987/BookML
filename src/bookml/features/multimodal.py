from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bookml.paths import DATA_DIR
from .embeddings import EMBEDDING_PREFIX, add_sentence_embeddings
from .images import IMAGE_FEATURE_PREFIX, add_image_resnet18_vectors

logger = logging.getLogger(__name__)

MULTIMODAL_CONCAT_PREFIX: str = "multimodal_concat_"
MULTIMODAL_CONCAT_CACHE_PATH: Path = DATA_DIR / "multimodal_concatenated_features.npz"


def _load_cached_multimodal(expected_count: int, cache_path: Path) -> np.ndarray | None:
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        features = data["features"]
        if len(features) == expected_count:
            logger.info("Loaded multimodal concatenated features from cache '%s'.", cache_path.name)
            return np.ascontiguousarray(features, dtype=np.float32)
    except Exception as e:
        logger.warning("Failed to load multimodal cache '%s': %s. Recomputing.", cache_path, e)
    return None


def _save_multimodal_cache(features: np.ndarray, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features)
    logger.info("Saved multimodal concatenated features cache to '%s'.", cache_path.name)


def add_multimodal_concat_features(df: pd.DataFrame) -> pd.DataFrame:
    features = _load_cached_multimodal(len(df), MULTIMODAL_CONCAT_CACHE_PATH)

    if features is None:
        logger.info("Building concatenated multimodal features (text embeddings + image vectors)...")
        text_cols = [c for c in df.columns if c.startswith(EMBEDDING_PREFIX)]
        if text_cols:
            text_matrix = df[text_cols].to_numpy(dtype=np.float32)
        else:
            text_df = add_sentence_embeddings(df)
            text_cols = [c for c in text_df.columns if c.startswith(EMBEDDING_PREFIX)]
            text_matrix = text_df[text_cols].to_numpy(dtype=np.float32)
            
        img_cols = [c for c in df.columns if c.startswith(IMAGE_FEATURE_PREFIX)]
        if img_cols:
            img_matrix = df[img_cols].to_numpy(dtype=np.float32)
        else:
            img_df = add_image_resnet18_vectors(df)
            img_cols = [c for c in img_df.columns if c.startswith(IMAGE_FEATURE_PREFIX)]
            img_matrix = img_df[img_cols].to_numpy(dtype=np.float32)

        features = np.hstack([text_matrix, img_matrix])
        _save_multimodal_cache(features, MULTIMODAL_CONCAT_CACHE_PATH)

    columns = [f"{MULTIMODAL_CONCAT_PREFIX}{i}" for i in range(features.shape[1])]
    feat_df = pd.DataFrame(features, index=df.index, columns=columns)
    return pd.concat([df, feat_df], axis=1)
