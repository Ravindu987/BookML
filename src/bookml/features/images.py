from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from bookml.paths import DATA_DIR, IMAGES_DIR

logger = logging.getLogger(__name__)

IMAGE_FEATURE_PREFIX = "img_resnet18_"
IMAGE_CACHE_PATH: Path = DATA_DIR / "image_resnet18_embeddings.npz"


def resolve_image_path(row: pd.Series, images_dir: Path) -> Path:
    title = str(row.get("title", ""))
    category = str(row.get("category", ""))

    if "11/22" in title:
        candidate = images_dir / "Fiction_11_22_63.jpg"
        if candidate.exists():
            return candidate

    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", title)[:50]
    candidate = images_dir / f"{category}_{sanitized}.jpg"
    if candidate.exists():
        return candidate

    for img_file in images_dir.glob("*.jpg"):
        if sanitized in img_file.name:
            return img_file

    raise FileNotFoundError(
        f"Cover image not found for '{title}' (searched in {images_dir})"
    )


def extract_resnet18_vectors(
    image_paths: list[Path],
    batch_size: int = 16,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing pretrained ResNet-18 model on device '%s'...", device)

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    # Standard ImageNet preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_embeddings: list[np.ndarray] = []
    num_samples = len(image_paths)
    logger.info("Extracting ResNet-18 vectors for %d images (batch_size=%d)...", num_samples, batch_size)

    for i in range(0, num_samples, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors: list[torch.Tensor] = []

        for path in batch_paths:
            with Image.open(path).convert("RGB") as img:
                batch_tensors.append(transform(img))

        batch_tensor = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            embeddings = model(batch_tensor)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.ascontiguousarray(np.vstack(all_embeddings), dtype=np.float32)


def _load_cached_embeddings(expected_count: int, cache_path: Path) -> np.ndarray | None:
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        embeddings = data["embeddings"]
        if len(embeddings) == expected_count:
            logger.info("Loaded cached features from '%s'.", cache_path.name)
            return np.ascontiguousarray(embeddings, dtype=np.float32)
    except Exception as e:
        logger.warning("Failed to load image cache '%s': %s. Recomputing.", cache_path, e)
    return None


def _save_embeddings_cache(embeddings: np.ndarray, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings)
    logger.info("Saved feature cache to '%s'.", cache_path.name)


BASIC_IMAGE_COLUMNS: list[str] = [
    "img_r_mean", "img_g_mean", "img_b_mean",
    "img_r_std", "img_g_std", "img_b_std",
    "img_h_mean", "img_s_mean", "img_v_mean",
    "img_h_std", "img_s_std", "img_v_std",
    "img_lum_mean", "img_contrast",
    *[f"img_r_hist_{i}" for i in range(8)],
    *[f"img_g_hist_{i}" for i in range(8)],
    *[f"img_b_hist_{i}" for i in range(8)],
]

BASIC_IMAGE_CACHE_PATH: Path = DATA_DIR / "image_basic_features.npz"


def extract_basic_image_features(image_paths: list[Path]) -> np.ndarray:
    feature_list: list[np.ndarray] = []
    logger.info("Extracting 38 handcrafted visual features for %d images...", len(image_paths))

    for path in image_paths:
        with Image.open(path).convert("RGB") as img:
            arr_rgb = np.array(img, dtype=np.float32) / 255.0
            r_mean, g_mean, b_mean = arr_rgb.mean(axis=(0, 1))
            r_std, g_std, b_std = arr_rgb.std(axis=(0, 1))
            img_hsv = img.convert("HSV")
            arr_hsv = np.array(img_hsv, dtype=np.float32) / 255.0
            h_mean, s_mean, v_mean = arr_hsv.mean(axis=(0, 1))
            h_std, s_std, v_std = arr_hsv.std(axis=(0, 1))
            gray = img.convert("L")
            arr_gray = np.array(gray, dtype=np.float32) / 255.0
            lum_mean = arr_gray.mean()
            contrast = arr_gray.std()
            r_hist, _ = np.histogram(arr_rgb[:, :, 0], bins=8, range=(0, 1), density=True)
            g_hist, _ = np.histogram(arr_rgb[:, :, 1], bins=8, range=(0, 1), density=True)
            b_hist, _ = np.histogram(arr_rgb[:, :, 2], bins=8, range=(0, 1), density=True)

            vec = np.hstack([
                [r_mean, g_mean, b_mean, r_std, g_std, b_std],
                [h_mean, s_mean, v_mean, h_std, s_std, v_std],
                [lum_mean, contrast],
                r_hist,
                g_hist,
                b_hist,
            ])
            feature_list.append(vec)

    return np.ascontiguousarray(np.array(feature_list), dtype=np.float32)


def add_image_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    features = _load_cached_embeddings(len(df), BASIC_IMAGE_CACHE_PATH)

    if features is None:
        images_dir = IMAGES_DIR
        if not images_dir.exists():
            root_candidate = DATA_DIR.parents[3] / "data" / "images"
            if root_candidate.exists():
                images_dir = root_candidate

        image_paths = [resolve_image_path(row, images_dir) for _, row in df.iterrows()]
        features = extract_basic_image_features(image_paths)
        _save_embeddings_cache(features, BASIC_IMAGE_CACHE_PATH)

    feat_df = pd.DataFrame(features, index=df.index, columns=BASIC_IMAGE_COLUMNS)
    return pd.concat([df, feat_df], axis=1)


def add_image_resnet18_vectors(df: pd.DataFrame) -> pd.DataFrame:
    embeddings = _load_cached_embeddings(len(df), IMAGE_CACHE_PATH)

    if embeddings is None:
        images_dir = IMAGES_DIR
        if not images_dir.exists():
            root_candidate = DATA_DIR.parents[3] / "data" / "images"
            if root_candidate.exists():
                images_dir = root_candidate

        image_paths = [resolve_image_path(row, images_dir) for _, row in df.iterrows()]
        embeddings = extract_resnet18_vectors(image_paths)
        _save_embeddings_cache(embeddings, IMAGE_CACHE_PATH)

    cols = [f"{IMAGE_FEATURE_PREFIX}{i}" for i in range(embeddings.shape[1])]
    img_df = pd.DataFrame(embeddings, index=df.index, columns=cols)
    return pd.concat([df, img_df], axis=1)

