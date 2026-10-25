from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import random
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys

# Ensure src is on sys.path
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from bookml.data_loader.load_data import (
    CATEGORY_MAP,
    ID_COLUMN,
    INVERSE_CATEGORY_MAP,
    TARGET_COLUMN,
    load_raw_data,
)
from bookml.features.images import extract_basic_image_features, resolve_image_path
from bookml.paths import BOOKS_CSV, DATA_DIR, IMAGES_DIR, OVERALL_CSV, RESULTS_DIR, VAL_IDS_TXT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("image_flow")

SEED = 42
IMG_SIZE = (128, 128)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ShallowBookCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(64 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PILImageDataset(Dataset):
    def __init__(self, images: list[Any], labels: list[int], orig_ids: list[int], transform=None):
        self.images = list(images)
        self.labels = list(labels)
        self.orig_ids = list(orig_ids)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.images[idx]
        if isinstance(item, (str, Path)):
            with Image.open(item).convert("RGB") as img:
                tensor_img = self.transform(img) if self.transform else transforms.ToTensor()(img)
        elif isinstance(item, Image.Image):
            tensor_img = self.transform(item) if self.transform else transforms.ToTensor()(item)
        else:
            raise TypeError(f"Unsupported image type: {type(item)}")

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_img, label


CNN_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


def generate_augmented_variants(image: Image.Image) -> list[tuple[str, Image.Image]]:
    w, h = image.size
    orig = image.copy()
    hflip = TF.hflip(orig)
    rot_pos = TF.rotate(orig, angle=10, fill=[245, 245, 245])
    rot_neg = TF.rotate(orig, angle=-10, fill=[245, 245, 245])

    crop_h = int(h * 0.88)
    crop_w = int(w * 0.88)
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    crop_img = TF.crop(orig, top, left, crop_h, crop_w)
    crop_zoom = TF.resize(crop_img, (h, w))

    return [
        ("original", orig),
        ("hflip", hflip),
        ("rotate_pos10", rot_pos),
        ("rotate_neg10", rot_neg),
        ("crop_zoom", crop_zoom),
    ]


def prepare_data() -> dict[str, Any]:
    df_all = load_raw_data(BOOKS_CSV)
    df_all["img_path"] = [resolve_image_path(row, IMAGES_DIR) for _, row in df_all.iterrows()]

    with open(VAL_IDS_TXT, "r", encoding="utf-8") as f:
        val_ids = [int(line.strip().split()[0]) for line in f if line.strip() and not line.startswith("#")]

    test_mask = df_all[ID_COLUMN].isin(val_ids)
    test_df = df_all[test_mask].copy().reset_index(drop=True)
    train_df = df_all[~test_mask].copy().reset_index(drop=True)

    logger.info("Total books: %d | Train: %d | Test: %d", len(df_all), len(train_df), len(test_df))

    test_images: list[Image.Image] = [Image.open(p).convert("RGB") for p in test_df["img_path"]]
    test_labels: list[int] = test_df[TARGET_COLUMN].tolist()
    test_ids_list: list[int] = test_df[ID_COLUMN].tolist()
    train_images: list[Image.Image] = [Image.open(p).convert("RGB") for p in train_df["img_path"]]
    train_labels: list[int] = train_df[TARGET_COLUMN].tolist()
    train_ids_list: list[int] = train_df[ID_COLUMN].tolist()

    aug_train_records = []
    for _, row in train_df.iterrows():
        with Image.open(row["img_path"]).convert("RGB") as img:
            variants = generate_augmented_variants(img)
            for aug_type, aug_img in variants:
                aug_train_records.append({
                    "orig_id": row[ID_COLUMN],
                    "title": row["title"],
                    "category": row["category"],
                    "target": row[TARGET_COLUMN],
                    "aug_type": aug_type,
                    "image": aug_img,
                })
    aug_train_df = pd.DataFrame(aug_train_records)
    logger.info("Augmented training set: %d samples (5x multiplier)", len(aug_train_df))

    return {
        "train_df": train_df,
        "test_df": test_df,
        "aug_train_df": aug_train_df,
        "train_images": train_images,
        "train_labels": train_labels,
        "train_ids": train_ids_list,
        "test_images": test_images,
        "test_labels": test_labels,
        "test_ids": test_ids_list,
    }



def save_results_and_model(
    experiment_name: str,
    model_name: str,
    feature_desc: str,
    model: Any,
    metrics: dict[str, float],
    cm_dict: dict[str, int],
    cv_score: float | None,
    best_params: dict[str, Any],
    predictions: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    models_dir = RESULTS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.now().isoformat(timespec="seconds")
    json_path = RESULTS_DIR / f"{experiment_name}.json"

    payload = {
        "experiment": experiment_name,
        "timestamp": now_iso,
        "model": model_name,
        "features": feature_desc,
        "cv_best_score": cv_score,
        "best_params": best_params,
        "validation_metrics": metrics,
        "confusion_matrix": cm_dict,
        "predictions": predictions.to_dict(orient="records"),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Results written to '%s'.", json_path)

    csv_row = {
        "timestamp": now_iso,
        "experiment": experiment_name,
        "model": model_name,
        "features": json.dumps([feature_desc]),
        "cv_best_score": cv_score,
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "best_params": json.dumps(best_params, default=str),
    }
    pd.DataFrame([csv_row]).to_csv(OVERALL_CSV, mode="a", header=False, index=False)
    logger.info("Summary appended to '%s'.", OVERALL_CSV)

    if isinstance(model, nn.Module):
        model_file = models_dir / f"{experiment_name}.pt"
        torch.save(model.state_dict(), model_file)
        logger.info("PyTorch weights saved to '%s'.", model_file)
    else:
        model_file = models_dir / f"{experiment_name}.joblib"
        joblib.dump(model, model_file)
        logger.info("Model artifact saved to '%s'.", model_file)


def extract_features_from_pil(images: list[Image.Image]) -> np.ndarray:
    feature_list = []
    for img in images:
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

    return np.ascontiguousarray(np.array(feature_list, dtype=np.float32))


def run_classical_experiment(
    model_type: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    set_seed(SEED)
    experiment_name = f"image_{model_type}_aug"
    feature_desc = "image_basic_features_augmented"

    logger.info("=" * 70)
    logger.info("RUNNING EXPERIMENT: %s (Model: %s | Augmented: True)", experiment_name, model_type)
    logger.info("=" * 70)

    test_X = extract_features_from_pil(data["test_images"])
    test_y = np.array(data["test_labels"])

    train_X = extract_features_from_pil(data["aug_train_df"]["image"].tolist())
    train_y = np.array(data["aug_train_df"]["target"])
    train_groups = np.array(data["aug_train_df"]["orig_id"])

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    if model_type == "lr":
        model_name = "LogisticRegression"
        model = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
        best_params = {"class_weight": "balanced", "max_iter": 1000, "random_state": SEED}
    elif model_type in ("svm", "svm_rbf"):
        model_name = "SVMRBFClassifier"
        model = SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", class_weight="balanced", random_state=SEED)
        best_params = {"C": 1.0, "gamma": "scale", "class_weight": "balanced", "random_state": SEED}
    elif model_type == "rf":
        model_name = "RandomForestClassifier"
        model = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight="balanced", random_state=SEED)
        best_params = {"n_estimators": 100, "max_depth": 4, "class_weight": "balanced", "random_state": SEED}
    else:
        raise ValueError(f"Unknown classical model type: {model_type}")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = cv.split(train_X_scaled, train_y, groups=train_groups)

    cv_f1_scores = []
    for train_idx, val_idx in splits:
        fold_model = model.__class__(**model.get_params())
        fold_model.fit(train_X_scaled[train_idx], train_y[train_idx])
        preds_val = fold_model.predict(train_X_scaled[val_idx])
        cv_f1_scores.append(f1_score(train_y[val_idx], preds_val, zero_division=0))
    cv_score = float(np.mean(cv_f1_scores))
    logger.info("5-Fold CV Score (F1): %.4f", cv_score)

    model.fit(train_X_scaled, train_y)

    val_preds = model.predict(test_X_scaled)
    val_probs = model.predict_proba(test_X_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = float(accuracy_score(test_y, val_preds))
    precision = float(precision_score(test_y, val_preds, zero_division=0))
    recall = float(recall_score(test_y, val_preds, zero_division=0))
    f1 = float(f1_score(test_y, val_preds, zero_division=0))
    roc_auc = float(roc_auc_score(test_y, val_probs)) if val_probs is not None else None

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
    }
    logger.info("Validation Metrics (Test Set): %s", metrics)

    tn, fp, fn, tp = confusion_matrix(test_y, val_preds, labels=[0, 1]).ravel()
    cm_dict = {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}

    predictions_df = pd.DataFrame({
        ID_COLUMN: data["test_ids"],
        "title": data["test_df"]["title"].tolist(),
        "actual": [INVERSE_CATEGORY_MAP[y] for y in test_y],
        "predicted": [INVERSE_CATEGORY_MAP[p] for p in val_preds],
        "probability_fiction": val_probs if val_probs is not None else [None] * len(val_preds),
    })

    save_results_and_model(
        experiment_name=experiment_name,
        model_name=model_name,
        feature_desc=feature_desc,
        model=model,
        metrics=metrics,
        cm_dict=cm_dict,
        cv_score=round(cv_score, 4),
        best_params=best_params,
        predictions=predictions_df,
    )

    return {"experiment": experiment_name, "metrics": metrics, "cv_score": cv_score}


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


def eval_dataset(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * imgs.size(0)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_probs), np.array(all_targets)


def run_cnn_experiment(augmented: bool, data: dict[str, Any], epochs: int = 30) -> dict[str, Any]:
    """Train and evaluate the ShallowBookCNN on raw 128x128 images."""
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = "_aug" if augmented else ""
    experiment_name = f"image_shallow_cnn{suffix}"
    feature_desc = "raw_images_128x128_augmented" if augmented else "raw_images_128x128"

    logger.info("=" * 70)
    logger.info("RUNNING EXPERIMENT: %s (Device: %s | Epochs: %d)", experiment_name, device, epochs)
    logger.info("=" * 70)

    test_ds = PILImageDataset(data["test_images"], data["test_labels"], data["test_ids"], transform=CNN_TRANSFORM)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    if augmented:
        train_images = data["aug_train_df"]["image"].tolist()
        train_labels = data["aug_train_df"]["target"].tolist()
        train_ids = data["aug_train_df"]["orig_id"].tolist()
    else:
        train_images = data["train_images"]
        train_labels = data["train_labels"]
        train_ids = data["train_ids"]

    logger.info("Conducting 5-Fold Cross-Validation...")
    if augmented:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = list(cv.split(train_images, train_labels, groups=train_ids))
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = list(cv.split(train_images, train_labels))

    cv_f1_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        fold_train_ds = PILImageDataset(
            [train_images[i] for i in train_idx],
            [train_labels[i] for i in train_idx],
            [train_ids[i] for i in train_idx],
            transform=CNN_TRANSFORM,
        )
        fold_val_ds = PILImageDataset(
            [train_images[i] for i in val_idx],
            [train_labels[i] for i in val_idx],
            [train_ids[i] for i in val_idx],
            transform=CNN_TRANSFORM,
        )
        f_train_loader = DataLoader(fold_train_ds, batch_size=16, shuffle=True)
        f_val_loader = DataLoader(fold_val_ds, batch_size=16, shuffle=False)

        fold_model = ShallowBookCNN().to(device)
        targets_arr = np.array([train_labels[i] for i in train_idx])
        class_w = torch.tensor([
            1.0 / (targets_arr == 0).sum(),
            1.0 / (targets_arr == 1).sum(),
        ], dtype=torch.float).to(device)
        class_w = class_w / class_w.sum()

        crit = nn.CrossEntropyLoss(weight=class_w)
        opt = torch.optim.Adam(fold_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15, eta_min=1e-5)

        for ep in range(15):
            train_epoch(fold_model, f_train_loader, crit, opt, device)
            sched.step()

        _, _, val_p, _, val_t = eval_dataset(fold_model, f_val_loader, crit, device)
        fold_f1 = f1_score(val_t, val_p, zero_division=0)
        cv_f1_scores.append(fold_f1)

    cv_score = float(np.mean(cv_f1_scores))
    logger.info("5-Fold CV Score (F1): %.4f", cv_score)

    full_train_ds = PILImageDataset(train_images, train_labels, train_ids, transform=CNN_TRANSFORM)
    full_train_loader = DataLoader(full_train_ds, batch_size=16, shuffle=True)

    final_model = ShallowBookCNN().to(device)
    targets_arr = np.array(train_labels)
    class_w = torch.tensor([
        1.0 / (targets_arr == 0).sum(),
        1.0 / (targets_arr == 1).sum(),
    ], dtype=torch.float).to(device)
    class_w = class_w / class_w.sum()

    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    logger.info("Training Final ShallowBookCNN on %d samples for %d epochs...", len(full_train_ds), epochs)
    for epoch in range(1, epochs + 1):
        loss, acc = train_epoch(final_model, full_train_loader, criterion, optimizer, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs:
            logger.info("  Epoch %2d/%2d - Loss: %.4f | Train Acc: %.4f", epoch, epochs, loss, acc)

    test_loss, test_acc, test_preds, test_probs, test_targets = eval_dataset(final_model, test_loader, criterion, device)

    accuracy = float(accuracy_score(test_targets, test_preds))
    precision = float(precision_score(test_targets, test_preds, zero_division=0))
    recall = float(recall_score(test_targets, test_preds, zero_division=0))
    f1 = float(f1_score(test_targets, test_preds, zero_division=0))
    roc_auc = float(roc_auc_score(test_targets, test_probs))

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
    }
    logger.info("Validation Metrics (Test Set): %s", metrics)

    tn, fp, fn, tp = confusion_matrix(test_targets, test_preds, labels=[0, 1]).ravel()
    cm_dict = {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}

    predictions_df = pd.DataFrame({
        ID_COLUMN: data["test_ids"],
        "title": data["test_df"]["title"].tolist(),
        "actual": [INVERSE_CATEGORY_MAP[y] for y in test_targets],
        "predicted": [INVERSE_CATEGORY_MAP[p] for p in test_preds],
        "probability_fiction": test_probs,
    })

    best_params = {
        "architecture": "ShallowBookCNN (3 conv stages + BN + 2-layer FC)",
        "epochs": epochs,
        "batch_size": 16,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "parameters": 56674,
    }

    save_results_and_model(
        experiment_name=experiment_name,
        model_name="ShallowBookCNN",
        feature_desc=feature_desc,
        model=final_model,
        metrics=metrics,
        cm_dict=cm_dict,
        cv_score=round(cv_score, 4),
        best_params=best_params,
        predictions=predictions_df,
    )

    return {"experiment": experiment_name, "metrics": metrics, "cv_score": cv_score}


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Image Flow (Augmented Classical Baselines & ShallowBookCNN on Raw and Augmented Data)."
    )
    parser.add_argument(
        "--model",
        choices=["lr", "svm", "rf", "cnn", "all"],
        default="all",
        help="Model to run: lr (Logistic Regression - augmented), svm (SVM RBF - augmented), "
             "rf (Random Forest - augmented), cnn (ShallowBookCNN - raw and/or augmented), all (default).",
    )
    parser.add_argument(
        "--augment",
        choices=["yes", "no", "both"],
        default="both",
        help="Whether to run CNN with data augmentation: yes, no, or both (default). "
             "Note: Classical models (lr, svm, rf) run on augmented data only.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for ShallowBookCNN (default: 30).")
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all 5 target experiments (image_lr_aug, image_svm_aug, image_rf_aug, image_shallow_cnn, image_shallow_cnn_aug).",
    )

    args = parser.parse_args()

    data = prepare_data()
    summary_results = []

    if args.run_all or args.model == "all":
        # 1. Classical models on augmented data
        for m in ["lr", "svm", "rf"]:
            res = run_classical_experiment(model_type=m, data=data)
            summary_results.append(res)

        # 2. ShallowBookCNN on raw data
        if args.augment in ("both", "no"):
            res = run_cnn_experiment(augmented=False, data=data, epochs=args.epochs)
            summary_results.append(res)

        # 3. ShallowBookCNN on augmented data
        if args.augment in ("both", "yes"):
            res = run_cnn_experiment(augmented=True, data=data, epochs=args.epochs)
            summary_results.append(res)

    elif args.model in ("lr", "svm", "rf"):
        res = run_classical_experiment(model_type=args.model, data=data)
        summary_results.append(res)

    elif args.model == "cnn":
        if args.augment in ("both", "no"):
            res = run_cnn_experiment(augmented=False, data=data, epochs=args.epochs)
            summary_results.append(res)
        if args.augment in ("both", "yes"):
            res = run_cnn_experiment(augmented=True, data=data, epochs=args.epochs)
            summary_results.append(res)

    print("\n" + "=" * 78)
    print("IMAGE FLOW EXECUTION SUMMARY:")
    print("=" * 78)
    summary_df = pd.DataFrame([
        {
            "Experiment": r["experiment"],
            "CV Score (F1)": r["cv_score"],
            "Accuracy": r["metrics"]["accuracy"],
            "Precision": r["metrics"]["precision"],
            "Recall": r["metrics"]["recall"],
            "F1-Score": r["metrics"]["f1"],
            "ROC-AUC": r["metrics"]["roc_auc"],
        }
        for r in summary_results
    ])
    print(summary_df.to_string(index=False))
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
