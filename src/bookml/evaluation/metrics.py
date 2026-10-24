
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_NAMES: list[str] = ["Nonfiction (Class 0)", "Fiction (Class 1)"]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics



def format_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=TARGET_NAMES, zero_division=0
    )
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        index=["Actual Nonfiction", "Actual Fiction"],
        columns=["Predicted Nonfiction", "Predicted Fiction"],
    )
    return f"{report}\nConfusion matrix:\n{cm.to_string()}"
