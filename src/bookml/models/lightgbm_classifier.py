from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import GridSearchClassifier, compute_imbalance_ratio


class LightGBMClassifier(GridSearchClassifier):

    def _build_estimator(self) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            subsample_freq=1,
            n_jobs=1,
            verbose=-1,
            random_state=self.random_state,
        )

    def _default_param_grid(self, y: np.ndarray) -> dict[str, list[Any]]:
        return {
            "max_depth": [3, 4, 5],
            "num_leaves": [15, 31],
            "learning_rate": [0.03, 0.05, 0.1],
            "n_estimators": [60, 100, 140],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "scale_pos_weight": sorted({1.0, compute_imbalance_ratio(y)}),
        }

    def feature_importances(self, top_k: int = 15) -> list[tuple[str, float]]:
        importances = self.best_estimator_.feature_importances_
        names = getattr(self.best_estimator_, "feature_names_in_", None)
        if names is None:
            names = getattr(self.best_estimator_, "feature_name_", None)
        top = np.argsort(importances)[::-1][:top_k]
        return [
            (str(names[i]) if names is not None else f"f{i}", float(importances[i]))
            for i in top
        ]

