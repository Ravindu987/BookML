from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

from .base import GridSearchClassifier


class RandomForestClassifier(GridSearchClassifier):

    def _build_estimator(self) -> SklearnRandomForestClassifier:
        return SklearnRandomForestClassifier(
            n_jobs=1,
            random_state=self.random_state,
        )

    def _default_param_grid(self, y: np.ndarray) -> dict[str, list[Any]]:
        return {
            "n_estimators": [60, 100, 140],
            "max_depth": [3, 5, 8, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced"],
        }

    def feature_importances(self, top_k: int = 15) -> list[tuple[str, float]]:
        importances = self.best_estimator_.feature_importances_
        names = getattr(self.best_estimator_, "feature_names_in_", None)
        top = np.argsort(importances)[::-1][:top_k]
        return [
            (str(names[i]) if names is not None else f"f{i}", float(importances[i]))
            for i in top
        ]

