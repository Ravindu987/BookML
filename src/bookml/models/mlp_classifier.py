from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier
from sklearn.pipeline import Pipeline

from .base import GridSearchClassifier, SelectiveScaler


class MLPClassifier(GridSearchClassifier):
    """Multi-Layer Perceptron classifier with selective feature scaling."""

    def __init__(
        self,
        tune: bool = True,
        param_grid: dict[str, list[Any]] | None = None,
        cv_folds: int = 5,
        refit_metric: str = "f1",
        random_state: int = 41,
        hidden_layer_sizes: tuple[int, ...] = (64,),
        alpha: float = 0.01,
        learning_rate_init: float = 0.001,
        max_iter: int = 600,
        early_stopping: bool = False,
    ) -> None:
        super().__init__(
            tune=tune,
            param_grid=param_grid,
            cv_folds=cv_folds,
            refit_metric=refit_metric,
            random_state=random_state,
        )
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping

    def _build_estimator(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", SelectiveScaler()),
                (
                    "classifier",
                    SklearnMLPClassifier(
                        hidden_layer_sizes=self.hidden_layer_sizes,
                        alpha=self.alpha,
                        learning_rate_init=self.learning_rate_init,
                        max_iter=self.max_iter,
                        early_stopping=self.early_stopping,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def _default_param_grid(self, y: np.ndarray) -> dict[str, list[Any]]:
        return {
            "classifier__hidden_layer_sizes": [(64,), (64, 32)],
            "classifier__alpha": [0.001, 0.01],
        }

    def fit(self, X: Any, y: Any) -> "MLPClassifier":
        if self.param_grid is not None:
            normalized_grid: dict[str, list[Any]] = {}
            for k, v in self.param_grid.items():
                if not k.startswith("classifier__") and not k.startswith("scaler__"):
                    normalized_grid[f"classifier__{k}"] = v
                else:
                    normalized_grid[k] = v
            self.param_grid = normalized_grid
        return super().fit(X, y)

    def feature_importances(self, top_k: int = 15) -> list[tuple[str, float]]:
        return []
