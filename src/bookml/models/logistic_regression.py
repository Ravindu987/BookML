from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.pipeline import Pipeline

from .base import GridSearchClassifier, SelectiveScaler


class LogisticRegressionClassifier(GridSearchClassifier):
    """Logistic Regression classifier with selective feature scaling."""

    def __init__(
        self,
        tune: bool = True,
        param_grid: dict[str, list[Any]] | None = None,
        cv_folds: int = 5,
        refit_metric: str = "f1",
        random_state: int = 41,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | dict[Any, float] | None = None,
    ) -> None:
        super().__init__(
            tune=tune,
            param_grid=param_grid,
            cv_folds=cv_folds,
            refit_metric=refit_metric,
            random_state=random_state,
        )
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight

    def _build_estimator(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", SelectiveScaler()),
                (
                    "classifier",
                    SklearnLogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        class_weight=self.class_weight,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def _default_param_grid(self, y: np.ndarray) -> dict[str, list[Any]]:
        return {
            "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        }

    def fit(self, X: Any, y: Any) -> "LogisticRegressionClassifier":
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        elif hasattr(X, "shape") and len(X.shape) > 1:
            self.feature_names_in_ = [f"f{i}" for i in range(X.shape[1])]
        else:
            self.feature_names_in_ = None

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
        classifier = getattr(self.best_estimator_, "named_steps", {}).get("classifier", self.best_estimator_)
        if hasattr(classifier, "coef_"):
            coef = classifier.coef_[0]
            top_indices = np.argsort(np.abs(coef))[::-1][:top_k]
            names = getattr(self, "feature_names_in_", None)
            return [
                (str(names[i]) if names is not None and i < len(names) else f"f{i}", float(coef[i]))
                for i in top_indices
            ]
        return []
