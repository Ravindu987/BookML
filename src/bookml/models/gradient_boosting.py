from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier

from .base import GridSearchClassifier


class GradientBoostingClassifier(GridSearchClassifier):
    """Gradient Boosting Decision Tree classifier from scikit-learn."""

    def __init__(
        self,
        tune: bool = True,
        param_grid: dict[str, list[Any]] | None = None,
        cv_folds: int = 5,
        refit_metric: str = "f1",
        random_state: int = 41,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
    ) -> None:
        super().__init__(
            tune=tune,
            param_grid=param_grid,
            cv_folds=cv_folds,
            refit_metric=refit_metric,
            random_state=random_state,
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample

    def _build_estimator(self) -> SklearnGradientBoostingClassifier:
        return SklearnGradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            random_state=self.random_state,
        )

    def _default_param_grid(self, y: np.ndarray) -> dict[str, list[Any]]:
        return {
            "n_estimators": [60, 100, 140],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5],
            "subsample": [0.8, 1.0],
        }

    def feature_importances(self, top_k: int = 15) -> list[tuple[str, float]]:
        importances = self.best_estimator_.feature_importances_
        names = getattr(self.best_estimator_, "feature_names_in_", None)
        top = np.argsort(importances)[::-1][:top_k]
        return [
            (str(names[i]) if names is not None else f"f{i}", float(importances[i]))
            for i in top
        ]
