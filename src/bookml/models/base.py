
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold


class GridSearchClassifier(ClassifierMixin, BaseEstimator):

    SCORING: dict[str, str] = {"f1": "f1", "accuracy": "accuracy", "roc_auc": "roc_auc"}

    def __init__(
        self,
        tune: bool = True,
        param_grid: dict[str, list[Any]] | None = None,
        cv_folds: int = 5,
        refit_metric: str = "f1",
        random_state: int = 41,
    ) -> None:
        self.tune = tune
        self.param_grid = param_grid
        self.cv_folds = cv_folds
        self.refit_metric = refit_metric
        self.random_state = random_state

    # Overriden functions
    def _build_estimator(self) -> BaseEstimator:
        raise NotImplementedError

    def _default_param_grid(self, y: np.ndarray) -> dict[str, list[Any]]:
        raise NotImplementedError

    # Train
    def fit(self, X: Any, y: Any) -> "GridSearchClassifier":
        self.classes_ = np.unique(y)

        estimator = self._build_estimator()

        if not self.tune:
            estimator.fit(X, y)
            self.best_estimator_ = estimator
            self.best_params_: dict[str, Any] = {}
            self.best_score_: float | None = None
            self.cv_results_: dict[str, Any] | None = None
            return self

        grid = self.param_grid if self.param_grid is not None else self._default_param_grid(y)
        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring=self.SCORING,
            refit=self.refit_metric,
            return_train_score=False,
        )
        search.fit(X, y)

        self.best_estimator_ = search.best_estimator_
        self.best_params_ = dict(search.best_params_)
        self.best_score_ = float(search.best_score_)
        self.cv_results_ = search.cv_results_
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.best_estimator_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.best_estimator_.predict_proba(X)


def compute_imbalance_ratio(y: np.ndarray) -> float:
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return round(neg / pos, 2)
