from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .base import GridSearchClassifier, SelectiveScaler


def _build_base_estimators(random_state: int) -> list[tuple[str, Any]]:
    lr_pipe = Pipeline([
        ("scaler", SelectiveScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    xgb_clf = xgb.XGBClassifier(
        eval_metric="logloss",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state,
        n_jobs=1,
    )
    mlp_pipe = Pipeline([
        ("scaler", SelectiveScaler()),
        (
            "classifier",
            MLPClassifier(
                hidden_layer_sizes=(64,),
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=600,
                early_stopping=False,
                random_state=random_state,
            ),
        ),
    ])
    return [("lr", lr_pipe), ("xgb", xgb_clf), ("mlp", mlp_pipe)]


class SoftVotingEnsembleClassifier(GridSearchClassifier):
    """Soft-Voting Ensemble combining Logistic Regression, XGBoost, and MLP."""

    def __init__(
        self,
        tune: bool = True,
        param_grid: dict[str, list[Any]] | None = None,
        cv_folds: int = 5,
        refit_metric: str = "f1",
        random_state: int = 41,
        weights: list[float] | None = None,
    ) -> None:
        super().__init__(
            tune=tune,
            param_grid=param_grid,
            cv_folds=cv_folds,
            refit_metric=refit_metric,
            random_state=random_state,
        )
        self.weights = weights

    def _build_estimator(self) -> VotingClassifier:
        return VotingClassifier(
            estimators=_build_base_estimators(self.random_state),
            voting="soft",
            weights=self.weights,
        )

    def _default_param_grid(self, y: np.ndarray) -> dict[str, list[Any]]:
        return {
            "weights": [
                None,
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0],
            ],
        }

    def feature_importances(self, top_k: int = 15) -> list[tuple[str, float]]:
        return []


class StackingLogRegClassifier(GridSearchClassifier):
    """Stacking Ensemble with Logistic Regression Meta-Learner over LogReg, XGBoost, and MLP."""

    def __init__(
        self,
        tune: bool = True,
        param_grid: dict[str, list[Any]] | None = None,
        cv_folds: int = 5,
        refit_metric: str = "f1",
        random_state: int = 41,
        meta_C: float = 1.0,
        stacking_cv: int = 3,
    ) -> None:
        super().__init__(
            tune=tune,
            param_grid=param_grid,
            cv_folds=cv_folds,
            refit_metric=refit_metric,
            random_state=random_state,
        )
        self.meta_C = meta_C
        self.stacking_cv = stacking_cv

    def _build_estimator(self) -> StackingClassifier:
        return StackingClassifier(
            estimators=_build_base_estimators(self.random_state),
            final_estimator=LogisticRegression(
                C=self.meta_C,
                max_iter=1000,
                random_state=self.random_state,
            ),
            cv=self.stacking_cv,
            n_jobs=1,
        )

    def _default_param_grid(self, y: np.ndarray) -> dict[str, list[Any]]:
        return {
            "final_estimator__C": [0.1, 1.0, 5.0],
        }

    def feature_importances(self, top_k: int = 15) -> list[tuple[str, float]]:
        return []
