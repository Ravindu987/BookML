
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from .base import GridSearchClassifier, compute_imbalance_ratio

TEXT_COLUMN = "cleaned_description"
NUMERIC_COLUMNS: list[str] = ["price", "rating_num", "availability"]


class DenseTransformer(TransformerMixin, BaseEstimator):

    def fit(self, X: Any, y: Any = None) -> "DenseTransformer":
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.ascontiguousarray(X.toarray(), dtype=np.float32)


class XGBTfidfClassifier(GridSearchClassifier):

    def _build_estimator(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                    TEXT_COLUMN,
                ),
                ("num", "passthrough", NUMERIC_COLUMNS),
            ]
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("to_dense", DenseTransformer()),
                (
                    "classifier",
                    xgb.XGBClassifier(
                        eval_metric="logloss",
                        n_jobs=1,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def _default_param_grid(self, y: np.ndarray) -> dict[str, list[Any]]:
        # Curated grid from the legacy script (64 combinations x 5 folds).
        return {
            "preprocessor__tfidf__max_features": [500, 1000],
            "classifier__max_depth": [3, 4],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__n_estimators": [50, 100],
            "classifier__subsample": [0.8, 1.0],
            "classifier__scale_pos_weight": sorted({1.0, compute_imbalance_ratio(y)}),
        }

    def feature_importances(self, top_k: int = 15) -> list[tuple[str, float]]:
        preprocessor = self.best_estimator_.named_steps["preprocessor"]
        classifier = self.best_estimator_.named_steps["classifier"]
        names = [
            n.replace("tfidf__", "text: ").replace("num__", "meta: ")
            for n in preprocessor.get_feature_names_out()
        ]
        importances = classifier.feature_importances_
        top = np.argsort(importances)[::-1][:top_k]
        return [(names[i], float(importances[i])) for i in top if importances[i] > 0]
