from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

# --- Feature functions -------------------------------------------------------
from bookml.features.embeddings import add_sentence_embeddings
from bookml.features.images import (
    add_image_basic_features,
    add_image_color_features,
    add_image_resnet18_vectors,
)
from bookml.features.multimodal import add_multimodal_concat_features
from bookml.features.tabular import (
    add_desc_char_len,
    add_desc_word_count,
    add_digit_count,
    add_domain_features,
    add_exclamation_count,
    add_has_subtitle,
    add_quote_count,
    add_rating_numeric,
    add_title_char_len,
    add_title_digit_count,
    add_title_upper_ratio,
    add_title_word_count,
    add_total_char_len,
    add_total_word_count,
)
from bookml.features.text import (
    add_cleaned_description,
    add_tfidf_80_features,
    add_tfidf_features,
)

# --- Models ------------------------------------------------------------------
from bookml.models.ensemble import SoftVotingEnsembleClassifier, StackingLogRegClassifier
from bookml.models.gradient_boosting import GradientBoostingClassifier
from bookml.models.lightgbm_classifier import LightGBMClassifier
from bookml.models.logistic_regression import LogisticRegressionClassifier
from bookml.models.mlp_classifier import MLPClassifier
from bookml.models.rf_classifier import RandomForestClassifier
from bookml.models.svm_classifier import LinearSVMClassifier, SVMRBFClassifier
from bookml.models.xgb_classifier import XGBoostClassifier

FeatureFn = Callable[[pd.DataFrame], pd.DataFrame]
FeatureEntry = FeatureFn | str
ModelFactory = Callable[..., Any]


# =============================================================================
# FEATURE REGISTRY: key -> function (df -> df) or direct CSV column name (str)
# =============================================================================
FEATURE_REGISTRY: dict[str, FeatureEntry] = {
    # Direct from CSV
    "price": "price",
    "title": "title",
    "availability": "availability",
    "description": "description",

    # Feature engineering functions (df -> df)
    "rating_numeric": add_rating_numeric,            # rating -> rating_num
    "clean_description": add_cleaned_description,    # description -> cleaned_description
    "tfidf": add_tfidf_features,
    "tfidf_80": add_tfidf_80_features,
    "sentence_embeddings": add_sentence_embeddings,
    "imageResNet18Vector": add_image_resnet18_vectors,
    "image_basic_features": add_image_basic_features,
    "image_color_features": add_image_color_features,
    "imageColorVector": add_image_color_features,
    "multimodal_concatenated_feature": add_multimodal_concat_features,

    # Domain & stylistic engineered features
    "domain_features": add_domain_features,
    "engineered_features": add_domain_features,

    # Individual feature functions
    "has_subtitle": add_has_subtitle,
    "title_upper_ratio": add_title_upper_ratio,
    "title_char_len": add_title_char_len,
    "desc_char_len": add_desc_char_len,
    "total_char_len": add_total_char_len,
    "title_word_count": add_title_word_count,
    "desc_word_count": add_desc_word_count,
    "total_word_count": add_total_word_count,
    "quote_count": add_quote_count,
    "digit_count": add_digit_count,
    "title_digit_count": add_title_digit_count,
    "exclamation_count": add_exclamation_count,
}


# =============================================================================
# MODEL REGISTRY: key -> callable returning a fresh model with fit/predict
# =============================================================================
MODEL_REGISTRY: dict[str, ModelFactory] = {
    # General XGBoost model
    "xgb": XGBoostClassifier,
    "xgboost": XGBoostClassifier,
    "general": XGBoostClassifier,
    "xgb_embeddings": XGBoostClassifier,

    # LightGBM model
    "lightgbm": LightGBMClassifier,
    "lgbm": LightGBMClassifier,

    # Random Forest model
    "rf": RandomForestClassifier,
    "random_forest": RandomForestClassifier,

    # Logistic Regression model
    "logistic_regression": LogisticRegressionClassifier,
    "logreg": LogisticRegressionClassifier,
    "lr": LogisticRegressionClassifier,

    # Multi-Layer Perceptron (Neural Net)
    "mlp": MLPClassifier,
    "neural_net": MLPClassifier,

    # Gradient Boosting model
    "gradient_boosting": GradientBoostingClassifier,
    "gb": GradientBoostingClassifier,

    # Support Vector Machine (RBF Kernel) with standard scaling
    "svm_rbf": SVMRBFClassifier,
    "svm": SVMRBFClassifier,

    # Support Vector Machine (Linear Kernel) with standard scaling
    "linear_svm": LinearSVMClassifier,

    # Ensembles from Notebook
    "ensemble_voting": SoftVotingEnsembleClassifier,
    "voting_soft": SoftVotingEnsembleClassifier,
    "voting": SoftVotingEnsembleClassifier,
    "ensemble_stacking": StackingLogRegClassifier,
    "stacking_logreg": StackingLogRegClassifier,
    "stacking": StackingLogRegClassifier,
}


def build_model(key: str, **params: Any) -> Any:
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{key}' not found in MODEL_REGISTRY. Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key](**params)
