
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

# --- Feature functions -------------------------------------------------------
from bookml.features.embeddings import add_sentence_embeddings
from bookml.features.images import add_image_basic_features, add_image_resnet18_vectors
from bookml.features.multimodal import add_multimodal_concat_features
from bookml.features.tabular import add_rating_numeric
from bookml.features.text import add_cleaned_description

# --- Models ------------------------------------------------------------------
from bookml.models.lightgbm_classifier import LightGBMClassifier
from bookml.models.rf_classifier import RandomForestClassifier
from bookml.models.svm_classifier import LinearSVMClassifier, SVMRBFClassifier
from bookml.models.xgb_classifier import XGBoostClassifier
from bookml.models.xgb_tfidf import XGBTfidfClassifier

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
    "sentence_embeddings": add_sentence_embeddings,  # description -> emb_0..emb_383
    "imageResNet18Vector": add_image_resnet18_vectors,  # image -> img_resnet18_0..img_resnet18_511
    "image_basic_features": add_image_basic_features,  # handcrafted 38-d visual features
    "multimodal_concatenated_feature": add_multimodal_concat_features,  # concatenated text emb + image vector (896-d)
}


# =============================================================================
# MODEL REGISTRY: key -> callable returning a fresh model with fit/predict
# =============================================================================
MODEL_REGISTRY: dict[str, ModelFactory] = {
    # General XGBoost model (works on any combination of numeric/tabular features or embeddings)
    "xgb": XGBoostClassifier,
    "general": XGBoostClassifier,
    "xgb_embeddings": XGBoostClassifier,
    # LightGBM model
    "lightgbm": LightGBMClassifier,
    # Random Forest model
    "rf": RandomForestClassifier,
    # Pipeline model with TF-IDF for text + numeric metadata
    "xgb_tfidf": XGBTfidfClassifier,
    # Support Vector Machine (RBF Kernel) with standard scaling
    "svm_rbf": SVMRBFClassifier,
    "svm": SVMRBFClassifier,
    # Support Vector Machine (Linear Kernel) with standard scaling
    "linear_svm": LinearSVMClassifier,
}



def build_model(key: str, **params: Any) -> Any:
    return MODEL_REGISTRY[key](**params)

