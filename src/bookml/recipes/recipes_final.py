from __future__ import annotations

from typing import Any

# =============================================================================
# FINAL EXPERIMENTS: every model x every feature set
# =============================================================================
# FINAL_FEATURE_SETS: feature-set name -> ordered list of FEATURE_REGISTRY keys
FINAL_FEATURE_SETS: dict[str, list[str]] = {
    "tabular": ["price", "availability", "rating_numeric"],
    "tabular_tfidf": ["price", "availability", "rating_numeric", "tfidf"],
    "sentence_embeddings": ["sentence_embeddings"],
    "image_data": ["image_basic_features", "imageResNet18Vector"],
    "image_basic": ["image_basic_features"],
    "image_resnet": ["imageResNet18Vector"],
    "multimodal": ["multimodal_concatenated_feature"],
    "tabular_domain": ["price", "availability", "rating_numeric", "domain_features"],
    "domain": ["domain_features"],
    "tabular_domain_tfidf_image": [
        "price",
        "availability",
        "rating_numeric",
        "domain_features",
        "tfidf_80",
        "image_color_features",
    ],
    "tabular_domain_embeddings_image": [
        "price",
        "availability",
        "rating_numeric",
        "domain_features",
        "sentence_embeddings",
        "imageResNet18Vector",
    ],
}

# MODEL_REGISTRY keys of all models to be tested
FINAL_MODELS: list[str] = [
    "xgb",
    "lightgbm",
    "rf",
    "logistic_regression",
    "mlp",
    "gradient_boosting",
    "svm_rbf",
    "linear_svm",
    "ensemble_voting",
    "ensemble_stacking",
]

FINAL_MODEL_PARAMS: dict[str, Any] = {"tune": True, "random_state": 41}

# name -> config, same shape as recipes.EXPERIMENTS
FINAL_EXPERIMENTS: dict[str, dict[str, Any]] = {
    f"{model}__{fs_name}": {
        "name": f"{model}__{fs_name}",
        "feature_set": fs_name,
        "features": list(features),
        "model": model,
        "model_params": dict(FINAL_MODEL_PARAMS),
    }
    for fs_name, features in FINAL_FEATURE_SETS.items()
    for model in FINAL_MODELS
}
