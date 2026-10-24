from __future__ import annotations

from typing import Any

# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================
# Keys:
#   name          : identifier used for the results file name
#   features      : ordered list of FEATURE_REGISTRY keys, applied sequentially
#   model         : a single MODEL_REGISTRY key
#   model_params  : (optional) kwargs passed to the model constructor
#
# Selectable with --experiment <name>.
# Team members: add your own configs here instead of editing the pipeline.
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "xgb_tfidf_baseline": {
        "name": "xgb_tfidf_baseline",
        "features": [
            "price",             
            "availability",      
            "rating_numeric",   
            "clean_description", 
        ],
        "model": "xgb_tfidf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "xgb_embeddings": {
        "name": "xgb_embeddings",
        "features": ["sentence_embeddings"],
        "model": "xgb",
        "model_params": {"tune": True, "random_state": 41},
    },
    "xgb_tabular": {
        "name": "xgb_tabular",
        "features": ["price", "availability", "rating_numeric"],
        "model": "xgb",
        "model_params": {"tune": True, "random_state": 41},
    },
    "rf_tabular": {
        "name": "rf_tabular",
        "features": ["price", "availability", "rating_numeric"],
        "model": "rf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "rf_embeddings": {
        "name": "rf_embeddings",
        "features": ["sentence_embeddings"],
        "model": "rf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "lightgbm_tabular": {
        "name": "lightgbm_tabular",
        "features": ["price", "availability", "rating_numeric"],
        "model": "lightgbm",
        "model_params": {"tune": True, "random_state": 41},
    },
    "lightgbm_embeddings": {
        "name": "lightgbm_embeddings",
        "features": ["sentence_embeddings"],
        "model": "lightgbm",
        "model_params": {"tune": True, "random_state": 41},
    },
    "svm_tabular": {
        "name": "svm_tabular",
        "features": ["price", "availability", "rating_numeric"],
        "model": "svm_rbf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "svm_embeddings": {
        "name": "svm_embeddings",
        "features": ["sentence_embeddings"],
        "model": "svm_rbf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "svm_images": {
        "name": "svm_images",
        "features": ["imageResNet18Vector"],
        "model": "svm_rbf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "svm_multimodal": {
        "name": "svm_multimodal",
        "features": ["sentence_embeddings", "imageResNet18Vector"],
        "model": "svm_rbf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "linear_svm_images": {
        "name": "linear_svm_images",
        "features": ["imageResNet18Vector"],
        "model": "linear_svm",
        "model_params": {"tune": True, "random_state": 41},
    },
    "linear_svm_multimodal": {
        "name": "linear_svm_multimodal",
        "features": ["sentence_embeddings", "imageResNet18Vector"],
        "model": "linear_svm",
        "model_params": {"tune": True, "random_state": 41},
    },
}



