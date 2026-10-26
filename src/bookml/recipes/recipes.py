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
    # --- TF-IDF Direct Feature Experiments ---
    "xgb_tfidf": {
        "name": "xgb_tfidf",
        "features": ["price", "availability", "rating_numeric", "tfidf"],
        "model": "xgb",
        "model_params": {"tune": True, "random_state": 41},
    },
    "rf_tfidf": {
        "name": "rf_tfidf",
        "features": ["price", "availability", "rating_numeric", "tfidf"],
        "model": "rf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "lightgbm_tfidf": {
        "name": "lightgbm_tfidf",
        "features": ["price", "availability", "rating_numeric", "tfidf"],
        "model": "lightgbm",
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
    "xgb_domain": {
        "name": "xgb_domain",
        "features": ["price", "availability", "rating_numeric", "domain_features"],
        "model": "xgb",
        "model_params": {"tune": True, "random_state": 41},
    },
    "xgb_multimodal": {
        "name": "xgb_multimodal",
        "features": [
            "price",
            "availability",
            "rating_numeric",
            "domain_features",
            "tfidf_80",
            "image_color_features",
        ],
        "model": "xgb",
        "model_params": {"tune": True, "random_state": 41},
    },
    "xgb_multimodal_embeddings": {
        "name": "xgb_multimodal_embeddings",
        "features": [
            "price",
            "availability",
            "rating_numeric",
            "domain_features",
            "sentence_embeddings",
            "imageResNet18Vector",
        ],
        "model": "xgb",
        "model_params": {"tune": True, "random_state": 41},
    },

    # --- Random Forest Experiments ---
    "rf_tabular": {
        "name": "rf_tabular",
        "features": ["price", "availability", "rating_numeric"],
        "model": "rf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "rf_domain": {
        "name": "rf_domain",
        "features": ["price", "availability", "rating_numeric", "domain_features"],
        "model": "rf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "rf_embeddings": {
        "name": "rf_embeddings",
        "features": ["sentence_embeddings"],
        "model": "rf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "rf_multimodal": {
        "name": "rf_multimodal",
        "features": [
            "price",
            "availability",
            "rating_numeric",
            "domain_features",
            "tfidf_80",
            "image_color_features",
        ],
        "model": "rf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "rf_multimodal_embeddings": {
        "name": "rf_multimodal_embeddings",
        "features": [
            "price",
            "availability",
            "rating_numeric",
            "domain_features",
            "sentence_embeddings",
            "imageResNet18Vector",
        ],
        "model": "rf",
        "model_params": {"tune": True, "random_state": 41},
    },

    # --- LightGBM Experiments ---
    "lightgbm_tabular": {
        "name": "lightgbm_tabular",
        "features": ["price", "availability", "rating_numeric"],
        "model": "lightgbm",
        "model_params": {"tune": True, "random_state": 41},
    },
    "lightgbm_domain": {
        "name": "lightgbm_domain",
        "features": ["price", "availability", "rating_numeric", "domain_features"],
        "model": "lightgbm",
        "model_params": {"tune": True, "random_state": 41},
    },
    "lightgbm_embeddings": {
        "name": "lightgbm_embeddings",
        "features": ["sentence_embeddings"],
        "model": "lightgbm",
        "model_params": {"tune": True, "random_state": 41},
    },
    "lightgbm_multimodal": {
        "name": "lightgbm_multimodal",
        "features": [
            "price",
            "availability",
            "rating_numeric",
            "domain_features",
            "tfidf_80",
            "image_color_features",
        ],
        "model": "lightgbm",
        "model_params": {"tune": True, "random_state": 41},
    },
    "lightgbm_multimodal_embeddings": {
        "name": "lightgbm_multimodal_embeddings",
        "features": [
            "price",
            "availability",
            "rating_numeric",
            "domain_features",
            "sentence_embeddings",
            "imageResNet18Vector",
        ],
        "model": "lightgbm",
        "model_params": {"tune": True, "random_state": 41},
    },

    # --- Logistic Regression Experiments ---
    "logistic_regression_tabular": {
        "name": "logistic_regression_tabular",
        "features": ["price", "availability", "rating_numeric"],
        "model": "logistic_regression",
        "model_params": {"tune": True, "random_state": 41},
    },
    "logistic_regression_domain": {
        "name": "logistic_regression_domain",
        "features": ["price", "availability", "rating_numeric", "domain_features"],
        "model": "logistic_regression",
        "model_params": {"tune": True, "random_state": 41},
    },
    "logistic_regression_embeddings": {
        "name": "logistic_regression_embeddings",
        "features": ["sentence_embeddings"],
        "model": "logistic_regression",
        "model_params": {"tune": True, "random_state": 41},
    },
    "logistic_regression_multimodal": {
        "name": "logistic_regression_multimodal",
        "features": [
            "price",
            "availability",
            "rating_numeric",
            "domain_features",
            "tfidf_80",
            "image_color_features",
        ],
        "model": "logistic_regression",
        "model_params": {"tune": True, "random_state": 41},
    },
    "logistic_regression_multimodal_embeddings": {
        "name": "logistic_regression_multimodal_embeddings",
        "features": [
            "price",
            "availability",
            "rating_numeric",
            "domain_features",
            "sentence_embeddings",
            "imageResNet18Vector",
        ],
        "model": "logistic_regression",
        "model_params": {"tune": True, "random_state": 41},
    },

    # --- MLP (Neural Network) Experiments ---
    "mlp_tabular": {
        "name": "mlp_tabular",
        "features": ["price", "availability", "rating_numeric"],
        "model": "mlp",
        "model_params": {"tune": True, "random_state": 41},
    },
    "mlp_domain": {
        "name": "mlp_domain",
        "features": ["price", "availability", "rating_numeric", "domain_features"],
        "model": "mlp",
        "model_params": {"tune": True, "random_state": 41},
    },
    "mlp_embeddings": {
        "name": "mlp_embeddings",
        "features": ["sentence_embeddings"],
        "model": "mlp",
        "model_params": {"tune": True, "random_state": 41},
    },
    "mlp_multimodal": {
        "name": "mlp_multimodal",
        "features": [
            "price",
            "availability",
            "rating_numeric",
            "domain_features",
            "tfidf_80",
            "image_color_features",
        ],
        "model": "mlp",
        "model_params": {"tune": True, "random_state": 41},
    },
    "mlp_multimodal_embeddings": {
        "name": "mlp_multimodal_embeddings",
        "features": [
            "price",
            "availability",
            "rating_numeric",
            "domain_features",
            "sentence_embeddings",
            "imageResNet18Vector",
        ],
        "model": "mlp",
        "model_params": {"tune": True, "random_state": 41},
    },

    # --- Gradient Boosting Experiments ---
    "gb_tabular": {
        "name": "gb_tabular",
        "features": ["price", "availability", "rating_numeric"],
        "model": "gradient_boosting",
        "model_params": {"tune": True, "random_state": 41},
    },
    "gb_domain": {
        "name": "gb_domain",
        "features": ["price", "availability", "rating_numeric", "domain_features"],
        "model": "gradient_boosting",
        "model_params": {"tune": True, "random_state": 41},
    },
    "gb_embeddings": {
        "name": "gb_embeddings",
        "features": ["sentence_embeddings"],
        "model": "gradient_boosting",
        "model_params": {"tune": True, "random_state": 41},
    },

    # --- SVM Experiments ---
    "svm_tabular": {
        "name": "svm_tabular",
        "features": ["price", "availability", "rating_numeric"],
        "model": "svm_rbf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "svm_domain": {
        "name": "svm_domain",
        "features": ["price", "availability", "rating_numeric", "domain_features"],
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
    "linear_svm_images": {
        "name": "linear_svm_images",
        "features": ["imageResNet18Vector"],
        "model": "linear_svm",
        "model_params": {"tune": True, "random_state": 41},
    },
    "rf_images": {
        "name": "rf_images",
        "features": ["imageResNet18Vector"],
        "model": "rf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "logistic_regression_images": {
        "name": "logistic_regression_images",
        "features": ["imageResNet18Vector"],
        "model": "logistic_regression",
        "model_params": {"tune": True, "random_state": 41},
    },
    "mlp_images": {
        "name": "mlp_images",
        "features": ["imageResNet18Vector"],
        "model": "mlp",
        "model_params": {"tune": True, "random_state": 41},
    },
    "svm_basic_images": {
        "name": "svm_basic_images",
        "features": ["image_basic_features"],
        "model": "svm_rbf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "linear_svm_basic_images": {
        "name": "linear_svm_basic_images",
        "features": ["image_basic_features"],
        "model": "linear_svm",
        "model_params": {"tune": True, "random_state": 41},
    },
    "rf_basic_images": {
        "name": "rf_basic_images",
        "features": ["image_basic_features"],
        "model": "rf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "xgb_basic_images": {
        "name": "xgb_basic_images",
        "features": ["image_basic_features"],
        "model": "xgb",
        "model_params": {"tune": True, "random_state": 41},
    },
    "logistic_regression_basic_images": {
        "name": "logistic_regression_basic_images",
        "features": ["image_basic_features"],
        "model": "logistic_regression",
        "model_params": {"tune": True, "random_state": 41},
    },
    "mlp_basic_images": {
        "name": "mlp_basic_images",
        "features": ["image_basic_features"],
        "model": "mlp",
        "model_params": {"tune": True, "random_state": 41},
    },
    "svm_multimodal_concat": {
        "name": "svm_multimodal_concat",
        "features": ["multimodal_concatenated_feature"],
        "model": "svm_rbf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "linear_svm_multimodal_concat": {
        "name": "linear_svm_multimodal_concat",
        "features": ["multimodal_concatenated_feature"],
        "model": "linear_svm",
        "model_params": {"tune": True, "random_state": 41},
    },
    "rf_multimodal_concat": {
        "name": "rf_multimodal_concat",
        "features": ["multimodal_concatenated_feature"],
        "model": "rf",
        "model_params": {"tune": True, "random_state": 41},
    },
    "xgb_multimodal_concat": {
        "name": "xgb_multimodal_concat",
        "features": ["multimodal_concatenated_feature"],
        "model": "xgb",
        "model_params": {"tune": True, "random_state": 41},
    },
    "lightgbm_multimodal_concat": {
        "name": "lightgbm_multimodal_concat",
        "features": ["multimodal_concatenated_feature"],
        "model": "lightgbm",
        "model_params": {"tune": True, "random_state": 41},
    },
    "logistic_regression_multimodal_concat": {
        "name": "logistic_regression_multimodal_concat",
        "features": ["multimodal_concatenated_feature"],
        "model": "logistic_regression",
        "model_params": {"tune": True, "random_state": 41},
    },
    "mlp_multimodal_concat": {
        "name": "mlp_multimodal_concat",
        "features": ["multimodal_concatenated_feature"],
        "model": "mlp",
        "model_params": {"tune": True, "random_state": 41},
    },

    # --- Ensemble Experiments ---
    "ensemble_voting_domain": {
        "name": "ensemble_voting_domain",
        "features": ["price", "availability", "rating_numeric", "domain_features"],
        "model": "ensemble_voting",
        "model_params": {"tune": True, "random_state": 41},
    },
    "ensemble_voting_multimodal": {
        "name": "ensemble_voting_multimodal",
        "features": [
            "price",
            "availability",
            "rating_numeric",
            "domain_features",
            "tfidf_80",
            "image_color_features",
        ],
        "model": "ensemble_voting",
        "model_params": {"tune": True, "random_state": 41},
    },
    "ensemble_voting_multimodal_embeddings": {
        "name": "ensemble_voting_multimodal_embeddings",
        "features": [
            "price",
            "availability",
            "rating_numeric",
            "domain_features",
            "sentence_embeddings",
            "imageResNet18Vector",
        ],
        "model": "ensemble_voting",
        "model_params": {"tune": True, "random_state": 41},
    },
    "ensemble_stacking_domain": {
        "name": "ensemble_stacking_domain",
        "features": ["price", "availability", "rating_numeric", "domain_features"],
        "model": "ensemble_stacking",
        "model_params": {"tune": True, "random_state": 41},
    },
    "ensemble_stacking_multimodal": {
        "name": "ensemble_stacking_multimodal",
        "features": [
            "price",
            "availability",
            "rating_numeric",
            "domain_features",
            "tfidf_80",
            "image_color_features",
        ],
        "model": "ensemble_stacking",
        "model_params": {"tune": True, "random_state": 41},
    },
    "ensemble_stacking_multimodal_embeddings": {
        "name": "ensemble_stacking_multimodal_embeddings",
        "features": [
            "price",
            "availability",
            "rating_numeric",
            "domain_features",
            "sentence_embeddings",
            "imageResNet18Vector",
        ],
        "model": "ensemble_stacking",
        "model_params": {"tune": True, "random_state": 41},
    },
}

for _model in ("mlp", "rf", "xgb", "lightgbm", "logistic_regression", "ensemble_voting", "ensemble_stacking"):
    EXPERIMENTS[f"multimodal_{_model}"] = EXPERIMENTS[f"{_model}_multimodal"]
    EXPERIMENTS[f"{_model}_multimodal_notebook"] = EXPERIMENTS[f"{_model}_multimodal"]

# Aliases for Logistic Regression image & multimodal recipes
EXPERIMENTS["lr_images"] = EXPERIMENTS["logistic_regression_images"]
EXPERIMENTS["lr_basic_images"] = EXPERIMENTS["logistic_regression_basic_images"]
EXPERIMENTS["lr_multimodal_concat"] = EXPERIMENTS["logistic_regression_multimodal_concat"]


