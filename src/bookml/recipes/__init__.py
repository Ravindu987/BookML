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
            "price",             # direct from CSV
            "availability",      # direct from CSV
            "rating_numeric",    # function: rating -> rating_num
            "clean_description", # function: description -> cleaned_description
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
}