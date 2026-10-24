# Developer Guide: Adding Features, Models, and Experiments

This guide explains how to extend BookML by adding new feature engineering functions, models, registering them, and creating experiment recipes.

---

## 1. Adding a Feature Function

Feature functions transform or extract columns from the dataset.

### Step 1: Define the function
Create or open a module under `src/bookml/features/` (e.g., `src/bookml/features/tabular.py` or `src/bookml/features/text.py`).

Your function must take a `pd.DataFrame` and return a `pd.DataFrame` containing the new column(s):

```python
# src/bookml/features/tabular.py
import pandas as pd

def add_title_word_count(df: pd.DataFrame) -> pd.DataFrame:
    """Computes word count of the book title."""
    out = df.copy()
    out["title_word_count"] = out["title"].str.split().str.len()
    return out
```

> **Note**: Only newly added columns will be picked up by the pipeline and added to the feature matrix.

### Step 2: Register in Feature Registry
Open `src/bookml/registry.py`, import your function, and add it to `FEATURE_REGISTRY`:

```python
# src/bookml/registry.py
from bookml.features.tabular import add_title_word_count

FEATURE_REGISTRY: dict[str, FeatureEntry] = {
    # Existing features...
    "title_word_count": add_title_word_count,
}
```
*(If you just want to use a raw column directly from the CSV without transformation, you can register it as `"col_key": "raw_column_name"`)*.

---

## 2. Adding a Model

Models must implement `fit(X, y)`, `predict(X)`, and optionally `predict_proba(X)`.

### Step 1: Define the model class
Create a new file in `src/bookml/models/` (e.g., `src/bookml/models/rf_classifier.py`).

You can inherit from `GridSearchClassifier` in `src/bookml/models/base.py` to get automated hyperparameter tuning and cross-validation out of the box:

```python
# src/bookml/models/rf_classifier.py
from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import GridSearchClassifier

class RandomForestModel(GridSearchClassifier):
    def _build_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(random_state=self.random_state)

    def _default_param_grid(self, y: np.ndarray) -> dict[str, list[Any]]:
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 8],
        }
```

### Step 2: Register in Model Registry
Open `src/bookml/registry.py`, import the model class (or factory function), and add it to `MODEL_REGISTRY`:

```python
# src/bookml/registry.py
from bookml.models.rf_classifier import RandomForestModel

MODEL_REGISTRY: dict[str, ModelFactory] = {
    # Existing models...
    "random_forest": RandomForestModel,
}
```

---

## 3. Adding an Experiment Recipe

Experiment recipes define which features, model, and parameters run together.

Open `src/bookml/recipes/recipes.py` and add a new configuration dictionary to `EXPERIMENTS`:

```python
# src/bookml/recipes/recipes.py
EXPERIMENTS: dict[str, dict[str, Any]] = {
    # ...
    "rf_tabular": {
        "name": "rf_tabular",
        "features": [
            "price",
            "availability",
            "rating_numeric",
            "title_word_count",   # Registered feature key
        ],
        "model": "random_forest", # Registered model key
        "model_params": {
            "tune": True,
            "random_state": 41,
        },
    },
}
```

### Configuration fields:
- **`name`**: Unique identifier for the experiment (used for results filename).
- **`features`**: List of keys matching `FEATURE_REGISTRY`.
- **`model`**: Model key matching `MODEL_REGISTRY`.
- **`model_params`**: Optional dictionary of kwargs passed to the model's constructor.

---

## 4. Running the Experiment

Run your experiment from the CLI using:

```bash
python .\src\bookml\main.py --experiment rf_tabular
```

### What happens next:
1. Data is loaded and split according to `validation_set.txt`.
2. Selected features are generated and passed to the model.
3. Model is trained / tuned.
4. Metrics and confusion matrix are printed to console.
5. Detailed JSON output is saved to `src/bookml/results/<experiment_name>.json`.
6. Summary row is appended to `src/bookml/results/overall.csv`.
