
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
ID_COLUMN = "id"
TARGET_COLUMN = "target"

# Columns that encode the label. They must NEVER be passed to a model as input,
# otherwise the model can trivially "cheat". The runner drops these from X.
LABEL_COLUMNS: tuple[str, ...] = ("category", TARGET_COLUMN)

# Target category mapping: Fiction -> 1 (positive class), Nonfiction -> 0
CATEGORY_MAP: dict[str, int] = {"Nonfiction": 0, "Fiction": 1}
INVERSE_CATEGORY_MAP: dict[int, str] = {v: k for k, v in CATEGORY_MAP.items()}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df[ID_COLUMN] = df.index
    df[TARGET_COLUMN] = df["category"].map(CATEGORY_MAP)
    logger.info("Loaded %d records from '%s'.", len(df), Path(csv_path).name)
    return df



# ---------------------------------------------------------------------------
# Train/validation split
# ---------------------------------------------------------------------------
def get_train_val_split(
    df: pd.DataFrame,
    val_file: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(val_file, "r", encoding="utf-8") as f:
        val_ids = {
            int(line.split("\t", 1)[0])
            for line in (raw.strip() for raw in f)
            if line and not line.startswith("#")
        }

    is_val = df[ID_COLUMN].isin(val_ids)
    logger.info("Using %d fixed validation IDs from '%s'.", int(is_val.sum()), Path(val_file).name)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
