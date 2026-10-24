
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Data
DATA_DIR: Path = PROJECT_ROOT / "data"
BOOKS_CSV: Path = DATA_DIR / "books.csv"
VAL_IDS_TXT: Path = DATA_DIR / "validation_set.txt"
IMAGES_DIR: Path = DATA_DIR / "images"


# Results
RESULTS_DIR: Path = PROJECT_ROOT / "results"
OVERALL_CSV: Path = PROJECT_ROOT / "results" / "overall.csv"
