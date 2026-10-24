
from __future__ import annotations

import pandas as pd

# Star-rating word -> numeric score, as scraped from books.toscrape.com
RATING_MAP: dict[str, int] = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def add_rating_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rating_num"] = out["rating"].map(RATING_MAP)
    return out

