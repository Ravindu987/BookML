from __future__ import annotations

import re
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


# =============================================================================
# Notebook Feature Engineering (Domain, Syntactic, Length & Stylistic Signals)
# =============================================================================

def _get_text_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    title = df["title"].fillna("").astype(str)
    desc = df["description"].fillna("").astype(str) if "description" in df.columns else pd.Series([""] * len(df), index=df.index)
    full_text = title + " " + desc
    return title, desc, full_text


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes engineered text length, syntactic, and stylistic features."""
    out = df.copy()
    title, desc, full_text = _get_text_series(out)

    out["title_char_len"] = title.str.len()
    out["desc_char_len"] = desc.str.len()
    out["total_char_len"] = full_text.str.len()

    out["title_word_count"] = title.apply(lambda s: len(s.split()))
    out["desc_word_count"] = desc.apply(lambda s: len(s.split()))
    out["total_word_count"] = full_text.apply(lambda s: len(s.split()))

    out["has_subtitle"] = title.str.contains(r":|—| - ", regex=True).astype(int)
    out["title_upper_ratio"] = title.apply(
        lambda s: sum(1 for c in s if c.isupper()) / max(len(s), 1)
    )
    out["quote_count"] = full_text.apply(
        lambda s: s.count('"') + s.count("'") + s.count("“") + s.count("”")
    )
    out["digit_count"] = full_text.apply(lambda s: sum(c.isdigit() for c in s))
    out["title_digit_count"] = title.apply(lambda s: sum(c.isdigit() for c in s))
    out["exclamation_count"] = full_text.apply(lambda s: s.count("!"))

    return out


# Individual helper functions for selective registration
def add_title_char_len(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["title_char_len"] = out["title"].fillna("").astype(str).str.len()
    return out


def add_desc_char_len(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    desc = out["description"].fillna("").astype(str) if "description" in out.columns else pd.Series([""] * len(out), index=out.index)
    out["desc_char_len"] = desc.str.len()
    return out


def add_total_char_len(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _, _, full_text = _get_text_series(out)
    out["total_char_len"] = full_text.str.len()
    return out


def add_title_word_count(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["title_word_count"] = out["title"].fillna("").astype(str).apply(lambda s: len(s.split()))
    return out


def add_desc_word_count(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    desc = out["description"].fillna("").astype(str) if "description" in out.columns else pd.Series([""] * len(out), index=out.index)
    out["desc_word_count"] = desc.apply(lambda s: len(s.split()))
    return out


def add_total_word_count(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _, _, full_text = _get_text_series(out)
    out["total_word_count"] = full_text.apply(lambda s: len(s.split()))
    return out


def add_has_subtitle(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["has_subtitle"] = out["title"].fillna("").astype(str).str.contains(r":|—| - ", regex=True).astype(int)
    return out


def add_title_upper_ratio(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    title = out["title"].fillna("").astype(str)
    out["title_upper_ratio"] = title.apply(
        lambda s: sum(1 for c in s if c.isupper()) / max(len(s), 1)
    )
    return out


def add_quote_count(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _, _, full_text = _get_text_series(out)
    out["quote_count"] = full_text.apply(
        lambda s: s.count('"') + s.count("'") + s.count("“") + s.count("”")
    )
    return out


def add_digit_count(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _, _, full_text = _get_text_series(out)
    out["digit_count"] = full_text.apply(lambda s: sum(c.isdigit() for c in s))
    return out


def add_title_digit_count(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    title = out["title"].fillna("").astype(str)
    out["title_digit_count"] = title.apply(lambda s: sum(c.isdigit() for c in s))
    return out


def add_exclamation_count(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _, _, full_text = _get_text_series(out)
    out["exclamation_count"] = full_text.apply(lambda s: s.count("!"))
    return out
