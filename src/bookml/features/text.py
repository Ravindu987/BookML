
from __future__ import annotations

import re
from functools import lru_cache

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

_NLTK_RESOURCES: tuple[str, ...] = ("stopwords", "wordnet", "omw-1.4")

TFIDF_PREFIX = "tfidf_"


@lru_cache(maxsize=1)
def _get_nlp_resources() -> tuple[WordNetLemmatizer, frozenset[str]]:
    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    return WordNetLemmatizer(), frozenset(stopwords.words("english"))


def clean_and_lemmatize(text: str) -> str:
    lemmatizer, stop_words = _get_nlp_resources()

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())

    cleaned_tokens = [
        lemmatizer.lemmatize(lemmatizer.lemmatize(token, pos="v"), pos="n")
        for token in text.split()
        if token not in stop_words and len(token) > 2
    ]
    return " ".join(cleaned_tokens)


def add_cleaned_description(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cleaned_description"] = out["description"].apply(clean_and_lemmatize)
    return out


def add_tfidf_features(
    df: pd.DataFrame,
    max_features: int = 1000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> pd.DataFrame:
    """Compute TF-IDF features from description text and append them as numeric columns.

    If 'cleaned_description' is already in the dataframe, it will be used directly.
    Otherwise, raw 'description' is cleaned and lemmatized first.
    """
    if "cleaned_description" in df.columns:
        texts = df["cleaned_description"]
    elif "description" in df.columns:
        texts = df["description"].apply(clean_and_lemmatize)
    else:
        raise ValueError("DataFrame must contain 'cleaned_description' or 'description' column.")

    texts = texts.fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()
    feature_names = vectorizer.get_feature_names_out()

    col_names: list[str] = []
    seen: set[str] = set()
    for i, name in enumerate(feature_names):
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        col = f"{TFIDF_PREFIX}{sanitized}"
        if col in seen:
            col = f"{TFIDF_PREFIX}{i}_{sanitized}"
        seen.add(col)
        col_names.append(col)

    tfidf_df = pd.DataFrame(
        np.ascontiguousarray(tfidf_matrix, dtype=np.float32),
        index=df.index,
        columns=col_names,
    )
    return pd.concat([df, tfidf_df], axis=1)


def add_tfidf_80_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts 80 top TF-IDF features from title and description."""
    title = df["title"].fillna("").astype(str)
    desc = df["description"].fillna("").astype(str) if "description" in df.columns else pd.Series([""] * len(df), index=df.index)
    full_text = title + " " + desc

    vectorizer = TfidfVectorizer(max_features=80, stop_words="english")
    matrix = vectorizer.fit_transform(full_text).toarray()
    cols = [f"{TFIDF_PREFIX}{i}" for i in range(matrix.shape[1])]
    tfidf_df = pd.DataFrame(
        np.ascontiguousarray(matrix, dtype=np.float32),
        index=df.index,
        columns=cols,
    )
    return pd.concat([df, tfidf_df], axis=1)


