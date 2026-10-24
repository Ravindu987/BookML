
from __future__ import annotations

import re
from functools import lru_cache

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_NLTK_RESOURCES: tuple[str, ...] = ("stopwords", "wordnet", "omw-1.4")


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
