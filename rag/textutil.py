"""Tiny tokenizer with light stemming so 'passwords' matches 'password'."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def _stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def tokenize(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ENGLISH_STOP_WORDS]
