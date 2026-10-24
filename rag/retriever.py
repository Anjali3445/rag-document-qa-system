"""Retrievers: TF-IDF (default, no downloads) and sentence-embeddings (optional)."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from .textutil import tokenize


class TfidfRetriever:
    name = "tfidf"

    def __init__(self):
        self.vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                   ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = None

    def fit(self, texts):
        self.matrix = self.vec.fit_transform(texts)

    def scores(self, query):
        return linear_kernel(self.vec.transform([query]), self.matrix).ravel()


class EmbeddingRetriever:
    name = "embeddings"

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.model = SentenceTransformer(model_name)
        self.matrix = None

    def fit(self, texts):
        self.matrix = self.model.encode(texts, normalize_embeddings=True)

    def scores(self, query):
        q = self.model.encode([query], normalize_embeddings=True)
        return (self.matrix @ q.T).ravel()


def make_retriever(backend="tfidf"):
    if backend == "embeddings":
        try:
            return EmbeddingRetriever()
        except ImportError:
            print("[warn] sentence-transformers not installed -> falling back to TF-IDF")
    return TfidfRetriever()


def top_k(scores, k):
    idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]
