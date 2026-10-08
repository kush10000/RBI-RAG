"""The ONLY place that knows which embedding model we use.

Swapping to Voyage (or anything else) means changing this file only.
index.py and search.py both import from here, which guarantees documents and
questions are always embedded by the SAME model.
"""
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384  # must match the vector(...) column size in Postgres

# bge was trained so that a short QUESTION and a long PASSAGE that answers it land close
# together. It works best when questions get this prefix and passages don't.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def _model_instance() -> SentenceTransformer:
    global _model
    if _model is None:  # load once (~2s), reuse afterwards
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_documents(texts: list[str]):
    """Many chunks -> array of shape (len(texts), DIM). Normalized, so dot product = cosine."""
    return _model_instance().encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True
    )


def embed_query(question: str):
    """One question -> one vector of shape (DIM,)."""
    return _model_instance().encode(QUERY_PREFIX + question, normalize_embeddings=True)
