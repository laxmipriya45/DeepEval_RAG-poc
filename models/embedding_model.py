"""Embedding model with lazy loading to avoid work during module import."""

from sentence_transformers import SentenceTransformer

from config.model_config import EMBEDDING_MODEL

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def create_embedding(text: str) -> list[float]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    return get_embedding_model().encode(text).tolist()
