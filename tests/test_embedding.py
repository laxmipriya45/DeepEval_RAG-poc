"""Unit tests for embedding input validation and output shape."""

import pytest

from models.embedding_model import create_embedding


def test_embedding_rejects_empty_text():
    with pytest.raises(ValueError):
        create_embedding("")


def test_embedding_returns_non_empty_vector(monkeypatch):
    class FakeModel:
        def encode(self, text):
            return [0.1, 0.2, 0.3]

    import models.embedding_model as embedding_module
    monkeypatch.setattr(embedding_module, "_model", FakeModel())
    vector = create_embedding("Python")
    assert isinstance(vector, list)
    assert len(vector) == 3
