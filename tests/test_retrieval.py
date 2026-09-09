"""Unit tests for retriever validation without requiring ChromaDB."""

import pytest

from retrieval.retriever import retrieve_context


def test_retriever_rejects_empty_question():
    with pytest.raises(ValueError):
        retrieve_context("")


def test_retriever_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        retrieve_context("What is Python?", top_k=0)
