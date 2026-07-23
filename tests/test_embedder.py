"""
Unit tests for rag/embedder.py.

SentenceTransformer is mocked so the real model is never downloaded
or loaded during the test run — keeps tests fast and offline-safe.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_embedder(encode_output: np.ndarray, dim: int = 384):
    """Helper: patch SentenceTransformer and return an Embedder instance."""
    with patch("rag.embedder.SentenceTransformer") as MockST:
        mock_model = MockST.return_value
        mock_model.encode.return_value = encode_output
        mock_model.get_embedding_dimension.return_value = dim
        from rag.embedder import Embedder
        return Embedder(), mock_model


def test_embed_raises_on_empty_list():
    raw = np.ones((1, 384), dtype=np.float32)
    embedder, _ = _make_embedder(raw)
    with pytest.raises(ValueError, match="empty"):
        embedder.embed([])


def test_embed_returns_float32():
    # encode returns float64 — embedder must cast to float32
    raw = np.ones((2, 384), dtype=np.float64)
    embedder, _ = _make_embedder(raw)
    result = embedder.embed(["hello", "world"])
    assert result.dtype == np.float32


def test_embed_returns_correct_shape():
    raw = np.ones((3, 384), dtype=np.float32)
    embedder, _ = _make_embedder(raw)
    result = embedder.embed(["a", "b", "c"])
    assert result.shape == (3, 384)


def test_embedding_dim_property():
    raw = np.ones((1, 384), dtype=np.float32)
    embedder, _ = _make_embedder(raw, dim=384)
    assert embedder.embedding_dim == 384


def test_embed_calls_encode_with_texts():
    raw = np.ones((2, 384), dtype=np.float32)
    embedder, mock_model = _make_embedder(raw)
    texts = ["first sentence", "second sentence"]
    embedder.embed(texts)
    call_args = mock_model.encode.call_args
    assert call_args[0][0] == texts
