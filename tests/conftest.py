"""
Shared pytest fixtures used across multiple test files.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def sample_record() -> dict:
    """A realistic ReliefWeb report record for use in chunker/retriever tests."""
    return {
        "text": (
            "Earthquake damage in Hatay province was severe. "
            "Many buildings collapsed across the affected area. "
            "Search and rescue operations are ongoing. "
            "The humanitarian situation remains critical with thousands displaced."
        ),
        "source": "data/raw/reliefweb/eq-2023-000015-tur/12345.json",
        "metadata": {
            "report_id": "12345",
            "title": "Turkey Earthquake Situation Report #3",
            "date": "2023-02-10",
            "countries": ["Türkiye"],
            "organizations": ["OCHA"],
            "doc_type": "reliefweb_report",
        },
    }


@pytest.fixture
def mock_embedder() -> MagicMock:
    """
    A fake Embedder that returns deterministic normalised float32 vectors.
    Avoids loading the real SentenceTransformer model during tests.
    """
    embedder = MagicMock()
    embedder.embedding_dim = 384

    def _fake_embed(texts: list[str], **kwargs) -> np.ndarray:
        rng = np.random.default_rng(seed=42)
        vecs = rng.standard_normal((len(texts), 384)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    embedder.embed.side_effect = _fake_embed
    return embedder
