"""
Unit tests for rag/retriever.py.

We bypass Retriever.__init__ (which loads FAISS + spatial from disk)
and test each method in isolation using __new__ + manually set attributes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag.retriever import Retriever

MOCK_CHUNKS = [
    {
        "text": "The earthquake caused severe damage in Hatay province.",
        "source": "a.json",
        "metadata": {"title": "Situation Report 1", "date": "2023-02-10"},
        "score": 0.92,
    },
    {
        "text": "Search and rescue operations ongoing in Kahramanmaras.",
        "source": "b.json",
        "metadata": {"title": "Situation Report 2", "date": "2023-02-11"},
        "score": 0.85,
    },
]


def _make_retriever(chunks=MOCK_CHUNKS, spatial=None) -> Retriever:
    """Build a Retriever with mocked store and no disk access."""
    mock_store = MagicMock()
    mock_store.index = MagicMock()
    mock_store.search.return_value = chunks

    r = Retriever.__new__(Retriever)
    r.store = mock_store
    r.top_k = 5
    r.spatial = spatial
    return r


# ── province detection ────────────────────────────────────────────────────────

def test_detects_known_province():
    r = _make_retriever()
    provinces = r._detect_provinces("The situation in Hatay is critical.")
    assert "hatay" in provinces


def test_detects_multiple_provinces():
    r = _make_retriever()
    provinces = r._detect_provinces("Hatay and Gaziantep were the worst affected.")
    assert "hatay" in provinces
    assert "gaziantep" in provinces


def test_does_not_detect_absent_province():
    r = _make_retriever()
    provinces = r._detect_provinces("The weather was mild today.")
    assert provinces == []


def test_detection_is_case_insensitive():
    r = _make_retriever()
    assert "hatay" in r._detect_provinces("HATAY province suffered damage.")
    assert "hatay" in r._detect_provinces("Hatay province suffered damage.")


# ── retrieve ─────────────────────────────────────────────────────────────────

def test_retrieve_returns_string():
    r = _make_retriever()
    result = r.retrieve("What happened in Hatay?")
    assert isinstance(result, str)


def test_retrieve_includes_report_title():
    r = _make_retriever()
    context = r.retrieve("What happened in Hatay?")
    assert "Situation Report 1" in context


def test_retrieve_includes_chunk_text():
    r = _make_retriever()
    context = r.retrieve("earthquake damage")
    assert "severe damage in Hatay" in context


def test_retrieve_no_chunks_returns_fallback():
    r = _make_retriever(chunks=[])
    context = r.retrieve("some query")
    assert context == "No relevant documents found."


def test_retrieve_passes_top_k_override():
    r = _make_retriever()
    r.retrieve("test query", top_k=3)
    r.store.search.assert_called_once_with("test query", top_k=3)


def test_retrieve_uses_instance_top_k_when_not_overridden():
    r = _make_retriever()
    r.retrieve("test query")
    r.store.search.assert_called_once_with("test query", top_k=5)


def test_retrieve_no_spatial_context_when_spatial_is_none():
    r = _make_retriever(spatial=None)
    context = r.retrieve("Hatay province damage")
    assert "Geospatial context" not in context
