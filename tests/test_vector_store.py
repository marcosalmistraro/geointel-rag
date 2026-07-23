"""
Tests for rag/vector_store.py.

Uses the mock_embedder fixture (from conftest.py) so FAISS gets real
vectors to index — without loading the actual sentence-transformers model.
"""

from __future__ import annotations

import pytest

from rag.vector_store import VectorStore

SAMPLE_CHUNKS = [
    {"text": "Earthquake damage in Hatay province was severe.", "source": "a.json", "metadata": {"report_id": "1"}},
    {"text": "Food security crisis in Gaziantep needs urgent attention.", "source": "b.json", "metadata": {"report_id": "2"}},
    {"text": "Search and rescue operations ongoing in Kahramanmaras.", "source": "c.json", "metadata": {"report_id": "3"}},
    {"text": "Water supply disrupted across the affected provinces.", "source": "d.json", "metadata": {"report_id": "4"}},
    {"text": "MMI intensity 8.5 recorded in southern Turkey.", "source": "e.json", "metadata": {"report_id": "5"}},
]


def test_build_creates_index(mock_embedder):
    store = VectorStore(mock_embedder)
    store.build(SAMPLE_CHUNKS)
    assert store.index is not None
    assert store.index.ntotal == len(SAMPLE_CHUNKS)


def test_build_stores_chunks(mock_embedder):
    store = VectorStore(mock_embedder)
    store.build(SAMPLE_CHUNKS)
    assert len(store.chunks) == len(SAMPLE_CHUNKS)


def test_build_empty_raises(mock_embedder):
    store = VectorStore(mock_embedder)
    with pytest.raises(ValueError):
        store.build([])


def test_search_before_build_raises(mock_embedder):
    store = VectorStore(mock_embedder)
    with pytest.raises(RuntimeError):
        store.search("test query")


def test_search_returns_top_k(mock_embedder):
    store = VectorStore(mock_embedder)
    store.build(SAMPLE_CHUNKS)
    results = store.search("earthquake", top_k=3)
    assert len(results) == 3


def test_search_results_have_score(mock_embedder):
    store = VectorStore(mock_embedder)
    store.build(SAMPLE_CHUNKS)
    results = store.search("earthquake", top_k=2)
    assert all("score" in r for r in results)


def test_search_results_have_text(mock_embedder):
    store = VectorStore(mock_embedder)
    store.build(SAMPLE_CHUNKS)
    results = store.search("earthquake", top_k=2)
    assert all("text" in r for r in results)


def test_save_and_load_round_trip(mock_embedder, tmp_path):
    store = VectorStore(mock_embedder)
    store.build(SAMPLE_CHUNKS)

    index_path = tmp_path / "index.bin"
    chunks_path = tmp_path / "chunks.jsonl"
    store.save(index_path, chunks_path)

    assert index_path.exists()
    assert chunks_path.exists()

    store2 = VectorStore(mock_embedder)
    store2.load(index_path, chunks_path)

    assert store2.index.ntotal == len(SAMPLE_CHUNKS)
    assert len(store2.chunks) == len(SAMPLE_CHUNKS)


def test_load_missing_index_raises(mock_embedder, tmp_path):
    store = VectorStore(mock_embedder)
    with pytest.raises(FileNotFoundError):
        store.load(tmp_path / "nonexistent.bin", tmp_path / "nonexistent.jsonl")
