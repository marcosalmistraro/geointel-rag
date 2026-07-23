"""
Unit tests for ingestion/chunker.py.

The chunker is pure Python string processing — no external dependencies,
no mocking required.
"""

from __future__ import annotations

import pytest

from ingestion.chunker import Chunk, chunk_record, chunk_records


def test_empty_text_returns_no_chunks():
    record = {"text": "", "source": "test", "metadata": {}}
    assert chunk_record(record) == []


def test_missing_text_returns_no_chunks():
    assert chunk_record({"source": "test", "metadata": {}}) == []


def test_short_text_produces_one_chunk(sample_record):
    chunks = chunk_record(sample_record, chunk_size=2000)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_long_text_produces_multiple_chunks():
    long_text = "This is a sentence about earthquake response operations. " * 60
    record = {"text": long_text, "source": "test", "metadata": {"report_id": "1"}}
    chunks = chunk_record(record, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1


def test_chunk_indices_are_sequential():
    long_text = "Short sentence here. " * 80
    record = {"text": long_text, "source": "test", "metadata": {}}
    chunks = chunk_record(record, chunk_size=100, chunk_overlap=20)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_metadata_preserved_in_every_chunk(sample_record):
    chunks = chunk_record(sample_record, chunk_size=100)
    for chunk in chunks:
        assert chunk.metadata == sample_record["metadata"]
        assert chunk.source == sample_record["source"]


def test_source_preserved(sample_record):
    chunks = chunk_record(sample_record)
    assert all(c.source == sample_record["source"] for c in chunks)


def test_chunk_text_is_non_empty(sample_record):
    chunks = chunk_record(sample_record)
    assert all(len(c.text.strip()) > 0 for c in chunks)


def test_to_dict_has_required_keys(sample_record):
    chunk = chunk_record(sample_record)[0]
    d = chunk.to_dict()
    assert set(d.keys()) == {"text", "source", "chunk_index", "metadata"}


def test_chunk_records_aggregates_multiple():
    records = [
        {"text": "First report about Hatay damage.", "source": "a.json", "metadata": {}},
        {"text": "Second report about Gaziantep operations.", "source": "b.json", "metadata": {}},
    ]
    chunks = chunk_records(records)
    assert len(chunks) >= 2
    sources = {c.source for c in chunks}
    assert "a.json" in sources
    assert "b.json" in sources


def test_chunk_records_skips_empty():
    records = [
        {"text": "", "source": "empty.json", "metadata": {}},
        {"text": "Valid text about earthquake response.", "source": "valid.json", "metadata": {}},
    ]
    chunks = chunk_records(records)
    assert all(c.source == "valid.json" for c in chunks)
