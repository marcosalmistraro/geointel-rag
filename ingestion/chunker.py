"""
Splits text records into overlapping chunks ready for embedding.

Takes the output of text.py (list of dicts with 'text', 'source', 'metadata')
and returns a list of Chunk objects, each with a slice of the text and the
same metadata as the parent record so we never lose track of where it came from.

Run with:
    python -m ingestion.chunker
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Sentence boundary — split after . ! ? followed by whitespace
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


def chunk_record(
    record: dict[str, Any],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """
    Split a single text record into overlapping chunks.
    Respects sentence boundaries where possible.
    """
    text: str = record.get("text", "").strip()
    source: str = record.get("source", "")
    metadata: dict = record.get("metadata", {})

    if not text:
        return []

    sentences = _SENTENCE_END.split(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    chunk_idx = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        s_len = len(sentence)

        if current_len + s_len > chunk_size and current:
            chunk_text = " ".join(current)
            chunks.append(Chunk(chunk_text, source, chunk_idx, dict(metadata)))
            chunk_idx += 1

            # carry overlap from end of previous chunk
            overlap_text = chunk_text[-chunk_overlap:]
            current = [overlap_text]
            current_len = len(overlap_text)

        current.append(sentence)
        current_len += s_len + 1

    # flush remainder
    if current:
        chunks.append(Chunk(" ".join(current), source, chunk_idx, dict(metadata)))

    return chunks


def chunk_records(
    records: list[dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Chunk a list of records and return all chunks combined."""
    all_chunks: list[Chunk] = []
    for record in records:
        all_chunks.extend(chunk_record(record, chunk_size, chunk_overlap))
    logger.info(
        "Chunked %d records into %d chunks (size=%d, overlap=%d)",
        len(records), len(all_chunks), chunk_size, chunk_overlap,
    )
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick smoke test with synthetic data
    test_record = {
        "text": "This is the first sentence. " * 30 + "This is about Hatay province. " * 20,
        "source": "test",
        "metadata": {"report_id": "test_001", "doc_type": "test"},
    }
    chunks = chunk_record(test_record, chunk_size=200, chunk_overlap=40)
    print(f"Produced {len(chunks)} chunks")
    for c in chunks:
        print(f"  chunk {c.chunk_index}: {len(c.text)} chars — '{c.text[:60]}...'")