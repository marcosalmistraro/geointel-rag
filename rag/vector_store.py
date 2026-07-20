"""
Builds and queries a FAISS vector index over text chunks.

Workflow:
    1. Build: embed chunks → build FAISS index → save to disk
    2. Load:  load index + chunks from disk at query time
    3. Search: embed query → find top-k nearest chunks

Run with:
    python -m rag.vector_store
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from rag.embedder import Embedder

logger = logging.getLogger(__name__)

INDEX_PATH  = Path("data/processed/faiss_index.bin")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")


class VectorStore:
    """
    Wraps a FAISS flat index with the chunk texts it was built from.
    'Flat' means exact search — no approximation, correct for our size.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()
        self.index: faiss.IndexFlatIP | None = None  # IP = inner product = cosine on normalized vectors
        self.chunks: list[dict] = []  # parallel list to the index rows

    def build(self, chunks: list[dict]) -> None:
        """
        Embed all chunks and build the FAISS index.
        chunks: list of dicts with at least a 'text' key.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        texts = [c["text"] for c in chunks]
        logger.info("Embedding %d chunks ...", len(texts))
        vectors = self.embedder.embed(texts, show_progress=True)

        # Build flat inner product index
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.chunks = chunks

        logger.info("FAISS index built — %d vectors, dim=%d", self.index.ntotal, dim)

    def save(
        self,
        index_path: Path = INDEX_PATH,
        chunks_path: Path = CHUNKS_PATH,
    ) -> None:
        """Save the FAISS index and chunks to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        logger.info("Saved FAISS index to %s", index_path)

        with chunks_path.open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        logger.info("Saved %d chunks to %s", len(self.chunks), chunks_path)

    def load(
        self,
        index_path: Path = INDEX_PATH,
        chunks_path: Path = CHUNKS_PATH,
    ) -> None:
        """Load a previously saved FAISS index and chunks from disk."""
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        self.index = faiss.read_index(str(index_path))
        logger.info("Loaded FAISS index — %d vectors", self.index.ntotal)

        self.chunks = []
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))
        logger.info("Loaded %d chunks", len(self.chunks))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Embed a query string and return the top-k most similar chunks.
        Each returned dict has 'text', 'source', 'metadata', and 'score'.
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded yet")

        query_vector = self.embedder.embed([query])
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Smoke test with synthetic chunks
    test_chunks = [
        {"text": "Earthquake damage in Hatay province was severe.", "source": "test", "metadata": {}},
        {"text": "Food security situation deteriorating in Gaziantep.", "source": "test", "metadata": {}},
        {"text": "Search and rescue operations ongoing in Kahramanmaras.", "source": "test", "metadata": {}},
        {"text": "Water supply disrupted across affected provinces.", "source": "test", "metadata": {}},
        {"text": "MMI intensity 8.5 recorded in southern Turkey.", "source": "test", "metadata": {}},
    ]

    store = VectorStore()
    store.build(test_chunks)
    store.save()

    # Test search
    results = store.search("what is the situation in Hatay?", top_k=3)
    print(f"\nTop 3 results for 'what is the situation in Hatay?'")
    for r in results:
        print(f"  score={r['score']:.4f} — {r['text']}")