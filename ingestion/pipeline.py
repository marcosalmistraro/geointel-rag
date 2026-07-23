"""
Ingestion pipeline orchestrator.

Runs the full ingestion flow:
  1. Load spatial data (ShakeMap + buildings) → data/processed/spatial.pkl
  2. Load text reports (ReliefWeb) → chunk → data/processed/chunks.jsonl
  3. Embed chunks → build FAISS index → data/processed/faiss_index.bin

Run with:
    python -m ingestion.pipeline
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.chunker import Chunk, chunk_records
from ingestion.loaders.spatial import load_all_spatial
from ingestion.loaders.text import load_all_reports


@dataclass
class PipelineResult:
    spatial_layers: list[str] = field(default_factory=list)
    chunks_written: int = 0
    index_vectors: int = 0
    duration_ms: float = 0.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


def save_spatial(spatial: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(spatial, f)
    logger.info("Saved spatial index to %s", path)


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Saved %d chunks to %s", len(chunks), path)


def _build_faiss_index(chunks: list[Chunk]) -> int:
    """Embed chunks and write the FAISS index. Returns vector count."""
    from rag.embedder import Embedder
    from rag.vector_store import VectorStore

    store = VectorStore(Embedder())
    chunk_dicts = [c.to_dict() for c in chunks]
    store.build(chunk_dicts)
    store.save(
        index_path=PROCESSED_DIR / "faiss_index.bin",
        chunks_path=PROCESSED_DIR / "chunks.jsonl",
    )
    return store.index.ntotal


def run_pipeline(rebuild_index: bool = True, force_index: bool = False) -> PipelineResult:
    logger.info("=== GeoIntel RAG — ingestion pipeline ===")
    t0 = time.perf_counter()
    result = PipelineResult()

    # 1. Spatial
    logger.info("--- Step 1/3: spatial data ---")
    spatial = load_all_spatial()
    save_spatial(spatial, PROCESSED_DIR / "spatial.pkl")
    result.spatial_layers = list(spatial.keys())

    # 2. Text → chunks
    logger.info("--- Step 2/3: text reports ---")
    records = load_all_reports()
    chunks: list[Chunk] = []
    if records:
        chunks = chunk_records(records)
        result.chunks_written = len(chunks)
        # chunks.jsonl is written by _build_faiss_index via store.save,
        # so only write it separately when skipping index build
        if not rebuild_index:
            save_chunks(chunks, PROCESSED_DIR / "chunks.jsonl")
    else:
        logger.warning("No text records found — chunks.jsonl not written")
        logger.warning("Re-run pipeline once ReliefWeb reports are downloaded")

    # 3. FAISS index
    index_path = PROCESSED_DIR / "faiss_index.bin"
    index_exists = index_path.exists()
    if rebuild_index and chunks and (force_index or not index_exists):
        logger.info("--- Step 3/3: building FAISS index ---")
        result.index_vectors = _build_faiss_index(chunks)
    elif index_exists:
        import faiss
        result.index_vectors = faiss.read_index(str(index_path)).ntotal
        logger.info("--- Step 3/3: FAISS index already exists (%d vectors) — skipping ---", result.index_vectors)
    else:
        logger.info("--- Step 3/3: skipping index build (rebuild_index=False or no chunks) ---")

    result.duration_ms = (time.perf_counter() - t0) * 1000
    logger.info("=== Pipeline complete in %.0f ms ===", result.duration_ms)
    logger.info("  Spatial layers: %s", result.spatial_layers)
    logger.info("  Text chunks:    %d", result.chunks_written)
    logger.info("  Index vectors:  %d", result.index_vectors)
    return result


if __name__ == "__main__":
    run_pipeline()