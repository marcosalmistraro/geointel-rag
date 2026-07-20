"""
Ingestion pipeline orchestrator.

Runs the full ingestion flow:
  1. Load spatial data (ShakeMap + buildings) → data/processed/spatial.pkl
  2. Load text reports (ReliefWeb) → chunk → data/processed/chunks.jsonl

Run with:
    python -m ingestion.pipeline
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

from ingestion.chunker import Chunk, chunk_records
from ingestion.loaders.spatial import load_all_spatial
from ingestion.loaders.text import load_all_reports

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


def run_pipeline() -> None:
    logger.info("=== GeoIntel RAG — ingestion pipeline ===")

    # 1. Spatial
    logger.info("--- Step 1/2: spatial data ---")
    spatial = load_all_spatial()
    save_spatial(spatial, PROCESSED_DIR / "spatial.pkl")

    # 2. Text
    logger.info("--- Step 2/2: text reports ---")
    records = load_all_reports()
    if records:
        chunks = chunk_records(records)
        save_chunks(chunks, PROCESSED_DIR / "chunks.jsonl")
    else:
        logger.warning("No text records found — chunks.jsonl not written")
        logger.warning("Re-run pipeline once ReliefWeb reports are downloaded")

    # Summary
    logger.info("=== Pipeline complete ===")
    logger.info("  Spatial layers: %s", list(spatial.keys()))
    logger.info("  Text chunks:    %d", len(chunks) if records else 0)


if __name__ == "__main__":
    run_pipeline()