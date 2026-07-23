"""
POST /ingest

Runs the full ingestion pipeline (spatial load → text chunking → FAISS index build)
and then hot-swaps the running chain so new queries immediately use the fresh index.
No server restart needed.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from api.schemas import IngestRequest, IngestResponse
from ingestion.pipeline import run_pipeline
from rag.chain import RAGChain

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: Request, body: IngestRequest) -> IngestResponse:
    logger.info("Ingest triggered (force_index=%s)", body.force_index)

    try:
        result = run_pipeline(rebuild_index=True, force_index=body.force_index)
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Hot-swap: replace the chain so the new index is live immediately
    request.app.state.chain = RAGChain()
    logger.info("Chain reloaded with fresh index (%d vectors)", result.index_vectors)

    return IngestResponse(
        status="ok",
        chunks_written=result.chunks_written,
        spatial_layers=result.spatial_layers,
        index_vectors=result.index_vectors,
        duration_ms=round(result.duration_ms, 1),
    )
