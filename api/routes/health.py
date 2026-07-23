"""
GET /health

Returns the server status, how many vectors are loaded in the FAISS index,
and which LLM model the chain is configured to use.
Useful for smoke-testing the deployment without spending tokens.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request

from api.schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    chain = request.app.state.chain
    index = chain.retriever.store.index
    index_vectors = index.ntotal if index is not None else 0

    return HealthResponse(
        status="ok",
        index_vectors=index_vectors,
        model_id=chain.model_id,
        timestamp=datetime.now(timezone.utc),
    )
