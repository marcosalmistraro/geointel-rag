"""
POST /query

Accepts a natural-language question, runs the full RAG chain
(retrieval + spatial enrichment + LLM generation), and returns
the answer together with the context passages that grounded it.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request

from api.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(request: Request, body: QueryRequest) -> QueryResponse:
    chain = request.app.state.chain
    logger.info("Received query: %s", body.question)

    t0 = time.perf_counter()
    result = chain.run(body.question, top_k=body.top_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info("Query answered in %.0f ms", latency_ms)

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        context=result["context"],
        latency_ms=round(latency_ms, 1),
    )
