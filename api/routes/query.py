"""
POST /query

Accepts a natural-language question, runs the full RAG chain
(retrieval + spatial enrichment + LLM generation), and returns
the answer together with the context passages that grounded it.
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.schemas import QueryRequest, QueryResponse
from tracking.mlflow_utils import log_query

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(request: Request, body: QueryRequest) -> QueryResponse:
    chain = request.app.state.chain
    logger.info("Received query: %s", body.question)

    t0 = time.perf_counter()
    result = chain.run(body.question, top_k=body.top_k, model_id=body.model_id)
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info("Query answered in %.0f ms", latency_ms)

    log_query(
        question=body.question,
        answer=result["answer"],
        top_k=body.top_k,
        latency_ms=round(latency_ms, 1),
        context_chars=len(result["context"]),
        model_id=result["model_id"],
    )

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        context=result["context"],
        latency_ms=round(latency_ms, 1),
    )


@router.post("/query/stream")
def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    chain = request.app.state.chain
    logger.info("Streaming query: %s", body.question)

    def generate():
        for event_type, data in chain.stream(body.question, top_k=body.top_k, model_id=body.model_id):
            yield f"data: {json.dumps({'type': event_type, 'value': data})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
