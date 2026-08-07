"""
Pydantic schemas for all API request and response bodies.
FastAPI uses these to validate inputs and serialise outputs automatically.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    model_id: str | None = Field(default=None)


class QueryResponse(BaseModel):
    question: str
    answer: str
    context: str
    latency_ms: float


class IngestRequest(BaseModel):
    force_index: bool = Field(
        default=False,
        description="Rebuild the FAISS index even if it already exists on disk.",
    )


class IngestResponse(BaseModel):
    status: str
    chunks_written: int
    spatial_layers: list[str]
    index_vectors: int
    duration_ms: float


class HealthResponse(BaseModel):
    status: str
    index_vectors: int
    model_id: str
    timestamp: datetime
