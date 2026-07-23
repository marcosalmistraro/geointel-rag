"""
GeoIntel RAG — FastAPI application entry point.

Start with:
    uvicorn api.main:app --reload

Interactive docs auto-generated at:
    http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import health, ingest, query
from rag.chain import RAGChain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup (before serving requests) and once at shutdown.
    We load the RAGChain here so the FAISS index and embedding model are
    in memory for the lifetime of the server — not reloaded per request.
    """
    logger.info("Loading RAG chain...")
    app.state.chain = RAGChain()
    logger.info("RAG chain ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="GeoIntel RAG",
    description="Natural-language querying of the 2023 Turkey-Syria earthquake response corpus.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(query.router)
app.include_router(ingest.router)
