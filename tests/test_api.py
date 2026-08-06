"""
Integration tests for the FastAPI layer.

FastAPI's TestClient runs the full app in-process (no real server).
RAGChain is mocked so tests never need the FAISS index on disk.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

MOCK_RUN_RESULT = {
    "question": "What happened in Hatay?",
    "answer": "Severe damage was reported across Hatay province.",
    "context": "[1] Situation Report 1 (2023-02-10)\nThe earthquake caused severe damage in Hatay.",
}


@pytest.fixture
def mock_chain() -> MagicMock:
    chain = MagicMock()
    chain.model_id = "llama-3.1-8b-instant"
    chain.run.return_value = MOCK_RUN_RESULT
    chain.retriever.store.index.ntotal = 21197
    return chain


@pytest.fixture
def client(mock_chain: MagicMock) -> TestClient:
    """
    Patch RAGChain before the app starts so app.state.chain is our mock.
    The 'with' block triggers the lifespan (startup + shutdown).
    """
    with patch("api.main.RAGChain", return_value=mock_chain):
        from api.main import app
        with TestClient(app) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_shape(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["index_vectors"] == 21197
    assert data["model_id"] == "llama-3.1-8b-instant"
    assert "timestamp" in data


# ── /query ────────────────────────────────────────────────────────────────────

def test_query_returns_200(client):
    response = client.post("/query", json={"question": "What happened in Hatay?"})
    assert response.status_code == 200


def test_query_response_shape(client):
    data = client.post("/query", json={"question": "What happened in Hatay?"}).json()
    assert "question" in data
    assert "answer" in data
    assert "context" in data
    assert "latency_ms" in data


def test_query_empty_question_is_rejected(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_top_k_too_large_is_rejected(client):
    response = client.post("/query", json={"question": "test", "top_k": 100})
    assert response.status_code == 422


def test_query_top_k_zero_is_rejected(client):
    response = client.post("/query", json={"question": "test", "top_k": 0})
    assert response.status_code == 422


def test_query_passes_top_k_to_chain(client, mock_chain):
    client.post("/query", json={"question": "test question", "top_k": 3})
    mock_chain.run.assert_called_once_with("test question", top_k=3)


def test_query_default_top_k_is_5(client, mock_chain):
    client.post("/query", json={"question": "test question"})
    mock_chain.run.assert_called_once_with("test question", top_k=5)


def test_query_latency_is_positive(client):
    data = client.post("/query", json={"question": "test"}).json()
    assert data["latency_ms"] >= 0


# ── /ingest ───────────────────────────────────────────────────────────────────

def test_ingest_returns_200(client):
    from ingestion.pipeline import PipelineResult
    mock_result = PipelineResult(
        spatial_layers=["shakemap", "buildings"],
        chunks_written=500,
        index_vectors=500,
        duration_ms=1200.0,
    )
    with patch("api.routes.ingest.run_pipeline", return_value=mock_result), \
         patch("api.routes.ingest.RAGChain"):
        response = client.post("/ingest", json={})
    assert response.status_code == 200


def test_ingest_response_shape(client):
    from ingestion.pipeline import PipelineResult
    mock_result = PipelineResult(
        spatial_layers=["shakemap", "buildings"],
        chunks_written=500,
        index_vectors=500,
        duration_ms=1200.0,
    )
    with patch("api.routes.ingest.run_pipeline", return_value=mock_result), \
         patch("api.routes.ingest.RAGChain"):
        data = client.post("/ingest", json={}).json()
    assert data["status"] == "ok"
    assert data["chunks_written"] == 500
    assert data["spatial_layers"] == ["shakemap", "buildings"]
