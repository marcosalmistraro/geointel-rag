"""
MLflow logging helpers for GeoIntel RAG.

Two functions:
  log_query()     — called after every /query request
  log_ingestion() — called after every /ingest request

Each call opens a new MLflow run, writes params + metrics, then closes it.
Run `mlflow ui` to browse all logged runs in the browser.
"""

from __future__ import annotations

import logging

import mlflow

from config import settings

logger = logging.getLogger(__name__)


def _setup() -> None:
    """Point MLflow at the local tracking folder and select the experiment."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def log_query(
    question: str,
    answer: str,
    top_k: int,
    latency_ms: float,
    context_chars: int,
    model_id: str | None = None,
) -> None:
    """Log a single RAG query as one MLflow run."""
    try:
        _setup()
        with mlflow.start_run():
            mlflow.log_params({
                "question": question[:500],   # MLflow caps param length at 500 chars
                "top_k": top_k,
                "model_id": model_id or settings.base_model_id,
            })
            mlflow.log_metrics({
                "latency_ms": latency_ms,
                "context_chars": context_chars,
            })
            mlflow.log_text(answer, "answer.txt")
    except Exception:
        logger.warning("MLflow logging failed — query still succeeded", exc_info=True)


def log_ingestion(
    chunks_written: int,
    index_vectors: int,
    spatial_layers: list[str],
    duration_ms: float,
) -> None:
    """Log an ingestion pipeline run as one MLflow run."""
    try:
        _setup()
        with mlflow.start_run(run_name="ingestion"):
            mlflow.log_param("spatial_layers", ",".join(spatial_layers))
            mlflow.log_metrics({
                "chunks_written": chunks_written,
                "index_vectors": index_vectors,
                "duration_ms": duration_ms,
            })
    except Exception:
        logger.warning("MLflow logging failed — ingestion still succeeded", exc_info=True)
