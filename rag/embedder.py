"""
Embeds text chunks using sentence-transformers/all-MiniLM-L6-v2.

The model runs locally — no API calls and no cost.
First run downloads ~80MB to the HuggingFace cache automatically.

Main entry point:
    from rag.embedder import Embedder
    embedder = Embedder()
    vectors = embedder.embed(["text one", "text two"])

Run with:
    python -m rag.embedder
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """
    Thin wrapper around SentenceTransformer.
    Keeps the model loaded in memory so repeated calls don't reload it.
    """

    def __init__(self, model_id: str = MODEL_ID) -> None:
        logger.info("Loading embedding model %s ...", model_id)
        self.model = SentenceTransformer(model_id)
        self.model_id = model_id
        logger.info("Embedding model ready")

    def embed(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of strings.
        Returns a float32 numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            raise ValueError("Cannot embed an empty list")

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity friendly
        )
        logger.info("Embedded %d texts → shape %s", len(texts), vectors.shape)
        return vectors.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        """Dimension of the output vectors — 384 for MiniLM-L6."""
        return self.model.get_embedding_dimension()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embedder = Embedder()
    print(f"Embedding dim: {embedder.embedding_dim}")

    test_texts = [
        "Earthquake damage assessment in Hatay province",
        "Food security situation in southern Turkey",
        "MMI intensity 8.5 zone covering Kahramanmaras",
    ]
    vectors = embedder.embed(test_texts, show_progress=True)
    print(f"Output shape: {vectors.shape}")
    print(f"First vector norm: {np.linalg.norm(vectors[0]):.4f}")  # should be ~1.0