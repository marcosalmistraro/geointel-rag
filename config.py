from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Model
    hf_token: str = Field(default="")
    base_model_id: str = Field(default="microsoft/Phi-3-mini-4k-instruct")
    embed_model_id: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # Paths
    data_raw_dir: Path = Field(default=Path("data/raw"))
    data_processed_dir: Path = Field(default=Path("data/processed"))
    data_training_dir: Path = Field(default=Path("data/training"))
    vector_store_path: Path = Field(default=Path("data/processed/faiss_index"))

    # Ingestion
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # RAG
    retriever_top_k: int = Field(default=5)

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # MLflow
    mlflow_tracking_uri: str = Field(default="mlruns")
    mlflow_experiment_name: str = Field(default="geointel-rag")

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for d in [self.data_raw_dir, self.data_processed_dir, self.data_training_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()