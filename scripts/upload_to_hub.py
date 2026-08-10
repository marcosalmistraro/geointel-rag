"""
Upload processed data files to HuggingFace Hub.
Run once locally before deploying to Render.

    python scripts/upload_to_hub.py

Requires HF_TOKEN in .env or environment.
The dataset repo is created automatically if it does not exist.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

REPO_ID = "marcosalmistraro/geointel-rag-data"
FILES = [
    Path("data/processed/faiss_index.bin"),
    Path("data/processed/chunks.jsonl"),
    Path("data/processed/spatial.pkl"),
]


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set — add it to .env")

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, private=False)
    print(f"Repo: https://huggingface.co/datasets/{REPO_ID}\n")

    for path in FILES:
        if not path.exists():
            print(f"SKIP  {path.name} — file not found locally")
            continue
        size_mb = path.stat().st_size / 1e6
        print(f"Uploading {path.name} ({size_mb:.1f} MB)…", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        print("done.")

    print(f"\nAll files available at https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
