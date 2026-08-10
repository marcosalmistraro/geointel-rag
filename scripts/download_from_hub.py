"""
Download processed data from HuggingFace Hub if not already on disk.
Called during Render build; can also be run locally.

    python scripts/download_from_hub.py
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "marcosalmistraro/geointel-rag-data"
FILES = ["faiss_index.bin", "chunks.jsonl", "spatial.pkl"]
DEST = Path("data/processed")


def download_if_missing() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        dest = DEST / filename
        if dest.exists():
            print(f"  {filename} already present — skipping")
            continue
        print(f"  Downloading {filename}…", end=" ", flush=True)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(DEST),
        )
        print("done.")


if __name__ == "__main__":
    print(f"Fetching data from {REPO_ID}…")
    download_if_missing()
    print("Data ready.")
