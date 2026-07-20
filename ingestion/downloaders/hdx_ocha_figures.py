"""
Downloads OCHA key figures for the 2023 Turkey-Syria earthquake
from the Humanitarian Data Exchange (HDX).

Dataset: Turkiye & Syria Earthquakes - OCHA SitReps Key Figures
Source:  https://data.humdata.org/dataset/turkiye-syria-earthquake-key-figures
Format:  XLSX (served as Excel despite CSV filename on HDX)

Each row is a daily snapshot of a key figure (deaths, displaced,
buildings damaged etc.) extracted from OCHA situation reports.
The 'Admin1' column contains province names — Hatay, Gaziantep,
Kahramanmaras etc. — which are the same names that appear in the
ReliefWeb report text and within the ShakeMap intensity zones.
That's the link across all three datasets.

Saves to: data/raw/xlsx/ocha_key_figures.xlsx

Run with:
    python -m ingestion.downloaders.hdx_ocha_figures
"""

import logging
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

OUT_DIR = Path("data/raw/xlsx")
OUT_FILE = OUT_DIR / "ocha_key_figures.xlsx"

DOWNLOAD_URL = (
    "https://data.humdata.org/dataset/turkiye-syria-earthquake-key-figures"
    "/resource/d5ef53ae-a323-41c4-8f74-c719529cb2e4/download"
    "/turkiye-syria-earthquake-key-figures.csv"
)


def download_key_figures() -> None:
    if OUT_FILE.exists():
        logger.info("File already exists at %s — skipping download", OUT_FILE)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading OCHA key figures CSV from HDX...")
    response = requests.get(DOWNLOAD_URL, timeout=60)
    response.raise_for_status()

    OUT_FILE.write_bytes(response.content)
    logger.info("Saved %d bytes to %s", len(response.content), OUT_FILE)


def main() -> None:
    logger.info("=== HDX OCHA figures downloader — 2023 Turkey earthquake ===")
    download_key_figures()
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()