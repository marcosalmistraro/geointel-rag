"""
Downloads the HOT OpenStreetMap destroyed buildings dataset for the
2023 Turkey earthquake from the Humanitarian Data Exchange (HDX).

Dataset: HOTOSM Turkey Destroyed Buildings
Source:  https://data.humdata.org/dataset/hotosm_tur_destroyed_buildings
Format:  GeoJSON (zipped)

Each feature is a building polygon tagged as destroyed on 2023-02-06,
with province and district attributes that link to ReliefWeb report text.

Saves to: data/raw/geojson/hotosm_tur_destroyed_buildings.geojson

Run with:
    python -m ingestion.downloaders.hdx_buildings
"""

import io
import logging
import zipfile
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

OUT_DIR = Path("data/raw/geojson")
OUT_FILE = OUT_DIR / "hotosm_tur_destroyed_buildings.geojson"

# Direct download URL from HDX — no auth required
DOWNLOAD_URL = (
    "https://data.humdata.org/dataset/hotosm_tur_destroyed_buildings/"
    "resource/fb8297e6-71a3-4368-80c1-c3e6973077e2/"
    "download/hotosm_tur_destroyed_buildings_polygons_geojson.zip"
)


def download_buildings() -> None:
    if OUT_FILE.exists():
        logger.info("File already exists at %s — skipping download", OUT_FILE)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading destroyed buildings GeoJSON from HDX...")
    response = requests.get(DOWNLOAD_URL, timeout=60)
    response.raise_for_status()

    # The file comes as a zip — extract the GeoJSON from memory
    logger.info("Extracting GeoJSON from zip...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # Find the .geojson file inside the zip
        geojson_names = [n for n in zf.namelist() if n.endswith(".geojson")]

        if not geojson_names:
            raise FileNotFoundError(
                f"No .geojson file found in zip. Contents: {zf.namelist()}"
            )

        with zf.open(geojson_names[0]) as geojson_file:
            content = geojson_file.read()

    OUT_FILE.write_bytes(content)
    logger.info("Saved %d bytes to %s", len(content), OUT_FILE)


def main() -> None:
    logger.info("=== HDX buildings downloader — 2023 Turkey earthquake ===")
    download_buildings()
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()