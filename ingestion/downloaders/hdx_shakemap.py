"""
Downloads USGS ShakeMap intensity contours for the 2023 Turkey earthquake
sequence from the Humanitarian Data Exchange (HDX).

Two files — one per earthquake in the doublet:
  - M 7.8 Pazarcik earthquake  (main shock, 04:17 local time)
  - M 7.5 Elbistan earthquake  (second shock, 9 hours later)

Each file is a GeoJSON FeatureCollection of intensity contour polygons.
Each feature has a 'PARAMVALUE' field with the MMI intensity level
(e.g. "II", "V", "VIII") and a geometry covering the area that
experienced that intensity.

The province names inside these polygons (Hatay, Gaziantep, Kahramanmaras
etc.) are the same names that appear in ReliefWeb report text — that's
the link between this geodata and the text corpus.

Saves to:
    data/raw/geojson/shakemap_7.8_pazarcik.geojson
    data/raw/geojson/shakemap_7.5_elbistan.geojson

Run with:
    python -m ingestion.downloaders.hdx_shakemap
"""

import logging
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

OUT_DIR = Path("data/raw/geojson")

SHAKEMAPS = [
    {
        "name": "shakemap_7.8_pazarcik.geojson",
        "url": (
            "https://data.humdata.org/dataset/50d93259-2d49-4f84-85e6-3cd0aa03dfaa"
            "/resource/eaa64ce6-8737-408d-8e51-672a5debeee0/download"
            "/m-7.8-pazarcik-earthquake-kahramanmaras-earthquake-sequence.json"
        ),
        "description": "M 7.8 Pazarcik earthquake — MMI intensity contours",
    },
    {
        "name": "shakemap_7.5_elbistan.geojson",
        "url": (
            "https://data.humdata.org/dataset/50d93259-2d49-4f84-85e6-3cd0aa03dfaa"
            "/resource/f7663c38-3a5f-45c8-9566-d4e57b3a8664/download"
            "/m-7.5-elbistan-earthquake-kahramanmaras-earthquake-sequence.json"
        ),
        "description": "M 7.5 Elbistan earthquake — MMI intensity contours",
    },
]


def download_shakemap(entry: dict) -> None:
    out_path = OUT_DIR / entry["name"]

    if out_path.exists():
        logger.info("Already exists — skipping %s", entry["name"])
        return

    logger.info("Downloading %s ...", entry["description"])
    response = requests.get(entry["url"], timeout=60)
    response.raise_for_status()

    out_path.write_bytes(response.content)
    logger.info("Saved %d bytes to %s", len(response.content), out_path)


def main() -> None:
    logger.info("=== HDX ShakeMap downloader — 2023 Turkey earthquake ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for entry in SHAKEMAPS:
        try:
            download_shakemap(entry)
        except requests.HTTPError as e:
            logger.error("Failed to download %s: %s", entry["name"], e)

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()