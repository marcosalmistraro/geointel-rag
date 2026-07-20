"""
Downloads situation reports for the 2023 Turkey-Syria earthquake
from the ReliefWeb API.

Saves one JSON file per report under:
    data/raw/reliefweb/eq-2023-000015-tur/
    data/raw/reliefweb/eq-2023-000015-syr/

Run with:
    python -m ingestion.downloaders.reliefweb
"""

import json
import logging
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# The two disaster IDs — Turkey side and Syria side of the same event
DISASTER_IDS = [
    "eq-2023-000015-tur",
    "eq-2023-000015-syr",
]

OUT_DIR = Path("data/raw/reliefweb")
API_URL = "https://api.reliefweb.int/v2/reports"

# We want the full report body, not just the title
FIELDS = [
    "title",
    "body",
    "date.created",
    "country.name",
    "source.name",
    "disaster_type.name",
    "theme.name",
]


def fetch_page(disaster_id: str, offset: int, limit: int = 100) -> dict:
    """Fetch one page of results from the ReliefWeb API."""
    response = requests.get(
        API_URL,
        params={
            "appname": "PLACEHOLDER",  # TODO: replace with your app name
            "limit": limit,
            "offset": offset,
            "filter[field]": "disaster.glide",
            "filter[value]": disaster_id,
            "fields[include][]": FIELDS,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def download_disaster(disaster_id: str) -> int:
    """
    Download all reports for one disaster ID.
    Returns the number of reports saved.
    """
    out_dir = OUT_DIR / disaster_id
    out_dir.mkdir(parents=True, exist_ok=True)

    offset = 0
    limit = 100
    total_saved = 0

    while True:
        logger.info("[%s] Fetching reports (offset=%d)...", disaster_id, offset)

        data = fetch_page(disaster_id, offset, limit)
        reports = data.get("data", [])

        if not reports:
            break

        for report in reports:
            report_id = report["id"]
            out_path = out_dir / f"{report_id}.json"

            # Skip if already downloaded — safe to re-run
            if out_path.exists():
                logger.debug("Skipping %s (already exists)", report_id)
                continue

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            total_saved += 1

        logger.info("[%s] Saved %d reports so far", disaster_id, total_saved)

        # Fewer results than the limit means we're on the last page
        if len(reports) < limit:
            break

        offset += limit
        time.sleep(1)  # avoid hammering the API

    return total_saved


def main() -> None:
    logger.info("=== ReliefWeb downloader — 2023 Turkey-Syria earthquake ===")

    total = 0
    for disaster_id in DISASTER_IDS:
        count = download_disaster(disaster_id)
        logger.info("[%s] Done — %d reports downloaded", disaster_id, count)
        total += count

    logger.info("=== Finished — %d total reports saved to %s ===", total, OUT_DIR)


if __name__ == "__main__":
    main()