"""
Loads ReliefWeb situation reports from data/raw/reliefweb/.

Each report is a JSON file with the structure:
    {
        "id": "123456",
        "fields": {
            "title": "...",
            "body": "...",
            "date": {"created": "2023-02-23"},
            "country": [{"name": "Türkiye"}],
            "source": [{"name": "OCHA"}]
        }
    }

Returns a list of dicts with a consistent shape:
    {
        "text": str,        # title + body combined
        "source": str,      # original file path
        "metadata": dict    # date, country, source org, report id
    }

Run with:
    python -m ingestion.loaders.text
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

RELIEFWEB_DIR = Path("data/raw/reliefweb")


def load_report(path: Path) -> dict | None:
    """
    Load a single ReliefWeb JSON report.
    Returns None if the file is missing required fields.
    """
    try:
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        logger.error("Failed to read %s: %s", path, e)
        return None

    fields = raw.get("fields", {})
    title  = fields.get("title", "").strip()
    body   = fields.get("body", "").strip()

    # Skip reports with no usable text
    if not body:
        logger.warning("Skipping %s — no body text", path.name)
        return None

    # Combine title and body so the chunk always has context
    text = f"{title}\n\n{body}" if title else body

    # Extract metadata
    date     = fields.get("date", {}).get("created", "")
    countries = [c["name"] for c in fields.get("country", []) if "name" in c]
    sources   = [s["name"] for s in fields.get("source", []) if "name" in s]

    return {
        "text": text,
        "source": str(path),
        "metadata": {
            "report_id": str(raw.get("id", "")),
            "title": title,
            "date": date,
            "countries": countries,
            "organizations": sources,
            "doc_type": "reliefweb_report",
        },
    }


def load_all_reports(directory: Path = RELIEFWEB_DIR) -> list[dict]:
    """
    Load all ReliefWeb JSON reports from a directory tree.
    Walks subdirectories so both disaster IDs are covered:
        data/raw/reliefweb/eq-2023-000015-tur/
        data/raw/reliefweb/eq-2023-000015-syr/
    """
    if not directory.exists():
        logger.warning("ReliefWeb directory not found: %s", directory)
        return []

    paths = sorted(directory.rglob("*.json"))
    if not paths:
        logger.warning("No JSON files found in %s", directory)
        return []

    records = []
    for path in paths:
        record = load_report(path)
        if record:
            records.append(record)

    logger.info("Loaded %d reports from %s", len(records), directory)
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    records = load_all_reports()
    if records:
        print(f"\nLoaded {len(records)} reports")
        print(f"First record title: {records[0]['metadata']['title']}")
        print(f"First record date:  {records[0]['metadata']['date']}")
        print(f"Text preview:       {records[0]['text'][:200]}")
    else:
        print("No reports loaded — waiting for ReliefWeb appname approval")