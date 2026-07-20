"""
Runs all downloaders in sequence.

Run with:
    python -m ingestion.downloaders.run_all
"""

import logging

from ingestion.downloaders import hdx_buildings, hdx_shakemap, hdx_ocha_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== GeoIntel RAG — running all downloaders ===")

    logger.info("--- 1/3 HOT OSM destroyed buildings ---")
    hdx_buildings.main()

    logger.info("--- 2/3 USGS ShakeMap intensity contours ---")
    hdx_shakemap.main()

    logger.info("--- 3/3 OCHA key figures ---")
    hdx_ocha_figures.main()

    logger.info("=== All downloaders complete ===")


if __name__ == "__main__":
    main()