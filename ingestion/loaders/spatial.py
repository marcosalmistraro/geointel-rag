"""
Loads structured spatial data for the 2023 Turkey-Syria earthquake.

Two sources:
  - ShakeMap intensity contours (M 7.8 and M 7.5)
  - HOT OSM destroyed buildings

These are NOT embedded into the vector store. Instead they are loaded
at query time and used to enrich LLM answers with grounded spatial
context — e.g. "Hatay experienced MMI 8.5 shaking and had 847
destroyed buildings recorded in that area."

Main entry point:
    from ingestion.loaders.spatial import load_all_spatial
    gdf = load_all_spatial()
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

# Paths relative to project root
SHAKEMAP_78 = Path("data/raw/geojson/shakemap_7.8_pazarcik.geojson")
SHAKEMAP_75 = Path("data/raw/geojson/shakemap_7.5_elbistan.geojson")
BUILDINGS   = Path("data/raw/geojson/hotosm_tur_destroyed_buildings.geojson")


def load_shakemap(path: Path, magnitude: float) -> gpd.GeoDataFrame:
    """
    Load a ShakeMap GeoJSON file.
    Adds 'magnitude' and 'mmi' columns, drops unnecessary ones.
    """
    logger.info("Loading ShakeMap M%.1f from %s ...", magnitude, path)
    gdf = gpd.read_file(str(path))

    # Normalise to WGS-84 if needed
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf = gdf.rename(columns={"value": "mmi"})
    gdf["magnitude"] = magnitude
    gdf["source"]    = "shakemap"

    # Keep only what's useful
    gdf = gdf[["mmi", "magnitude", "source", "geometry"]]

    logger.info("  Loaded %d intensity contours", len(gdf))
    return gdf


def load_buildings(path: Path = BUILDINGS) -> gpd.GeoDataFrame:
    """
    Load the HOT OSM destroyed buildings GeoJSON.
    Adds a 'source' tag and computes centroid lat/lon for map rendering.
    """
    logger.info("Loading destroyed buildings from %s ...", path)
    gdf = gpd.read_file(str(path))

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf["source"]        = "hotosm_buildings"
    centroids = gdf.geometry.to_crs(epsg=3857).centroid.to_crs(epsg=4326)
    gdf["centroid_lat"] = centroids.y
    gdf["centroid_lon"] = centroids.x

    # Keep only useful columns
    keep = ["source", "damage_type", "centroid_lat", "centroid_lon", "geometry"]
    gdf = gdf[[c for c in keep if c in gdf.columns]]

    logger.info("  Loaded %d destroyed buildings", len(gdf))
    return gdf


def load_all_spatial() -> dict[str, gpd.GeoDataFrame]:
    """
    Load all spatial datasets and return them as a named dictionary.
    The RAG chain calls this at query time.

    Returns:
        {
            "shakemap": GeoDataFrame of combined intensity contours,
            "buildings": GeoDataFrame of destroyed buildings,
        }
    """
    shakemap_78 = load_shakemap(SHAKEMAP_78, magnitude=7.8)
    shakemap_75 = load_shakemap(SHAKEMAP_75, magnitude=7.5)
    shakemap    = gpd.GeoDataFrame(
        pd.concat([shakemap_78, shakemap_75], ignore_index=True),
        crs="EPSG:4326",
    )

    buildings = load_buildings()

    logger.info(
        "Spatial index ready — %d intensity contours, %d buildings",
        len(shakemap), len(buildings),
    )
    return {"shakemap": shakemap, "buildings": buildings}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spatial = load_all_spatial()
    for name, gdf in spatial.items():
        print(f"\n{name}: {len(gdf)} features")
        print(gdf.dtypes)