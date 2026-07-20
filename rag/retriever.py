"""
Retrieves relevant chunks for a query and enriches them with spatial context.

Flow:
    query → vector store search → top-k chunks
          → spatial enrichment (MMI intensity + building counts)
          → formatted context string for LLM prompt

Main entry point:
    from rag.retriever import Retriever
    retriever = Retriever()
    context = retriever.retrieve(query="what happened in Hatay?")

Run with:
    python -m rag.retriever
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

SPATIAL_PATH = Path("data/processed/spatial.pkl")

# Provinces mentioned in the Turkey-Syria earthquake response
PROVINCE_NAMES = [
    "hatay", "kahramanmaras", "gaziantep", "adiyaman", "malatya",
    "osmaniye", "adana", "sanliurfa", "diyarbakir", "kilis",
    "elazig", "idlib", "aleppo", "hama",
]


class Retriever:

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        spatial_path: Path = SPATIAL_PATH,
        top_k: int = 5,
    ) -> None:
        self.store = vector_store or VectorStore()
        self.store.load()
        self.top_k = top_k
        self.spatial = self._load_spatial(spatial_path)

    def _load_spatial(self, path: Path) -> dict | None:
        if not path.exists():
            logger.warning("Spatial index not found at %s", path)
            return None
        with path.open("rb") as f:
            spatial = pickle.load(f)
        logger.info("Spatial index loaded")
        return spatial

    def _detect_provinces(self, text: str) -> list[str]:
        """Return any province names found in the text."""
        text_lower = text.lower()
        return [p for p in PROVINCE_NAMES if p in text_lower]

    def _get_spatial_context(self, provinces: list[str]) -> str:
        """
        For each detected province, look up MMI intensity and
        building counts from the spatial index.
        Returns a formatted string ready for the LLM prompt.
        """
        if not self.spatial or not provinces:
            return ""

        shakemap: gpd.GeoDataFrame = self.spatial["shakemap"]
        buildings: gpd.GeoDataFrame = self.spatial["buildings"]
        lines = []

        for province in provinces:
            # Rough province centroids (WGS-84) for spatial lookup
            province_coords = {
                "hatay":         (36.20, 36.50),
                "kahramanmaras": (36.93, 37.58),
                "gaziantep":     (37.05, 37.07),
                "adiyaman":      (38.27, 37.76),
                "malatya":       (38.35, 38.35),
                "osmaniye":      (36.25, 37.07),
                "adana":         (35.33, 37.00),
                "sanliurfa":     (38.79, 37.16),
                "diyarbakir":    (40.23, 37.91),
                "kilis":         (37.12, 36.72),
                "elazig":        (39.22, 38.67),
                "idlib":         (36.63, 35.93),
                "aleppo":        (37.16, 36.20),
                "hama":          (36.75, 35.13),
            }

            if province not in province_coords:
                continue

            lon, lat = province_coords[province]
            point = Point(lon, lat)

            # Find which MMI zone this province falls in
            point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326").to_crs(epsg=3857)
            shakemap_proj = shakemap.to_crs(epsg=3857)
            containing = shakemap_proj[
                shakemap_proj.geometry.distance(point_gdf.geometry.iloc[0]) < 10000
            ]
            if not containing.empty:
                max_mmi = containing["mmi"].max()
                lines.append(
                    f"- {province.title()}: MMI intensity {max_mmi:.1f} "
                    f"(M{containing.iloc[0]['magnitude']} earthquake)"
                )

            # Count destroyed buildings within ~50km of province centroid
            point_gdf = gpd.GeoDataFrame(
                geometry=[point], crs="EPSG:4326"
            ).to_crs(epsg=3857)
            buildings_proj = buildings.to_crs(epsg=3857)
            nearby = buildings_proj[
                buildings_proj.geometry.distance(point_gdf.geometry.iloc[0]) < 50000
            ]
            if not nearby.empty:
                lines.append(
                    f"- {province.title()}: {len(nearby)} destroyed buildings "
                    f"recorded within 50km"
                )

        return "\n".join(lines)

    def retrieve(self, query: str) -> str:
        """
        Retrieve relevant chunks and return a formatted context string.
        """
        chunks = self.store.search(query, top_k=self.top_k)

        if not chunks:
            return "No relevant documents found."

        context_parts = []

        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            header = f"[{i}] {meta.get('title', 'Unknown report')} ({meta.get('date', '')})"
            context_parts.append(f"{header}\n{chunk['text']}")

        # Detect provinces across all retrieved chunks combined
        all_text = " ".join(c["text"] for c in chunks)
        provinces = self._detect_provinces(all_text)
        spatial_context = self._get_spatial_context(provinces)

        context = "\n\n".join(context_parts)

        if spatial_context:
            context += "\n\nGeospatial context:\n" + spatial_context

        return context


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retriever = Retriever()
    context = retriever.retrieve("what was the situation in Hatay?")
    print("\n--- Retrieved context ---")
    print(context)