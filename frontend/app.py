"""
GeoIntel RAG — Streamlit frontend.

Two-panel layout:
  Left  — question input, answer, retrieved context
  Right — Folium map of ShakeMap intensity zones + destroyed buildings

Start with (keep the FastAPI server running in a separate terminal):
    streamlit run frontend/app.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import folium
import requests
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

SPATIAL_PATH = Path("data/processed/spatial.pkl")
DEFAULT_API_URL = "http://localhost:8000"

# ── helpers ──────────────────────────────────────────────────────────────────

def _mmi_color(mmi: float) -> str:
    """Map an MMI value to a fill colour for the ShakeMap polygons."""
    if mmi >= 8:
        return "#d73027"   # deep red  — severe
    if mmi >= 7:
        return "#f46d43"   # orange-red
    if mmi >= 6:
        return "#fdae61"   # orange
    if mmi >= 5:
        return "#fee090"   # yellow
    return "#e0f3f8"       # light blue — minor


@st.cache_resource
def load_spatial() -> dict | None:
    """Load the spatial index from disk once and keep it in memory."""
    if not SPATIAL_PATH.exists():
        return None
    with SPATIAL_PATH.open("rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_map() -> folium.Map:
    """
    Build the base Folium map with ShakeMap contours and building markers.
    Cached so it is only constructed once per server session.
    """
    m = folium.Map(
        location=[37.0, 37.5],
        zoom_start=7,
        tiles=None,   # no external tile CDN — works offline / behind corporate proxy
    )
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap",
        name="Street map",
        show=True,
    ).add_to(m)

    spatial = load_spatial()
    if spatial is None:
        return m

    # — ShakeMap intensity contours —
    shakemap = spatial["shakemap"]
    shakemap_geojson = json.loads(shakemap.to_json())

    folium.GeoJson(
        shakemap_geojson,
        name="ShakeMap intensity",
        style_function=lambda feat: {
            "fillColor": _mmi_color(float(feat["properties"].get("mmi") or 0)),
            "color": "#555",
            "weight": 0.4,
            "fillOpacity": 0.45,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["mmi", "magnitude"],
            aliases=["MMI intensity", "Earthquake magnitude"],
        ),
    ).add_to(m)

    # — Destroyed buildings (sampled for map performance) —
    buildings = spatial["buildings"]
    if "centroid_lat" in buildings.columns and "centroid_lon" in buildings.columns:
        sample = buildings.dropna(subset=["centroid_lat", "centroid_lon"])
        if len(sample) > 4000:
            sample = sample.sample(4000, random_state=42)

        cluster = MarkerCluster(name="Destroyed buildings").add_to(m)
        for _, row in sample.iterrows():
            folium.CircleMarker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                radius=3,
                color="#8b0000",
                fill=True,
                fill_opacity=0.7,
                tooltip="Destroyed building (HOT OSM)",
            ).add_to(cluster)

    folium.LayerControl().add_to(m)
    return m


# ── page layout ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GeoIntel RAG",
    page_icon="🌍",
    layout="wide",
)

st.title("GeoIntel RAG")
st.caption("Natural-language intelligence over the 2023 Turkey-Syria earthquake response corpus.")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    st.divider()
    st.markdown("**Data sources**")
    st.markdown("- ReliefWeb situation reports")
    st.markdown("- USGS ShakeMap M7.8 + M7.5")
    st.markdown("- HOT OSM destroyed buildings")
    st.markdown("- OCHA key figures")
    st.divider()
    st.markdown("**Model**")
    st.markdown("`llama-3.1-8b-instant`")
    st.markdown("via Groq API")

# Two-column layout
col_qa, col_map = st.columns([1, 1], gap="large")

# ── left column: Q&A ─────────────────────────────────────────────────────────
with col_qa:
    st.subheader("Ask a question")

    question = st.text_area(
        "question",
        placeholder=(
            "e.g. What was the humanitarian situation in Hatay?\n"
            "e.g. How many people were displaced in Gaziantep?\n"
            "e.g. What search and rescue operations were conducted?"
        ),
        label_visibility="collapsed",
        height=100,
    )

    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="How many text passages from the reports to send to the LLM as context.",
    )

    submit = st.button("Ask", type="primary", use_container_width=True)

    if submit and question.strip():
        with st.spinner("Retrieving context and generating answer…"):
            try:
                resp = requests.post(
                    f"{api_url}/query",
                    json={"question": question.strip(), "top_k": top_k},
                    timeout=90,
                )
                resp.raise_for_status()
                data = resp.json()

                st.markdown("### Answer")
                st.markdown(data["answer"])
                st.caption(f"Latency: {data['latency_ms']:.0f} ms")

                with st.expander("Retrieved context passages"):
                    st.text(data["context"])

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach the API. "
                    "Make sure `uvicorn api.main:app --reload` is running."
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text[:300]}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    elif submit:
        st.warning("Please enter a question first.")

# ── right column: map ─────────────────────────────────────────────────────────
with col_map:
    st.subheader("Affected area — ShakeMap + destroyed buildings")

    spatial = load_spatial()
    if spatial is None:
        st.warning(
            "Spatial data not found at `data/processed/spatial.pkl`. "
            "Run `python -m ingestion.pipeline` first."
        )
    else:
        m = build_map()
        st_folium(m, use_container_width=True, height=560)

        n_buildings = len(spatial["buildings"])
        n_contours = len(spatial["shakemap"])
        st.caption(
            f"{n_contours} intensity contours · "
            f"{n_buildings:,} destroyed buildings recorded"
        )
