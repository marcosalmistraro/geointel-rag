"""
GeoIntel RAG — Streamlit frontend.

Tabs:
  Ask     — question input, answer, retrieved context + ShakeMap/buildings map
  Sources — description of every data source with links

Start with (keep the FastAPI server running in a separate terminal):
    streamlit run frontend/app.py
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import folium
import requests
import streamlit as st
import streamlit.components.v1 as st_components
from folium.plugins import MarkerCluster

SPATIAL_PATH = Path("data/processed/spatial.pkl")
DEFAULT_API_URL = "http://localhost:8000"

SOURCES = [
    {
        "name": "ReliefWeb",
        "homepage": "https://reliefweb.int/",
        "dataset": "https://api.reliefweb.int/v2/reports",
        "provider": "UN OCHA",
        "description": (
            "ReliefWeb is the UN's humanitarian information portal, aggregating "
            "situation reports, news, and analysis from hundreds of humanitarian "
            "organizations worldwide."
        ),
        "used_for": (
            "Situation reports covering the Feb–May 2023 earthquake response are "
            "downloaded via the ReliefWeb API, chunked into 512-token passages, "
            "embedded, and stored in the FAISS vector index. These passages form "
            "the retrieval corpus for all Q&A."
        ),
        "format": "JSON (API)",
    },
    {
        "name": "USGS ShakeMap — M7.8 Pazarcik",
        "homepage": "https://earthquake.usgs.gov/data/shakemap/",
        "dataset": "https://data.humdata.org/dataset/50d93259-2d49-4f84-85e6-3cd0aa03dfaa",
        "provider": "USGS / HDX",
        "description": (
            "USGS ShakeMap provides ground-shaking intensity estimates for the "
            "M7.8 Pazarcik mainshock (6 Feb 2023). Intensity is expressed as "
            "Modified Mercalli Intensity (MMI), ranging from I (not felt) to X+ "
            "(extreme damage)."
        ),
        "used_for": (
            "MMI contour polygons are overlaid on the map as colour-coded zones "
            "(red = severe, yellow = moderate, blue = minor) so you can visually "
            "correlate humanitarian reports with affected areas."
        ),
        "format": "GeoJSON",
    },
    {
        "name": "USGS ShakeMap — M7.5 Elbistan",
        "homepage": "https://earthquake.usgs.gov/data/shakemap/",
        "dataset": "https://data.humdata.org/dataset/50d93259-2d49-4f84-85e6-3cd0aa03dfaa",
        "provider": "USGS / HDX",
        "description": (
            "ShakeMap for the M7.5 Elbistan aftershock (6 Feb 2023, ~9 hours after "
            "the mainshock), which caused additional damage across Kahramanmaras "
            "and neighbouring provinces."
        ),
        "used_for": (
            "Combined with the M7.8 ShakeMap on the same intensity layer to show "
            "the cumulative shaking extent from both major events."
        ),
        "format": "GeoJSON",
    },
    {
        "name": "HOT OSM — Destroyed Buildings",
        "homepage": "https://www.hotosm.org/",
        "dataset": "https://data.humdata.org/dataset/hotosm_tur_destroyed_buildings",
        "provider": "Humanitarian OpenStreetMap Team / HDX",
        "description": (
            "The Humanitarian OpenStreetMap Team (HOT) coordinated a rapid mapping "
            "activation after the earthquake, tracing destroyed and damaged buildings "
            "from satellite imagery across affected Turkish provinces."
        ),
        "used_for": (
            "Building footprint centroids are plotted as clustered red markers on "
            "the map, giving a spatial sense of destruction density at the "
            "neighbourhood level."
        ),
        "format": "GeoJSON (zipped)",
    },
    {
        "name": "OCHA — Key Figures",
        "homepage": "https://www.unocha.org/",
        "dataset": "https://data.humdata.org/dataset/turkiye-syria-earthquake-key-figures",
        "provider": "UN OCHA / HDX",
        "description": (
            "OCHA's consolidated summary statistics for the earthquake response: "
            "casualties, displaced persons, affected population, shelter needs, "
            "and response funding — updated regularly during the crisis."
        ),
        "used_for": (
            "Key figures are ingested as structured text and included in the "
            "retrieval corpus, so the RAG system can answer quantitative questions "
            "such as 'How many people were displaced?' with grounded numbers."
        ),
        "format": "CSV",
    },
]


EXAMPLE_QUESTIONS = [
    "What was the humanitarian situation in Hatay?",
    "How many people were displaced across the affected provinces?",
    "What search and rescue operations were conducted?",
    "What was the food security situation in Gaziantep?",
    "How did the earthquake affect Syrian refugees?",
]


def _parse_context(context: str) -> tuple[list[dict], str]:
    """Split the retriever context string into chunk dicts + geospatial block."""
    parts = context.split("\n\nGeospatial context:")
    spatial_text = parts[1].strip() if len(parts) > 1 else ""

    raw_chunks = re.split(r"\n\n(?=\[\d+\])", parts[0].strip())
    chunks = []
    for raw in raw_chunks:
        lines = raw.strip().split("\n", 1)
        header = lines[0]
        text = lines[1].strip() if len(lines) > 1 else ""
        m = re.match(r"\[(\d+)\]\s+(.*?)\s+\(([^)]*)\)\s*$", header)
        if m:
            chunks.append({"index": int(m.group(1)), "title": m.group(2), "date": m.group(3), "text": text})
        else:
            chunks.append({"index": 0, "title": header, "date": "", "text": text})
    return chunks, spatial_text


# ── helpers ───────────────────────────────────────────────────────────────────

def _mmi_color(mmi: float) -> str:
    if mmi >= 8:
        return "#d73027"
    if mmi >= 7:
        return "#f46d43"
    if mmi >= 6:
        return "#fdae61"
    if mmi >= 5:
        return "#fee090"
    return "#e0f3f8"


@st.cache_resource
def load_spatial() -> dict | None:
    if not SPATIAL_PATH.exists():
        return None
    with SPATIAL_PATH.open("rb") as f:
        return pickle.load(f)


def _build_map() -> folium.Map:
    m = folium.Map(
        location=[37.0, 37.5],
        zoom_start=7,
        tiles=None,
        width="100%",
        height=540,
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

    shakemap = spatial["shakemap"]
    folium.GeoJson(
        json.loads(shakemap.to_json()),
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


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GeoIntel RAG",
    page_icon="🌍",
    layout="wide",
)

st.title("GeoIntel RAG")
st.caption("Natural-language intelligence over the 2023 Turkey-Syria earthquake response corpus.")

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    st.divider()
    st.markdown("**Model**")
    st.markdown("`llama-3.1-8b-instant`")
    st.markdown("via [Groq API](https://groq.com)")

# ── tabs ──────────────────────────────────────────────────────────────────────

tab_ask, tab_sources = st.tabs(["Ask", "Sources"])

# ── Ask tab ───────────────────────────────────────────────────────────────────

with tab_ask:
    col_qa, col_map = st.columns([1, 1], gap="large")

    with col_qa:
        st.subheader("Ask a question")

        if "question" not in st.session_state:
            st.session_state.question = ""

        def _apply_example():
            val = st.session_state._example_select
            if val != "— pick an example question —":
                st.session_state.question = val

        st.selectbox(
            "example",
            ["— pick an example question —"] + EXAMPLE_QUESTIONS,
            key="_example_select",
            on_change=_apply_example,
            label_visibility="collapsed",
        )

        question = st.text_area(
            "question",
            key="question",
            placeholder="Type your question or pick an example above…",
            height=100,
            label_visibility="collapsed",
        )

        top_k = st.slider(
            "Chunks to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="How many text passages from the reports to send to the LLM as context.",
        )

        if "history" not in st.session_state:
            st.session_state.history = []

        col_ask, col_export = st.columns([3, 1])
        submit = col_ask.button("Ask", type="primary", use_container_width=True)

        if st.session_state.history:
            import io, csv
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=["question", "answer", "latency_ms"])
            writer.writeheader()
            writer.writerows(st.session_state.history)
            col_export.download_button(
                "Export CSV",
                data=buf.getvalue(),
                file_name="geointel_session.csv",
                mime="text/csv",
                use_container_width=True,
            )

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

                    st.session_state.history.append({
                        "question": question.strip(),
                        "answer": data["answer"],
                        "latency_ms": round(data["latency_ms"]),
                    })

                    st.markdown("### Answer")
                    st.markdown(data["answer"])
                    st.caption(f"Latency: {data['latency_ms']:.0f} ms")

                    with st.expander("Retrieved context passages"):
                        chunks, spatial = _parse_context(data["context"])
                        for chunk in chunks:
                            with st.container(border=True):
                                st.markdown(
                                    f"**[{chunk['index']}] {chunk['title']}**"
                                    + (f"  `{chunk['date']}`" if chunk["date"] else "")
                                )
                                st.caption(chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else ""))
                        if spatial:
                            st.markdown("**Geospatial context**")
                            st.info(spatial)

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

        if len(st.session_state.history) > 1:
            st.divider()
            st.markdown("#### Session history")
            for item in reversed(st.session_state.history[:-1]):
                with st.expander(item["question"][:80]):
                    st.markdown(item["answer"])
                    st.caption(f"Latency: {item['latency_ms']} ms")

    with col_map:
        st.subheader("Affected area — ShakeMap + destroyed buildings")

        spatial = load_spatial()
        if spatial is None:
            st.warning(
                "Spatial data not found at `data/processed/spatial.pkl`. "
                "Run `python -m ingestion.pipeline` first."
            )
        else:
            if "_folium_map_html" not in st.session_state:
                st.session_state._folium_map_html = _build_map()._repr_html_()
            st_components.html(st.session_state._folium_map_html, height=560)

            n_buildings = len(spatial["buildings"])
            n_contours = len(spatial["shakemap"])
            st.caption(
                f"{n_contours} intensity contours · "
                f"{n_buildings:,} destroyed buildings recorded"
            )

# ── Sources tab ───────────────────────────────────────────────────────────────

with tab_sources:
    st.subheader("Data sources")
    st.markdown(
        "All data is open and publicly available. "
        "Click any dataset link to download or explore the original source."
    )
    st.divider()

    for src in SOURCES:
        with st.container():
            col_info, col_links = st.columns([3, 1])

            with col_info:
                st.markdown(f"### {src['name']}")
                st.caption(f"Provider: {src['provider']} · Format: {src['format']}")
                st.markdown(src["description"])
                st.markdown(f"**How it is used:** {src['used_for']}")

            with col_links:
                st.markdown("&nbsp;")
                st.link_button("Homepage", src["homepage"], use_container_width=True)
                st.link_button("Dataset", src["dataset"], use_container_width=True)

        st.divider()
