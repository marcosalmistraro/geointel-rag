"""
GeoIntel RAG - self-contained Streamlit app.

The RAGChain is loaded once at startup via @st.cache_resource.
Data files are downloaded from HuggingFace Hub on first run.

Deploy: HuggingFace Spaces (Streamlit SDK)
Local:  streamlit run app.py
"""

from __future__ import annotations

import csv
import io
import json as _json
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import folium
import streamlit as st
import streamlit.components.v1 as st_components
from folium.plugins import MarkerCluster

from scripts.download_from_hub import download_if_missing
from rag.chain import RAGChain

SPATIAL_PATH = Path("data/processed/spatial.pkl")

AVAILABLE_MODELS = {
    "GPT-OSS 20B - fast": "openai/gpt-oss-20b",
    "GPT-OSS 120B - quality": "openai/gpt-oss-120b",
}

PROVINCE_QUESTIONS = {
    "Hatay": "What was the humanitarian situation in Hatay?",
    "Kahramanmaraş": "What damage was reported in Kahramanmaraş?",
    "Gaziantep": "How many people were displaced in Gaziantep?",
    "Adıyaman": "What relief operations were conducted in Adıyaman?",
    "Malatya": "What was the impact of the earthquake in Malatya?",
    "Osmaniye": "What was the situation in Osmaniye after the earthquake?",
    "Idlib (Syria)": "How did the earthquake affect Idlib in Syria?",
    "Aleppo (Syria)": "What was the humanitarian situation in Aleppo after the earthquake?",
}

SOURCES = [
    {
        "name": "ReliefWeb",
        "homepage": "https://reliefweb.int/",
        "dataset": "https://apidoc.reliefweb.int/",
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
        "name": "USGS ShakeMap - M7.8 Pazarcik",
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
        "name": "USGS ShakeMap - M7.5 Elbistan",
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
        "name": "HOT OSM - Destroyed Buildings",
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
        "name": "OCHA - Key Figures",
        "homepage": "https://www.unocha.org/",
        "dataset": "https://data.humdata.org/dataset/turkiye-syria-earthquake-key-figures",
        "provider": "UN OCHA / HDX",
        "description": (
            "OCHA's consolidated summary statistics for the earthquake response: "
            "casualties, displaced persons, affected population, shelter needs, "
            "and response funding - updated regularly during the crisis."
        ),
        "used_for": (
            "Key figures are ingested as structured text and included in the "
            "retrieval corpus, so the RAG system can answer quantitative questions "
            "such as 'How many people were displaced?' with grounded numbers."
        ),
        "format": "CSV",
    },
]

EVAL_TEST_CASES = [
    {"question": "How many people were killed in the earthquake?",
     "keywords": ["killed", "dead", "deaths", "casualties", "people"]},
    {"question": "What was the humanitarian situation in Hatay?",
     "keywords": ["hatay", "shelter", "displaced", "damage", "buildings"]},
    {"question": "How many people were displaced by the earthquake?",
     "keywords": ["displaced", "people", "shelter", "million"]},
    {"question": "What search and rescue operations were conducted?",
     "keywords": ["rescue", "search", "teams", "survivors"]},
    {"question": "What was the food security situation in affected areas?",
     "keywords": ["food", "aid", "distribution", "assistance"]},
    {"question": "How did the earthquake affect Syrian refugees in Turkey?",
     "keywords": ["refugees", "syrian", "turkey", "displaced"]},
    {"question": "What was the scale of building destruction?",
     "keywords": ["buildings", "destroyed", "collapsed", "damage"]},
    {"question": "What international assistance was mobilised after the earthquake?",
     "keywords": ["international", "aid", "teams", "support"]},
    {"question": "What was the situation in Kahramanmaras after the earthquake?",
     "keywords": ["kahramanmaras", "earthquake", "damage", "affected"]},
    {"question": "What were the health and medical needs after the earthquake?",
     "keywords": ["health", "medical", "hospital", "injuries"]},
]

EXAMPLE_QUESTIONS = [
    "What was the humanitarian situation in Hatay?",
    "How many people were displaced across the affected provinces?",
    "What search and rescue operations were conducted?",
    "What was the food security situation in Gaziantep?",
    "How did the earthquake affect Syrian refugees?",
]


# ── chain (loaded once, shared across all reruns) ─────────────────────────────

@st.cache_resource(show_spinner="Loading RAG chain - this takes about a minute on first run…")
def _load_chain() -> RAGChain:
    download_if_missing()
    return RAGChain()


chain = _load_chain()


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_context(context: str) -> tuple[list[dict], str]:
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


def _mmi_color(mmi: float) -> str:
    if mmi >= 8:
        return "#b2182b"
    if mmi >= 7:
        return "#ef6548"
    if mmi >= 6:
        return "#fc8d59"
    if mmi >= 5:
        return "#fdbb84"
    if mmi >= 4:
        return "#fdd49e"
    return "#fff7ec"


@st.cache_resource
def load_spatial() -> dict | None:
    if not SPATIAL_PATH.exists():
        return None
    with SPATIAL_PATH.open("rb") as f:
        return pickle.load(f)


def _build_map() -> folium.Map:
    m = folium.Map(location=[37.0, 37.5], zoom_start=7, tiles=None, width="100%", height=540)
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
    shakemap_json = _json.loads(shakemap.to_json())
    folium.GeoJson(
        shakemap_json,
        name="ShakeMap intensity",
        style_function=lambda feat: {
            "fillColor": _mmi_color(float(
                feat["properties"].get("mmi")
                or feat["properties"].get("PARAMVALUE")
                or feat["properties"].get("value")
                or 0
            )),
            "color": "#333",
            "weight": 0.6,
            "fillOpacity": 0.65,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=list(shakemap_json["features"][0]["properties"].keys()) if shakemap_json["features"] else [],
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

st.set_page_config(page_title="GeoIntel RAG", page_icon="🌍", layout="wide")

st.title("🌍 GeoIntel RAG")
st.caption("Natural-language intelligence over the 2023 Turkey-Syria earthquake response corpus.")

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    with st.expander("What is this?"):
        st.markdown(
            "- In February 2023, a 7.8-magnitude earthquake hit southern Turkey and northern Syria\n"
            "- Over 50,000 people were killed and millions were displaced\n"
            "- This tool lets you ask questions in plain English about what happened\n"
            "- Answers are drawn from hundreds of humanitarian situation reports published during the response by UN agencies and NGOs on the ground"
        )
    with st.expander("What does it do?"):
        st.markdown(
            '- Type a question like *"What was the situation in Hatay?"* or *"How many people were displaced?"*\n'
            "- The system finds the most relevant passages from those reports\n"
            "- It does not invent anything - every answer is built from real documents\n"
            "- You can see exactly which passages were used"
        )
    with st.expander("What does it show?"):
        st.markdown(
            "- Next to the answer, an interactive map shows the earthquake's intensity zones - how hard each area was shaken\n"
            "- It also shows the locations of over 100,000 destroyed buildings, traced from satellite imagery by volunteer mappers\n"
            "- You can see which areas were hit hardest and read what aid organisations reported from the ground"
        )

# ── tabs ──────────────────────────────────────────────────────────────────────

tab_ask, tab_eval, tab_sources, tab_arch = st.tabs(["Ask", "Evaluate", "Data Sources", "Architecture"])

# ── Ask tab ───────────────────────────────────────────────────────────────────

with tab_ask:
    col_qa, col_map = st.columns([1, 1], gap="large")

    with col_qa:
        st.subheader("Ask a question")

        # ── model picker ──────────────────────────────────────────────────────
        st.caption(
            "GPT-OSS 20B is faster. GPT-OSS 120B gives longer, more detailed answers. "
            "Toggle compare mode to run both at once and see the results side by side."
        )
        compare_mode = st.toggle("Compare both models", value=False, key="_compare_mode")
        is_compare = compare_mode

        if is_compare:
            active_model_id = None
            model_a_label = list(AVAILABLE_MODELS.keys())[0]
            model_b_label = list(AVAILABLE_MODELS.keys())[1]
        else:
            active_model_label = st.selectbox(
                "Model",
                list(AVAILABLE_MODELS.keys()),
                index=0,
                key="_active_model",
            )
            active_model_id = AVAILABLE_MODELS[active_model_label]
            model_a_label = None
            model_b_label = None

        st.divider()

        st.caption(
            "❓ **Ready-made** - pick from a list of pre-written questions.  \n"
            "📍 **By province** - select a location to get a question about it.  \n"
            "✏️ **Your own** - type whatever you want to ask.  \n"
            "Choosing one locks the other two. Use ✕ inside the selector to clear it."
        )

        if "question" not in st.session_state:
            st.session_state.question = ""
        if "_input_mode" not in st.session_state:
            st.session_state._input_mode = None

        def _apply_example():
            val = st.session_state._example_select
            if val is not None:
                st.session_state.question = val
                st.session_state._input_mode = "example"
                st.session_state._province_select = None
            else:
                st.session_state._input_mode = None
                st.session_state.question = ""

        def _apply_province():
            val = st.session_state._province_select
            if val in PROVINCE_QUESTIONS:
                st.session_state.question = PROVINCE_QUESTIONS[val]
                st.session_state._input_mode = "province"
                st.session_state._example_select = None
            else:
                st.session_state._input_mode = None
                st.session_state.question = ""

        def _on_question_change():
            if st.session_state.question.strip():
                st.session_state._input_mode = "manual"
                st.session_state._example_select = None
                st.session_state._province_select = None
            else:
                st.session_state._input_mode = None

        mode = st.session_state._input_mode

        st.selectbox(
            "❓ Ready-made questions",
            EXAMPLE_QUESTIONS,
            index=None,
            placeholder="Choose an option",
            key="_example_select",
            on_change=_apply_example,
            disabled=(mode in ("province", "manual")),
        )

        st.selectbox(
            "📍 Focus on a province",
            list(PROVINCE_QUESTIONS.keys()),
            index=None,
            placeholder="Choose an option",
            key="_province_select",
            on_change=_apply_province,
            disabled=(mode in ("example", "manual")),
        )

        question = st.text_area(
            "✏️ Write your own",
            key="question",
            placeholder="Type your question here…",
            height=100,
            on_change=_on_question_change,
            disabled=(mode in ("example", "province")),
        )

        top_k = st.slider(
            "Chunks to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="How many text passages to send to the LLM as context.",
        )

        if "history" not in st.session_state:
            st.session_state.history = []

        submit = st.button("Ask", type="primary", use_container_width=True)

        def _stream_tokens(q: str, k: int):
            for event_type, data in chain.stream(q, top_k=k, model_id=active_model_id):
                if event_type == "context":
                    st.session_state._stream_context = data
                elif event_type == "token":
                    yield data

        if submit and question.strip():
            q = question.strip()

            if compare_mode:
                def _query_model(model_label: str) -> dict:
                    model_id = AVAILABLE_MODELS[model_label]
                    t0 = time.perf_counter()
                    try:
                        result = chain.run(q, top_k=top_k, model_id=model_id)
                        return {
                            "label": model_label,
                            "model_id": model_id,
                            "answer": result["answer"],
                            "context": result["context"],
                            "latency_ms": round((time.perf_counter() - t0) * 1000),
                            "error": None,
                        }
                    except Exception as exc:
                        return {
                            "label": model_label,
                            "model_id": model_id,
                            "answer": "",
                            "context": "",
                            "latency_ms": 0,
                            "error": str(exc),
                        }

                with st.spinner("Running both models in parallel…"):
                    with ThreadPoolExecutor(max_workers=2) as ex:
                        fut_a = ex.submit(_query_model, model_a_label)
                        fut_b = ex.submit(_query_model, model_b_label)
                        st.session_state._comparison = [fut_a.result(), fut_b.result()]

            else:
                st.session_state._comparison = None
                st.session_state._stream_context = ""
                st.markdown("### Answer")
                t0 = time.perf_counter()
                try:
                    answer = st.write_stream(_stream_tokens(q, top_k))
                except Exception as e:
                    st.error(f"Error: {e}")
                    answer = ""

                latency_ms = round((time.perf_counter() - t0) * 1000)
                if answer:
                    st.caption(f"Latency: {latency_ms} ms")
                    st.session_state.history.append({
                        "question": q,
                        "answer": answer,
                        "latency_ms": latency_ms,
                    })

                ctx = st.session_state.get("_stream_context", "")
                if ctx:
                    with st.expander("Retrieved context passages"):
                        chunks, spatial_txt = _parse_context(ctx)
                        for chunk in chunks:
                            with st.container(border=True):
                                st.markdown(
                                    f"**[{chunk['index']}] {chunk['title']}**"
                                    + (f"  `{chunk['date']}`" if chunk["date"] else "")
                                )
                                st.caption(chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else ""))
                        if spatial_txt:
                            st.markdown("**Geospatial context**")
                            st.info(spatial_txt)

        elif submit:
            st.warning("Please enter a question first.")

        if len(st.session_state.history) > 1:
            st.divider()
            col_hist_hdr, col_hist_export, col_hist_clear = st.columns([3, 1, 1])
            col_hist_hdr.markdown("#### Session history")
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=["question", "answer", "latency_ms"])
            writer.writeheader()
            writer.writerows(st.session_state.history)
            col_hist_export.download_button(
                "Export CSV",
                data=buf.getvalue(),
                file_name="geointel_session.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if col_hist_clear.button("Clear all", use_container_width=True, help="Delete all history"):
                st.session_state.history = []
                st.rerun()
            for idx, item in enumerate(reversed(st.session_state.history[:-1])):
                real_idx = len(st.session_state.history) - 2 - idx
                col_q, col_del = st.columns([8, 1])
                with col_del:
                    if st.button("✕", key=f"_del_hist_{real_idx}", help="Remove this entry", use_container_width=True):
                        st.session_state.history.pop(real_idx)
                        st.rerun()
                with col_q:
                    with st.expander(item["question"][:80]):
                        st.markdown(item["answer"])
                        st.caption(f"Latency: {item['latency_ms']} ms")

    with col_map:
        st.subheader("Affected area - ShakeMap + destroyed buildings")
        spatial = load_spatial()
        if spatial is None:
            st.info("Map data is downloading - it will appear after the first query.")
        else:
            n_buildings = len(spatial["buildings"])
            n_contours = len(spatial["shakemap"])
            st.caption(f"{n_contours} intensity contours · {n_buildings:,} destroyed buildings recorded")
            if "_folium_map_html_v4" not in st.session_state:
                st.session_state._folium_map_html_v4 = _build_map()._repr_html_()
            st_components.html(st.session_state._folium_map_html_v4, height=560, scrolling=False)

    if compare_mode and st.session_state.get("_comparison"):
        st.divider()
        st.subheader("Model comparison")
        results = st.session_state._comparison
        col_a, col_b = st.columns(2, gap="large")
        for col, r in zip([col_a, col_b], results):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{r['label']}**")
                    st.caption(f"`{r['model_id']}`")
                    if r["error"]:
                        st.error(r["error"])
                    else:
                        st.markdown(r["answer"])
                        st.caption(f"Latency: {r['latency_ms']} ms")
                        if r["context"]:
                            with st.expander("Context passages"):
                                chunks, _ = _parse_context(r["context"])
                                for chunk in chunks:
                                    with st.container(border=True):
                                        st.markdown(
                                            f"**[{chunk['index']}] {chunk['title']}**"
                                            + (f"  `{chunk['date']}`" if chunk["date"] else "")
                                        )
                                        st.caption(chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else ""))

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

# ── Eval tab ──────────────────────────────────────────────────────────────────

with tab_eval:
    import pandas as pd

    st.subheader("RAG Evaluation")

    eval_model_label = st.selectbox(
        "Model to evaluate",
        list(AVAILABLE_MODELS.keys()),
        key="_eval_model_select",
    )
    run_eval_btn = st.button("Run evaluation", type="primary", key="_run_eval")

    if run_eval_btn:
        model_id = AVAILABLE_MODELS[eval_model_label]
        eval_results = []
        progress = st.progress(0, text="Starting…")

        for i, case in enumerate(EVAL_TEST_CASES):
            progress.progress(
                i / len(EVAL_TEST_CASES),
                text=f"Question {i + 1}/{len(EVAL_TEST_CASES)}: {case['question'][:60]}…",
            )
            t0 = time.perf_counter()
            try:
                result = chain.run(case["question"], top_k=5, model_id=model_id)
                answer = result["answer"]
                context = result["context"]
            except Exception as exc:
                answer = f"[Error: {exc}]"
                context = ""
            latency_ms = round((time.perf_counter() - t0) * 1000)

            answer_lower = answer.lower()
            context_lower = context.lower()
            found_in_answer = [kw for kw in case["keywords"] if kw in answer_lower]
            found_in_context = [kw for kw in case["keywords"] if kw in context_lower]

            eval_results.append({
                "question": case["question"],
                "answer_recall": round(len(found_in_answer) / len(case["keywords"]), 3),
                "context_recall": round(len(found_in_context) / len(case["keywords"]), 3),
                "answer_length": len(answer.split()),
                "keywords_in_answer": ", ".join(found_in_answer),
                "keywords_in_context": ", ".join(found_in_context),
                "latency_ms": latency_ms,
                "answer": answer,
            })

        progress.progress(1.0, text="Done.")
        st.session_state._eval_results = eval_results
        st.session_state._eval_model_label = eval_model_label

    if st.session_state.get("_eval_results"):
        results = st.session_state._eval_results

        mean_answer_recall = sum(r["answer_recall"] for r in results) / len(results)
        mean_context_recall = sum(r["context_recall"] for r in results) / len(results)
        mean_length = sum(r["answer_length"] for r in results) / len(results)
        mean_latency = sum(r["latency_ms"] for r in results) / len(results)

        st.divider()
        st.caption(
            "**Answer recall** - Keywords found in the answer. (Did the LLM say the right things?)  \n"
            "**Context recall** - Keywords found in the retrieved passages. (Did retrieval find the right chunks?)  \n"
            "**Answer length** - Word count, a proxy for response depth."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Answer recall", f"{mean_answer_recall:.0%}")
        m2.metric("Context recall", f"{mean_context_recall:.0%}")
        m3.metric("Avg answer length", f"{mean_length:.0f} words")
        m4.metric("Mean latency", f"{mean_latency:.0f} ms")

        df = pd.DataFrame(results)
        df.index = [f"Q{i+1}" for i in range(len(df))]

        st.bar_chart(df[["answer_recall", "context_recall"]], height=220)

        st.divider()
        st.dataframe(
            df[[
                "question", "answer_recall", "context_recall",
                "answer_length", "keywords_in_answer", "latency_ms",
            ]].rename(columns={
                "question": "Question",
                "answer_recall": "Answer recall",
                "context_recall": "Context recall",
                "answer_length": "Length (words)",
                "keywords_in_answer": "Keywords in answer",
                "latency_ms": "Latency (ms)",
            }),
            use_container_width=True,
            hide_index=False,
        )

        with st.expander("Full answers"):
            for i, r in enumerate(results, 1):
                st.markdown(f"**Q{i}. {r['question']}**")
                st.markdown(r["answer"])
                st.divider()

# ── Architecture tab ──────────────────────────────────────────────────────────

with tab_arch:
    st.subheader("System Architecture")
    st.markdown(
        "GeoIntel RAG is split into two phases: an offline ingestion pipeline that builds "
        "the knowledge base, and an online query pipeline that answers questions in real time."
    )

    # ── Pipeline diagram ──────────────────────────────────────────────────────
    st.graphviz_chart("""
    digraph {
        rankdir=LR
        graph [fontname="Helvetica" bgcolor="transparent" pad="0.6" nodesep="0.7" ranksep="1.2"]
        node  [fontname="Helvetica" fontsize=14 shape=box style="rounded,filled" margin="0.3,0.2" width=2]
        edge  [fontname="Helvetica" fontsize=12 color="#555555"]

        subgraph cluster_offline {
            label="Offline - ingestion"
            style=dashed color="#aaaaaa" fontcolor="#444444" fontsize=15

            rw      [label="ReliefWeb API\n(situation reports)"  fillcolor="#dbeafe" color="#93c5fd"]
            ocha    [label="OCHA\n(key figures)"                 fillcolor="#dbeafe" color="#93c5fd"]
            chunker [label="Chunker\n512 tokens · 64 overlap"    fillcolor="#fef9c3" color="#fcd34d"]
            emb_off [label="Embedder\nall-MiniLM-L6-v2"          fillcolor="#fef9c3" color="#fcd34d"]
            faiss   [label="FAISS\nvector index"                  fillcolor="#d1fae5" color="#6ee7b7"]
            hfhub   [label="HuggingFace Hub\ndata storage"        fillcolor="#ede9fe" color="#c4b5fd"]

            shk     [label="USGS ShakeMap\n(GeoJSON)"             fillcolor="#dbeafe" color="#93c5fd"]
            osm     [label="HOT OSM\n(buildings)"                 fillcolor="#dbeafe" color="#93c5fd"]
            spatial [label="Spatial processor\nGeoDataFrame"      fillcolor="#d1fae5" color="#6ee7b7"]

            rw   -> chunker -> emb_off -> faiss -> hfhub
            ocha -> chunker
            shk  -> spatial -> hfhub
            osm  -> spatial
        }

        subgraph cluster_online {
            label="Online - query"
            style=dashed color="#aaaaaa" fontcolor="#444444" fontsize=15

            user    [label="User question"                  fillcolor="#fee2e2" color="#fca5a5"]
            emb_on  [label="Embedder\n(same model)"         fillcolor="#fef9c3" color="#fcd34d"]
            retr    [label="Retriever\ntop-k chunks"         fillcolor="#d1fae5" color="#6ee7b7"]
            enrich  [label="Spatial enrichment\n+ province info" fillcolor="#d1fae5" color="#6ee7b7"]
            groq    [label="Groq API\nGPT-OSS 20B / 120B"         fillcolor="#fed7aa" color="#fb923c"]
            answer  [label="Streamed answer\n+ Folium map"       fillcolor="#fee2e2" color="#fca5a5"]

            user -> emb_on -> retr -> enrich -> groq -> answer
        }

        hfhub -> retr [style=dashed label="downloaded at startup" fontcolor="#666666" fontsize=12]
    }
    """, use_container_width=True)

    st.divider()

    # ── Component cards ───────────────────────────────────────────────────────
    st.markdown("### Components")

    CARDS = [
        (
            "Data ingestion",
            "Raw documents are pulled from ReliefWeb (situation reports) and OCHA (key figures) "
            "via their public APIs. Each document is split into 512-token chunks with 64-token "
            "overlap to preserve context at boundaries.",
        ),
        (
            "Geospatial processing",
            "USGS ShakeMap GeoJSON contours and HOT OSM destroyed-building footprints are loaded "
            "into GeoDataFrames. Province centroids are matched to ShakeMap intensity zones, "
            "producing a spatial context string appended to each query.",
        ),
        (
            "Embedder & vector store",
            "Chunks are encoded with all-MiniLM-L6-v2 (sentence-transformers) into 384-dimensional "
            "vectors and indexed in FAISS for fast cosine similarity search. The index, chunks, and "
            "spatial data are stored on HuggingFace Hub and downloaded on first startup.",
        ),
        (
            "Retriever",
            "At query time the user's question is embedded with the same model and the top-k nearest "
            "chunks are retrieved from FAISS. Spatial enrichment is appended - ShakeMap intensity "
            "and affected provinces for the query region.",
        ),
        (
            "LLM - Groq API",
            "Retrieved context is passed to GPT-OSS 20B (fast) or GPT-OSS 120B (quality) "
            "via Groq. Answers stream token-by-token. Compare mode runs both models in "
            "parallel using a thread pool.",
        ),
        (
            "Deployment",
            "The app runs on Streamlit Cloud (free tier, ~1 GB RAM). Code lives on GitHub and "
            "deploys automatically on every push to main. Data files (~49 MB) are fetched from "
            "HuggingFace Hub at startup. MLflow tracks every query: model, latency, context length.",
        ),
    ]

    for title, body in CARDS:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(body)
