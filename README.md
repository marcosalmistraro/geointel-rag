---
title: geo_intel_rag
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# 🌍 GeoIntel RAG

RAG-powered geospatial intelligence system over the 2023 Turkey-Syria earthquake corpus - combines FAISS vector search, ShakeMap enrichment, and Groq LLM inference with a live Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What is this?

In February 2023, a 7.8-magnitude earthquake hit southern Turkey and northern Syria. Over 50,000 people were killed and millions were displaced. This tool lets you ask questions in plain English about what happened and get answers drawn from the hundreds of humanitarian situation reports published during the response by UN agencies and NGOs on the ground.

## What does it do?

Type a question like *"What was the situation in Hatay?"* or *"How many people were displaced?"* and the system finds the most relevant passages from those reports, then uses an AI model to put together an answer from them. It does not invent anything - every answer is built from real documents, and you can see exactly which passages were used.

## What does it show?

Next to the answer, an interactive map shows the earthquake's intensity zones - how hard each area was shaken - and the locations of over 100,000 destroyed buildings, traced from satellite imagery by volunteer mappers after the earthquake. You can see which areas were hit hardest and read what aid organisations reported from the ground.

---

## Architecture

```
User query
    |
    v
Streamlit frontend (app.py)
    |
    +-- FAISS retriever (top-k chunks from ReliefWeb corpus)
    |
    +-- Spatial enricher (ShakeMap MMI + province info)
    |
    v
Groq API (GPT-OSS 20B / 120B)
    |
    v
Streamed answer + Folium map
```

Data is downloaded from HuggingFace Hub at startup. No separate backend - everything runs inside the Streamlit app.

---

## Data sources

| Source | Content |
|--------|---------|
| [ReliefWeb](https://reliefweb.int/) | Situation reports, Feb-May 2023 |
| [USGS ShakeMap via HDX](https://data.humdata.org/dataset/50d93259-2d49-4f84-85e6-3cd0aa03dfaa) | MMI intensity contours (M7.8 + M7.5) |
| [HOT OSM Destroyed Buildings](https://data.humdata.org/dataset/hotosm_tur_destroyed_buildings) | ~100k destroyed/damaged buildings |
| [OCHA Key Figures](https://data.humdata.org/dataset/turkiye-syria-earthquake-key-figures) | Key humanitarian figures |

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| LLM | GPT-OSS 20B / 120B via [Groq](https://groq.com) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector store | FAISS (flat L2 index) |
| Geospatial | GeoPandas, Shapely, Folium |
| Frontend | Streamlit |
| Data hosting | HuggingFace Hub |
| CI/CD | GitHub Actions |
| Fine-tuning | QLoRA (4-bit NF4) + LoRA adapters via `peft` + `trl` (notebook, not yet run) |

---

## Project structure

```
geointel_rag/
├── app.py                       # Self-contained Streamlit app (deployed)
├── rag/
│   ├── embedder.py              # Sentence-transformer wrapper
│   ├── vector_store.py          # FAISS index (save / load)
│   ├── retriever.py             # Top-k retrieval + spatial enrichment
│   └── chain.py                 # RAG chain -> Groq API
├── ingestion/
│   ├── downloaders/             # ReliefWeb, USGS ShakeMap, HOT OSM, OCHA
│   ├── loaders/                 # Text + spatial loaders
│   ├── chunker.py               # Sliding-window text chunker
│   └── pipeline.py              # End-to-end ingestion orchestrator
├── api/                         # FastAPI app (portfolio artifact, not deployed)
├── scripts/
│   └── download_from_hub.py     # Downloads data from HuggingFace Hub at startup
├── notebooks/
│   └── 01_finetune_colab.ipynb  # QLoRA fine-tuning on Colab (not yet run)
├── tests/                       # pytest unit + integration tests
├── config.py                    # Pydantic Settings (reads .env)
└── .github/workflows/ci.yml
```

---

## Quick start

### Prerequisites
- Python 3.11
- A free [Groq API key](https://console.groq.com)

### 1 - Clone and install

```bash
git clone https://github.com/marcosalmistraro/geointel-rag.git
cd geointel-rag
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 - Configure environment

```bash
cp .env.example .env   # then edit .env
```

```
GROQ_API_KEY=your_key_here
HF_TOKEN=your_hf_token   # needed only for the fine-tuning notebook
```

### 3 - Run the app

```bash
streamlit run app.py
# -> http://localhost:8501
```

Data files (~49 MB) are downloaded automatically from HuggingFace Hub on first run.

---

## Running tests

```bash
pytest tests/ -v
```

---

## Fine-tuning notebook

`notebooks/01_finetune_colab.ipynb` walks through QLoRA fine-tuning of Llama 3.1 on a synthetically generated earthquake QA dataset derived from the ingested corpus. Designed to run on a free Colab T4 GPU. Not yet run - fine-tuned adapter not integrated.

---

## License

MIT
