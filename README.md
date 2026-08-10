---
title: geo_intel_rag
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# GeoIntel RAG

Natural-language intelligence over the 2023 Turkey-Syria earthquake humanitarian response corpus, combining Retrieval-Augmented Generation (RAG) with geospatial map visualisation.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What it does

Ask a question in plain English — for example *"What was the humanitarian situation in Hatay?"* — and the system:

1. **Retrieves** the most relevant passages from ReliefWeb situation reports using FAISS vector search
2. **Enriches** the context with geospatial data (ShakeMap intensity, affected provinces)
3. **Generates** a grounded answer via Groq (Llama 3.1 8B Instant)
4. **Visualises** ShakeMap intensity contours and 100 k+ destroyed buildings on an interactive map

---

## Architecture

```
User query
    │
    ▼
Streamlit frontend  ──POST /query──►  FastAPI backend
                                            │
                                    ┌───────┴────────┐
                                    │                │
                              FAISS retriever    Spatial enricher
                              (top-k chunks)     (province → MMI)
                                    │                │
                                    └───────┬────────┘
                                            │
                                     Groq API
                                   (Llama 3.1 8B)
                                            │
                                       Answer + context
```

**Data sources**
| Source | Content |
|--------|---------|
| [ReliefWeb](https://reliefweb.int/) | Situation reports, Feb–May 2023 |
| [USGS ShakeMap via HDX](https://data.humdata.org/dataset/50d93259-2d49-4f84-85e6-3cd0aa03dfaa) | MMI intensity contours (M7.8 + M7.5) |
| [HOT OSM Destroyed Buildings](https://data.humdata.org/dataset/hotosm_tur_destroyed_buildings) | ~100 k destroyed/damaged buildings |
| [OCHA Key Figures](https://data.humdata.org/dataset/turkiye-syria-earthquake-key-figures) | Key humanitarian figures |

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| LLM | Llama 3.1 8B Instant via [Groq](https://groq.com) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector store | FAISS (flat L2 index) |
| Geospatial | GeoPandas, Shapely, Folium |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit + streamlit-folium |
| Tracking | MLflow |
| Container | Docker |
| CI/CD | GitHub Actions |
| Fine-tuning | QLoRA (4-bit NF4) + LoRA adapters via `peft` + `trl` |

---

## Project structure

```
geointel_rag/
├── ingestion/
│   ├── downloaders/        # ReliefWeb, USGS ShakeMap, HOT OSM, OCHA
│   ├── loaders/            # Text + spatial loaders
│   ├── chunker.py          # Sliding-window text chunker
│   └── pipeline.py         # End-to-end ingestion orchestrator
├── rag/
│   ├── embedder.py         # Sentence-transformer wrapper
│   ├── vector_store.py     # FAISS index (save / load)
│   ├── retriever.py        # Top-k retrieval + spatial enrichment
│   └── chain.py            # RAG chain → Groq API
├── api/
│   ├── main.py             # FastAPI app + lifespan
│   ├── schemas.py          # Pydantic request/response models
│   └── routes/             # /health  /query  /ingest
├── frontend/
│   └── app.py              # Streamlit two-panel UI
├── tracking/
│   └── mlflow_utils.py     # MLflow logging helpers
├── notebooks/
│   └── 01_finetune_colab.ipynb  # QLoRA fine-tuning on Colab
├── tests/                  # pytest unit + integration tests
├── config.py               # Pydantic Settings (reads .env)
├── Dockerfile
└── .github/workflows/ci.yml
```

---

## Quick start

### Prerequisites
- Python 3.11
- A free [Groq API key](https://console.groq.com)

### 1 — Clone and install

```bash
git clone https://github.com/marcosalmistraro/geointel-rag.git
cd geointel-rag
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Configure environment

```bash
cp .env.example .env   # then edit .env
```

```
GROQ_API_KEY=your_key_here
HF_TOKEN=your_hf_token   # needed only for the fine-tuning notebook
```

### 3 — Run the ingestion pipeline

Downloads raw data, chunks text, builds the FAISS index and saves spatial layers. Safe to re-run — skips the index build if it already exists.

```bash
python -m ingestion.downloaders.run_all   # fetch raw data (~5 min)
python -m ingestion.pipeline              # embed + index
```

### 4 — Start the API

```bash
uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

### 5 — Start the frontend

```bash
streamlit run frontend/app.py
# → http://localhost:8501
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — returns index size and model ID |
| `POST` | `/query` | Ask a question; returns answer, context, latency |
| `POST` | `/ingest` | Re-run the ingestion pipeline and hot-swap the chain |

Example query:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many people were displaced in Gaziantep?", "top_k": 5}'
```

---

## Running tests

```bash
pytest tests/ -v
```

38 tests covering chunking, embedding, vector store, retrieval, and all API endpoints.

---

## Fine-tuning notebook

`notebooks/01_finetune_colab.ipynb` walks through QLoRA fine-tuning of Llama 3.1 on a synthetically generated earthquake QA dataset derived from the ingested corpus. Designed to run on a free Colab T4 GPU.

---

## Docker

```bash
docker build -t geointel-rag .
docker run -p 8000:8000 --env-file .env geointel-rag
```

---

## License

MIT
