# Amazon Product Recommendation & Review Intelligence

End-to-end Big Data pipeline for the [McAuley-Lab Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset: **PySpark** ingestion and preprocessing, **Spark ML** feature engineering and model tournament (ALS recommender, classifiers, clustering), **MLflow** experiment tracking and model registry, plus a **FastAPI** service and **Streamlit** dashboard for analytics and lightweight sentiment scoring.

---

## What's in this repository

| Layer | Role |
|--------|------|
| **Notebooks 01–05** | Data ingestion → clean Parquet → engineered features → model tournament & MLflow → registry export |
| **`spark_write_helper.py`** | Reusable Spark writes to Parquet (used in preprocessing / feature notebooks) |
| **`fastapi_app.py`** | REST API: dataset stats, per-user top products, keyword-based sentiment (aligned with Streamlit fallback) |
| **`app.py`** | Single-file Streamlit app: sidebar navigation (Home, Business Dashboard, Customer personalization, Review intelligence) |

---

## Architecture (high level)

1. **Data:** Raw JSONL → PySpark → Parquet (`data/`)
2. **Features:** Spark ML pipelines; engineered features under `data/engineered_features.parquet` (Notebook 03)
3. **Models:** ALS + additional models (Notebook 04); metrics logged to **MLflow** (`mlruns/` locally)
4. **Registry:** Best ALS registered / exported via Notebook 05 (e.g. `exported_models/`)
5. **Serving:** **FastAPI** on port 8000 (optional for sentiment + consistent JSON); **Streamlit** reads cleaned Parquet for charts

---

## Prerequisites

- **Python 3.10+** (3.12 used in development)
- **Apache Spark / PySpark** (via `requirements.txt`)
- **Java 17** for local Spark (Spark 3.5 expects Java 17). On macOS:

  ```bash
  export JAVA_HOME=$(/usr/libexec/java_home -v 17)
  ```

  Using Java 11 often causes `JAVA_GATEWAY_EXITED` / `UnsupportedClassVersionError`.

- **Jupyter** or VS Code with Jupyter support for the notebooks

---

## Repository layout

```
Final_project/
├── 01_Data_Ingestion_and_EDA.ipynb
├── 02_Data_Preprocessing_and_Optimization.ipynb
├── 03_Feature_Engineering_Pipeline.ipynb
├── 04_Model_Tournament_and_MLflow.ipynb
├── 05_Model_Registry_and_Export.ipynb
├── app.py                 # Streamlit (all pages in one file)
├── fastapi_app.py         # FastAPI backend
├── spark_write_helper.py  # Parquet write helpers for Spark
├── requirements.txt
├── README.md
├── data/                  # Created locally (large JSONL / Parquet often gitignored)
│   ├── *.jsonl
│   ├── amazon_reviews.parquet/
│   └── amazon_clean.parquet/
├── mlruns/                # MLflow tracking (typically gitignored)
└── exported_models/       # Optional export from Notebook 05 (typically gitignored)
```

Paths assumed by **Streamlit** and **FastAPI**:

- Clean ratings / reviews: `data/amazon_clean.parquet/clean_data.parquet`
- Ingested raw Parquet folder: `data/amazon_reviews.parquet/`
- Raw JSONL glob: `data/*.jsonl`

---

## Setup

```bash
cd Final_project
python -m venv .venv

# Windows: .venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Dependencies include: `pyspark`, `mlflow`, `streamlit`, `altair`, `fastapi`, `uvicorn`, `pandas`, `numpy`, `pyarrow`, `requests`, `matplotlib`, `seaborn`, `pydantic`.

---

## Data download

1. Open [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
2. Download at least one category JSONL into `data/` (e.g. `data/reviews.jsonl`).
3. Point Notebook 01's ingestion path at your file.

---

## Running the pipeline (notebooks)

Run **in order**:

1. **`01_Data_Ingestion_and_EDA.ipynb`** — JSONL → Parquet under `data/` (e.g. `amazon_reviews.parquet/`).
2. **`02_Data_Preprocessing_and_Optimization.ipynb`** — Cleaning; writes `data/amazon_clean.parquet/clean_data.parquet` (prefer Parquet writes over huge `toPandas()`).
3. **`03_Feature_Engineering_Pipeline.ipynb`** — Reads clean Parquet; builds Spark ML pipeline; writes `data/engineered_features.parquet` and saves pipeline under `models/feature_pipeline` where configured.
4. **`04_Model_Tournament_and_MLflow.ipynb`** — ALS + classifiers + clustering; logs to MLflow.
5. **`05_Model_Registry_and_Export.ipynb`** — Registry / `@champion` / export (e.g. `exported_models/als_champion`).

Start MLflow UI before training if you want live tracking:

```bash
mlflow ui
```

Open `http://127.0.0.1:5000` (default).

---

## Streamlit dashboard

Requires **clean Parquet** at `data/amazon_clean.parquet/clean_data.parquet` for full charts.

```bash
streamlit run app.py
```

Sections:

- **Business Dashboard** — Metrics, storage layers (raw vs ingested vs clean), rating charts.
- **Customer personalization** — Per-`user_id` charts from Parquet.
- **Review intelligence** — Calls **`POST http://127.0.0.1:8000/predict_sentiment`** when FastAPI is up; otherwise the same keyword heuristic as `fastapi_app.py` (positive / negative word lists, net score, optional **Neutral**).

---

## FastAPI backend

```bash
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + dataset configured |
| GET | `/stats` | Aggregates from clean Parquet when available |
| GET | `/users/{user_id}/top-products` | Top star ratings per user |
| POST | `/predict_sentiment` | Body `{"text": "..."}` → sentiment + confidence |

Docs: `http://127.0.0.1:8000/docs`

---

## Git and large files

A `.gitignore` may exclude: `.venv/`, `__pycache__/`, `mlruns/`, `exported_models/`, `data/*.jsonl`, `data/**/*.parquet`. Commit code and notebooks; restore data locally per these steps.

---

## Troubleshooting

| Issue | What to try |
|--------|-------------|
| **`JAVA_GATEWAY_EXITED` / UnsupportedClassVersionError** | Use **Java 17** (`JAVA_HOME`). |
| **Spark OOM / `maxResultSize`** | Raise driver memory; write Parquet instead of collecting huge frames. |
| **Empty Streamlit charts** | Run Notebook 02 and confirm clean Parquet path exists. |
| **Sentiment uses fallback only** | Start FastAPI on port 8000. |
| **Port in use** | e.g. `uvicorn fastapi_app:app --port 8080` (and match Streamlit if you hardcode URLs). |

---

## Dataset license

Follow Hugging Face / dataset terms for **Amazon Reviews 2023**. This repo is pipeline code unless you add data locally.
