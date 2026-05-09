# Amazon Product Recommendation & Review Intelligence System

---

## 1. Executive Summary

This project implements an end-to-end Big Data pipeline and machine learning ecosystem designed to process, analyze, and generate recommendations from the **McAuley-Lab Amazon Reviews 2023** dataset. The primary objective is to demonstrate a highly scalable architecture capable of digesting large-scale JSONL data, performing distributed feature engineering, and executing a robust model tournament. 

The resulting system features a fully functional recommendation engine powered by **Alternating Least Squares (ALS)**, sentiment and text classification models (Logistic Regression, Random Forest, Naive Bayes), and unsupervised clustering (K-Means). The entire experiment lifecycle is tracked via **MLflow**, and the finalized models and data artifacts are served through a high-performance **FastAPI** backend and an interactive **Streamlit** dashboard.

---

## 2. System Architecture

The architecture is designed to handle massive volumes of raw data through a distributed processing engine (Apache Spark) and persist intermediate states in columnar format (Parquet) for efficient downstream ML tasks.

```mermaid
graph TD
    A[Raw JSONL Data] -->|PySpark Ingestion| B(Ingested Parquet)
    B -->|PySpark Preprocessing| C(Clean Parquet)
    C -->|Spark MLlib| D{Feature Engineering}
    D -->|NLP / Embeddings| E(Engineered Features Parquet)
    
    E --> F[Model Tournament]
    F -->|ALS Recommender| G[MLflow Tracking]
    F -->|LR / RF / NB Classifiers| G
    F -->|K-Means Clustering| G
    
    G -->|Model Registry| H((Champion Model Export))
    
    C --> I[Streamlit Dashboard]
    H --> J[FastAPI Backend]
    I <-->|API Requests| J
```

### 2.1 Storage Layers
- **Raw JSONL**: The raw, unprocessed text and metadata directly from the source.
- **Ingested Parquet**: Initial conversion of JSONL to Parquet for faster subsequent reads and reduced storage footprint.
- **Clean Parquet**: Null values removed, schemas enforced, and noisy data filtered. This acts as the source of truth for the Streamlit dashboard analytics.
- **Engineered Features**: Features derived from text (TF-IDF/CountVectorizer) and encoded categorical variables, utilized by the classification and clustering models.

---

## 3. Data Processing & Feature Engineering Pipeline

The data pipeline is organized sequentially across multiple Jupyter Notebooks to enforce modularity and reproducibility.

### 3.1 Data Ingestion and EDA (`01_Data_Ingestion_and_EDA.ipynb`)
- **Action**: Reads massive JSONL streams using PySpark.
- **Outcome**: Outputs the initial `amazon_reviews.parquet` directory. It also includes exploratory data analysis (EDA) to understand distributions of ratings, missing values, and review lengths.

### 3.2 Data Preprocessing and Optimization (`02_Data_Preprocessing_and_Optimization.ipynb`)
- **Action**: Cleanses the data by handling null values, deduplicating records, and casting data types correctly.
- **Optimization**: Writes the clean data to a compressed Parquet format (`clean_data.parquet`), avoiding memory bottlenecks commonly associated with converting large Spark DataFrames to Pandas.

### 3.3 Feature Engineering (`03_Feature_Engineering_Pipeline.ipynb`)
- **Action**: Constructs the `Spark ML` pipelines.
- **NLP Processing**: Extracts the `text` column (review body) to generate Natural Language Processing (NLP) features. This involves tokenization, stop-word removal, and vectorization (e.g., TF-IDF).
- **Categorical Encoding**: Transforms string-based user IDs and product IDs into numeric indices required by the ALS algorithm.
- **Outcome**: Saves the engineered dataset and the trained `Spark ML` feature pipeline for reuse.

---

## 4. Machine Learning & Model Tournament

The project utilizes a "Tournament" approach to evaluate multiple model architectures simultaneously (`04_Model_Tournament_and_MLflow.ipynb`).

### 4.1 Collaborative Filtering (Recommendation)
- **Model**: Alternating Least Squares (ALS).
- **Mechanism**: Matrix factorization technique suited for implicit/explicit feedback. It decomposes the user-item interaction matrix into lower-dimensional dense vectors to predict missing user ratings for products.
- **Evaluation**: RMSE (Root Mean Square Error) and precision metrics.

### 4.2 Classification Models (Sentiment & Review Intelligence)
- Evaluated models: **Logistic Regression (LR)**, **Random Forest (RF)**, and **Naive Bayes (NB)**.
- **Purpose**: To classify the sentiment or helpfulness of a review based on the engineered NLP features.

### 4.3 Unsupervised Clustering
- **Model**: **K-Means Clustering**.
- **Purpose**: Groups similar reviews or products based on textual embeddings and metadata to discover hidden patterns.

### 4.4 Experiment Tracking (MLflow)
- Every run within the tournament is logged using **MLflow**.
- Metrics, hyperparameters, and model artifacts are tracked.
- **Notebook 05 (`05_Model_Registry_and_Export.ipynb`)** evaluates the MLflow registry, tags the best-performing ALS model as the `@champion`, and exports it for production serving (e.g., `exported_models/als_champion`).

---

## 5. Deployment & User Interface

### 5.1 FastAPI Backend (`fastapi_app.py`)
Provides RESTful endpoints for scalable model serving and data querying.
- `GET /stats`: Retrieves aggregate dataset statistics from the clean Parquet files.
- `GET /users/{user_id}/top-products`: Returns the highest-rated products for a specific user.
- `POST /predict_sentiment`: Accepts raw review text and returns a sentiment score and confidence metric, acting as a lightweight NLP serving layer.

### 5.2 Streamlit Dashboard (`app.py`)
A comprehensive, interactive UI for business intelligence and user personalization.
- **Business Dashboard**: Visualizes KPI metrics, storage layer comparisons, rating distributions, and review volume over time using Altair charts.
- **Customer Personalization**: Displays a specific user's historical review timeline, rating distribution, and top-rated products.
- **Review Intelligence**: Interfaces with the FastAPI backend (or uses robust keyword heuristics as a fallback) to provide real-time sentiment analysis on arbitrary text input.

---

## 6. Technologies Used

| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Big Data Engine** | Apache Spark / PySpark (Java 17) |
| **Data Storage** | Parquet, PyArrow |
| **Machine Learning** | Spark MLlib |
| **Experiment Tracking**| MLflow |
| **Backend API** | FastAPI, Uvicorn |
| **Frontend/Dashboard** | Streamlit, Altair, Pandas |
| **Data Source** | Hugging Face (McAuley-Lab/Amazon-Reviews-2023) |

---

## 7. Setup and Execution

To run the full pipeline:

1. **Environment Setup**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
2. **Execute Notebooks sequentially**: Run Notebooks `01` through `05` to ingest data, engineer features, train models, and export the champion model.
3. **Start the API**:
    ```bash
    uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
    ```
4. **Launch the Dashboard**:
    ```bash
    streamlit run app.py
    ```

---

*This report documents the design, implementation, and deployment of a scalable recommendation and analytics pipeline, fulfilling the requirements for advanced data engineering and machine learning model deployment methodologies.*
