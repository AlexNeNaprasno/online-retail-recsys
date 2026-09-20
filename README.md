# Online Retail RecSys — End-to-End DE + DL Pipeline

[![CI](https://github.com/AlexNeNaprasno/online-retail-recsys/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexNeNaprasno/online-retail-recsys/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-compose-blue.svg)](https://docs.docker.com/compose/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end Data Engineering + Deep Learning pipeline for **next-item recommendation** on the [Online Retail dataset](https://www.kaggle.com/datasets/carrie1/ecommerce-data).

**Result:** GRU4Rec model reaches **HitRate@10 = 31.0%** on test vs **3.9%** for a popularity baseline — an **8× improvement**.

---

## TL;DR

| Component | What it does |
|---|---|
| **Data pipeline** | Kaggle CSV → cleaning → PostgreSQL → marts (`dim_customer`, `dim_product`, `fact_sales`) |
| **DL model** | GRU4Rec (PyTorch) trained on user purchase sequences |
| **Orchestration** | Airflow DAG runs the whole DE pipeline end-to-end |
| **Experiment tracking** | MLflow logs params, metrics, artifacts |
| **Serving** | FastAPI `/recommend` endpoint returns top-K next-item predictions |
| **Infra** | One `docker compose up -d` — 5 services: API, MLflow, Postgres, Airflow webserver, Airflow scheduler |

---

## Architecture

```text
                Kaggle CSV
                    │
                    ▼
        ┌───────────────────────┐
        │   Airflow DAG         │  retail_pipeline
        │  (clean → validate →  │
        │   Postgres → marts)   │
        └───────────┬───────────┘
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
  raw_retail   dim_customer   dim_product
  fact_sales
                    │
                    ▼
        ┌───────────────────────┐
        │  Sequence building    │  per-user, top-3000 SKU vocab
        │  train / val / test   │  (time-aware split)
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  GRU4Rec (PyTorch)    │  Embedding → GRU → Linear
        │  + Popularity baseline│  HitRate / NDCG / MRR
        └───────────┬───────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
    MLflow tracking      FastAPI /recommend
    (params, metrics,    (Docker, port 8000)
     artifacts, plots)