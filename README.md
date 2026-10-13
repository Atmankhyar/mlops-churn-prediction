# End-to-End MLOps: Customer Churn Prediction

[![CI](https://github.com/Atmankhyar/mlops-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Atmankhyar/mlops-churn-prediction/actions/workflows/ci.yml)
[![CD](https://github.com/Atmankhyar/mlops-churn-prediction/actions/workflows/cd.yml/badge.svg)](https://github.com/Atmankhyar/mlops-churn-prediction/actions/workflows/cd.yml)

A **portfolio-grade, production-ready MLOps pipeline** for customer churn prediction, demonstrating the full lifecycle from raw data to a monitored REST API deployed on AWS.

---

## Architecture

```text
                       ┌──────────────┐
                       │   GitHub     │
                       └──────┬───────┘
                          CI/CD │
                              ▼
                       ┌──────────────┐
                       │   GitHub     │
                       │   Actions    │
                       └──────┬───────┘
                              │
                              ▼
┌─────────────┐        ┌──────────────┐
│ Data Sources│───────►│  Databricks  │
│ 100K cust.  │        │  PySpark     │
│ 1M+ tx      │        │  Delta Lake  │
└─────────────┘        └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  Snowflake   │
                       │  Data Warehouse│
                       │  RAW→ANALYTICS│
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │ ML Training  │
                       │ sklearn/XGB  │
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │    MLflow    │
                       │ Tracking +   │
                       │ Model Reg.   │
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │   AWS EC2    │
                       │   Docker     │
                       │   FastAPI    │
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │ Evidently AI │
                       │ Drift Monitor│
                       └──────────────┘
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Feature Engineering | PySpark, Databricks, Delta Lake |
| Data Warehouse | Snowflake (3-layer: RAW → ANALYTICS → ML) |
| ML Training | scikit-learn, XGBoost, imbalanced-learn |
| Experiment Tracking | MLflow (experiments + model registry) |
| Model Serving | FastAPI, Uvicorn |
| Containerization | Docker, Docker Compose |
| Cloud Hosting | AWS EC2 (t3.medium) + S3 |
| CI/CD | GitHub Actions |
| Drift Monitoring | Evidently AI |

---

## Project Metrics

| Metric | Value |
|---|---|
| Customers | 100,000 |
| Transactions | 1,000,000+ |
| Support interactions | 500,000+ |
| Features engineered | 9 |
| Test coverage | 73% (CI gate: 70%) |
| Best model AUC-ROC | *(fill in after training)* |
| Best model F1 | *(fill in after training)* |
| API latency (p99) | *(fill in after deployment)* |
| Databricks processing time | *(fill in after run)* |

Dataset sizes are the defaults of `scripts/generate_data.py` (synthetic data).
Model and latency figures are left blank until measured.

---

## Quickstart — Phase 1 (Local ML Pipeline)

### 1. Clone and set up environment

```bash
git clone https://github.com/Atmankhyar/mlops-churn-prediction.git
cd mlops-churn-prediction

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements-dev.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Generate synthetic dataset

```bash
python scripts/generate_data.py   # options: --customers --transactions --interactions
# Generates: data/raw/customers.csv (100K rows)
#            data/raw/transactions.csv (1M rows)
#            data/raw/support_interactions.csv (500K rows)
#            data/raw/churn_labels.csv
```

### 4. Start MLflow tracking server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

Open MLflow UI: http://localhost:5000

### 5. Run training

```bash
# Random Forest baseline
python scripts/train_local.py --model random_forest --version v1.0

# XGBoost
python scripts/train_local.py --model xgboost --version v1.0
```

### 6. Run tests

```bash
pytest tests/ -v --cov=src --cov=api
```

---

## Running the API Locally

```bash
# With Docker Compose (API + MLflow server)
docker compose up --build

# Or directly with uvicorn
uvicorn api.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

| Endpoint | Description |
|---|---|
| `GET /health` | Service and model-load status |
| `GET /model/info` | Registered model name, version and stage |
| `POST /predict` | Single churn prediction |
| `POST /predict/batch` | Batch predictions |

Every prediction is logged to `monitoring/inference_log.db` for drift monitoring.

> **Note:** `docker-compose.yml` mounts the local `mlflow.db` and `mlflow-artifacts/`
> (both git-ignored). A fresh clone must train a model first, or restore these files,
> before `/predict` can load a registered model.

### Example prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 6,
    "contract_type": "Month-to-month",
    "total_spending_3m": 287.5,
    "avg_monthly_charge": 95.8,
    "payment_failure_rate": 0.15,
    "transaction_count_3m": 3,
    "support_tickets_3m": 4,
    "avg_resolution_days": 7.2,
    "avg_sentiment": -0.6
  }'
```

Expected response:

```json
{
  "prediction": 1,
  "churn_probability": 0.87,
  "model_version": "2",
  "timestamp": "2026-09-18T20:00:00Z"
}
```

---

## Implementation Phases

| Phase | Status | Description |
|---|---|---|
| 1 — Local ML | ✅ | Data generation, feature engineering, sklearn/XGBoost baseline |
| 2 — MLflow | ✅ | Experiment tracking, model registry (`churn-predictor`, 11 versions) |
| 3 — PySpark | ✅ | Distributed feature engineering (local Spark) + validation |
| 4 — Databricks | 🟡 | Notebooks and job definition written; cloud run not yet recorded |
| 5 — Snowflake | 🟡 | Schema, queries and `train_snowflake.py` written; not yet verified end to end |
| 6 — FastAPI + Docker | ✅ | Containerized model serving with inference logging |
| 7 — AWS EC2 | ⬜ | Production deployment |
| 8 — GitHub Actions | 🟡 | `ci.yml` and `cd.yml` written; repo not yet pushed, secrets not set |
| 9 — Evidently | ✅ | Drift monitoring runner (`scripts/run_drift_monitoring.py`) |

✅ done · 🟡 implemented, not fully verified · ⬜ not started

---

## Drift Monitoring

```bash
python scripts/run_drift_monitoring.py
```

Compares recently logged inference requests against the training reference data
using Evidently and writes an HTML report to `monitoring/reports/`. If fewer than
30 recent predictions exist, monitoring is skipped rather than reporting on
insufficient data. Settings live in `configs/monitoring_config.yaml`.

---

## Repository Structure

```text
mlops-churn-prediction/
├── .github/workflows/    # CI/CD: ci.yml, cd.yml
├── api/                  # FastAPI app (main.py, schemas.py)
├── configs/              # model_config.yaml, monitoring_config.yaml
├── data/raw/             # Generated CSVs (git-ignored)
├── databricks/           # PySpark notebooks + job definitions
├── mlflow/               # MLproject
├── monitoring/reports/   # Evidently HTML reports (git-ignored)
├── notebooks/            # EDA + training notebooks
├── scripts/              # generate_data, train_local, train_snowflake,
│                         # run_drift_monitoring, validate_drift_monitoring, deploy.sh
├── snowflake/            # schema.sql + SQL queries
├── src/
│   ├── data/             # ingestion.py, cleaning.py
│   ├── features/         # engineering.py, spark_engineering.py, validate_spark_output.py
│   ├── models/           # train.py, evaluate.py, predict.py
│   └── monitoring/       # drift.py (Evidently), inference_logger.py
├── tests/                # pytest test suite
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## GitHub Secrets Required (for CI/CD)

| Secret | Description |
|---|---|
| `EC2_HOST` | Public IP of your EC2 instance |
| `EC2_SSH_KEY` | PEM private key for EC2 SSH access |
| `REPO_URL` | HTTPS URL of this GitHub repository |
| `MLFLOW_MODEL_NAME` | e.g., `churn-predictor` |
| `SNOWFLAKE_ACCOUNT` | Snowflake account identifier |
| `SNOWFLAKE_USER` | Snowflake username |
| `SNOWFLAKE_PASSWORD` | Snowflake password |
| `S3_ARTIFACT_BUCKET` | S3 bucket name for MLflow artifacts |

---

## License

MIT
