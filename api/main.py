"""FastAPI application for the churn prediction model."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.models.predict import load_model_from_registry, predict_churn

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Churn Prediction API",
    description="End-to-End MLOps — Customer Churn Prediction Service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model singleton (loaded at startup)
# ---------------------------------------------------------------------------
_model = None
_model_version: str = "unknown"


@app.on_event("startup")
def load_model() -> None:
    global _model, _model_version
    try:
        _model = load_model_from_registry(stage="Production")
        if hasattr(_model, "metadata") and hasattr(_model.metadata, "run_id"):
            _model_version = _model.metadata.run_id[:8]
        logger.info("Model loaded successfully (version=%s)", _model_version)
    except Exception as exc:
        logger.warning("Could not load model from registry: %s", exc)
        logger.warning("API will start but /predict returns 503 until a model is registered.")


def _get_model():
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Ensure a model is registered in MLflow as 'Production'.",
        )
    return _model


def _request_to_dataframe(req: PredictRequest) -> pd.DataFrame:
    """Convert a PredictRequest into a single-row feature DataFrame."""
    data = req.model_dump(exclude={"customer_id"})
    contract = data.pop("contract_type")
    for ct in ["Month-to-month", "One year", "Two year"]:
        data[f"contract_type_{ct}"] = int(contract == ct)
    return pd.DataFrame([data])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=_model_version,
        environment=os.getenv("ENVIRONMENT", "production"),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name=os.getenv("MLFLOW_MODEL_NAME", "churn-predictor"),
        model_version=_model_version,
        stage="Production",
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = _get_model()
    features = _request_to_dataframe(req)
    try:
        preds, probas = predict_churn(model, features)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return PredictResponse(
        customer_id=req.customer_id,
        prediction=int(preds[0]),
        churn_probability=round(float(probas[0]), 4),
        model_version=_model_version,
        timestamp=datetime.now(timezone.utc),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    model = _get_model()
    responses = []
    for customer in req.customers:
        features = _request_to_dataframe(customer)
        try:
            preds, probas = predict_churn(model, features)
            responses.append(PredictResponse(
                customer_id=customer.customer_id,
                prediction=int(preds[0]),
                churn_probability=round(float(probas[0]), 4),
                model_version=_model_version,
                timestamp=datetime.now(timezone.utc),
            ))
        except Exception as exc:
            logger.error("Batch error for %s: %s", customer.customer_id, exc)
            raise HTTPException(status_code=500, detail=str(exc))

    return BatchPredictResponse(predictions=responses, total=len(responses))
