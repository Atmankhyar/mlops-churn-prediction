"""Model loading and inference from MLflow Model Registry."""
from __future__ import annotations

import os
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def load_model_from_registry(
    model_name: Optional[str] = None,
    stage: str = "Production",
):
    """Load a model from the MLflow Model Registry.

    Args:
        model_name: Registered model name. Defaults to MLFLOW_MODEL_NAME env var.
        stage: Registry stage — 'Production', 'Staging', or 'None'.

    Returns:
        Loaded sklearn-compatible model object.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    name = model_name or os.environ.get("MLFLOW_MODEL_NAME", "churn-predictor")
    model_uri = f"models:/{name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def predict_churn(
    model,
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return predictions + probabilities.

    Args:
        model: Loaded MLflow pyfunc model.
        features: DataFrame with the same columns used during training.

    Returns:
        Tuple of (binary_predictions, churn_probabilities).
    """
    proba = model.predict(features)
    if hasattr(proba, "values"):
        proba = proba.values
    if proba.ndim == 2:
        churn_prob = proba[:, 1]
    else:
        churn_prob = proba
    predictions = (churn_prob >= 0.5).astype(int)
    return predictions, churn_prob
