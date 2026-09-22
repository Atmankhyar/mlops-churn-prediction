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


def _native_estimator(model):
    """Return the estimator underneath an MLflow pyfunc wrapper, if reachable.

    MLflow wraps sklearn and xgboost models in a pyfunc whose ``predict``
    delegates to the estimator's ``predict`` — which for a classifier returns
    hard 0/1 class labels, not probabilities. To report a real churn
    probability we need ``predict_proba``, so unwrap to the estimator itself.
    Returns None when no probability-capable estimator can be reached, in
    which case the caller falls back to ``model.predict``.
    """
    if hasattr(model, "predict_proba"):
        return model

    impl = getattr(model, "_model_impl", None)
    if impl is None:
        return None

    for attr in ("sklearn_model", "xgb_model"):
        inner = getattr(impl, attr, None)
        if inner is not None and hasattr(inner, "predict_proba"):
            return inner

    return None


def predict_churn(
    model,
    features: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return predictions + probabilities.

    Args:
        model: Loaded MLflow pyfunc model (or a bare sklearn-style estimator).
        features: DataFrame with the same columns used during training.
        threshold: Probability at or above which a customer is flagged churned.

    Returns:
        Tuple of (binary_predictions, churn_probabilities).
    """
    estimator = _native_estimator(model)

    if estimator is not None:
        proba = estimator.predict_proba(features)
    else:
        # Last resort: the wrapper only exposes predict(). The values may be
        # class labels rather than probabilities.
        proba = model.predict(features)

    proba = np.asarray(getattr(proba, "values", proba))

    if proba.ndim == 2 and proba.shape[1] > 1:
        churn_prob = proba[:, 1]
    else:
        churn_prob = proba.ravel()

    churn_prob = churn_prob.astype(float)
    predictions = (churn_prob >= threshold).astype(int)
    return predictions, churn_prob
