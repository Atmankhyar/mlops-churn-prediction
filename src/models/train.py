"""Model training with full MLflow tracking."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.models.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "model_config.yaml"


def load_config() -> dict:
    """Load model configuration from YAML."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def prepare_features(
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select feature columns and target, encode categoricals."""
    numerical_cols = config["features"]["numerical"]
    categorical_cols = config["features"]["categorical"]
    target = config["target"]

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    # Build final feature list (includes OHE columns)
    feature_cols = numerical_cols + [
        c for c in df.columns
        if any(c.startswith(cat + "_") for cat in categorical_cols)
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[target].astype(int)
    return X, y


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    params = config["random_forest"]
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> XGBClassifier:
    """Train an XGBoost classifier."""
    params = {k: v for k, v in config["xgboost"].items() if k != "eval_metric"}
    model = XGBClassifier(**params, eval_metric="auc")
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model


def run_training(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    dataset_version: str = "v1.0",
    tags: dict[str, Any] | None = None,
) -> str:
    """Full training run with MLflow tracking.

    Args:
        df: Feature DataFrame including the target column.
        model_type: 'random_forest' or 'xgboost'
        dataset_version: Version tag for reproducibility.
        tags: Additional MLflow run tags.

    Returns:
        MLflow run_id of the completed run.
    """
    config = load_config()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(config["experiment_name"])

    X, y = prepare_features(df, config)
    split_cfg = config["train_test_split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y if split_cfg["stratify"] else None,
    )

    run_name = f"{model_type}-{dataset_version}"

    with mlflow.start_run(run_name=run_name) as run:
        # Set tags
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("dataset_version", dataset_version)
        mlflow.set_tag("train_rows", len(X_train))
        mlflow.set_tag("test_rows", len(X_test))
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)

        # Train model
        if model_type == "random_forest":
            model = train_random_forest(X_train, y_train, config)
            params = config["random_forest"]
        elif model_type == "xgboost":
            model = train_xgboost(X_train, y_train, config)
            params = config["xgboost"]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Log hyperparameters
        mlflow.log_params(params)

        # Evaluate
        metrics, y_proba = compute_metrics(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Log artifacts
        fig_cm = plot_confusion_matrix(y_test, model.predict(X_test))
        fig_roc = plot_roc_curve(y_test, y_proba)
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        mlflow.log_figure(fig_roc, "roc_curve.png")

        # Log model
        if model_type == "random_forest":
            mlflow.sklearn.log_model(
                model, "model",
                registered_model_name=config["model_name"],
            )
        else:
            mlflow.xgboost.log_model(
                model, "model",
                registered_model_name=config["model_name"],
            )

        logger.info(
            "Run %s complete | AUC=%.4f | F1=%.4f",
            run.info.run_id[:8],
            metrics["roc_auc"],
            metrics["f1"],
        )
        return run.info.run_id
