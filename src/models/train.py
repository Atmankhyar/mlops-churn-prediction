from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import mlflow
from src.models.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "model_config.yaml"


def load_config() -> dict:
    """Load model configuration from YAML."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Registration gate
# ---------------------------------------------------------------------------

# Fallback only — the real value lives in configs/model_config.yaml under
# registration.min_roc_auc so the acceptance criterion is reviewable in config
# rather than buried in code.
DEFAULT_MIN_REGISTRATION_AUC = 0.66


def get_min_registration_auc(config: dict) -> float:
    """Read the registration gate threshold from the model config."""
    return float(
        config.get("registration", {}).get(
            "min_roc_auc",
            DEFAULT_MIN_REGISTRATION_AUC,
        )
    )


def should_register_model(
    roc_auc: float,
    min_roc_auc: float,
    registered_model_name: str | None,
) -> bool:
    """Decide whether a trained model earns a Model Registry version.

    A model is registered only when registration is enabled for the run *and*
    its test ROC-AUC reaches the configured acceptance threshold. Failing the
    gate does not fail the run: the experiment, its metrics and its artifacts
    are still tracked, only the registry version is withheld.
    """
    if registered_model_name is None:
        return False
    return roc_auc >= min_roc_auc


def prepare_features(
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select feature columns and target, encode categoricals."""

    numerical_cols = config["features"]["numerical"]
    categorical_cols = config["features"]["categorical"]
    target = config["target"]

    # One-hot encode categorical features
    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=False,
    )

    # Build final feature list
    feature_cols = numerical_cols + [
        c for c in df.columns if any(c.startswith(cat + "_") for cat in categorical_cols)
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[target].astype(int)

    return X, y


def split_train_validation_test(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Split data into train, validation, and final test sets.

    Default configuration:
        64% train
        16% validation
        20% test
    """

    split_cfg = config["train_test_split"]

    test_size = split_cfg["test_size"]
    validation_size = split_cfg.get("validation_size", 0.20)
    random_state = split_cfg["random_state"]
    stratify = y if split_cfg["stratify"] else None

    # First split:
    # 80% temporary train/validation
    # 20% untouched final test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Convert validation_size from a percentage of the
    # complete dataset to the corresponding fraction
    # of the train_val dataset.
    validation_fraction = validation_size / (1.0 - test_size)

    stratify_train_val = y_train_val if split_cfg["stratify"] else None

    # Second split:
    # approximately 64% train
    # approximately 16% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=stratify_train_val,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> RandomForestClassifier:
    """Train a Random Forest classifier."""

    params = config["random_forest"]

    model = RandomForestClassifier(**params)

    model.fit(
        X_train,
        y_train,
    )

    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    model_config_key: str = "xgboost",
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> XGBClassifier:
    """
    Train an XGBoost classifier.

    If validation data is provided, it is used as the
    evaluation set instead of the training data.
    """

    params = {k: v for k, v in config[model_config_key].items() if k != "eval_metric"}

    eval_metric = config[model_config_key].get(
        "eval_metric",
        "auc",
    )

    model = XGBClassifier(
        **params,
        eval_metric=eval_metric,
    )

    # Use validation set when available.
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
    else:
        # Backward-compatible fallback.
        eval_set = [(X_train, y_train)]

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )

    return model


def run_training(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    dataset_version: str = "v1.0",
    tags: dict[str, Any] | None = None,
    extra_params: dict[str, Any] | None = None,
    register: bool = True,
    min_registration_auc: float | None = None,
) -> str:
    """
    Full training run with MLflow tracking.

    Data is split into:

        64% training
        16% validation
        20% final test

    The validation set is available during model development,
    while the final test set is reserved for final evaluation.

    Args:
        df: Feature DataFrame including the target column.
        model_type:
            'random_forest',
            'xgboost',
            or 'xgboost_tuned'
        dataset_version: Version tag for reproducibility.
        tags: Additional MLflow run tags.
        extra_params:
            Additional MLflow params, e.g. dataset provenance such as
            training_rows / feature_count.
        register:
            When False the run is tracked but never considered for the Model
            Registry (useful for experiments and tests).
        min_registration_auc:
            Override the registration gate from configs/model_config.yaml.

    Returns:
        MLflow run_id of the completed run.
    """

    config = load_config()

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5000",
        )
    )

    mlflow.set_experiment(config["experiment_name"])

    # ---------------------------------------------------------
    # Feature preparation
    # ---------------------------------------------------------

    X, y = prepare_features(
        df,
        config,
    )

    # ---------------------------------------------------------
    # Train / Validation / Test split
    # ---------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_train_validation_test(
        X,
        y,
        config,
    )

    logger.info(
        "Dataset split | train=%d | validation=%d | test=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # ---------------------------------------------------------
    # MLflow run
    # ---------------------------------------------------------

    run_name = f"{model_type}-{dataset_version}"

    with mlflow.start_run(run_name=run_name) as run:

        # -----------------------------------------------------
        # Set tags
        # -----------------------------------------------------

        mlflow.set_tag(
            "model_type",
            model_type,
        )

        mlflow.set_tag(
            "dataset_version",
            dataset_version,
        )

        mlflow.set_tag(
            "train_rows",
            len(X_train),
        )

        mlflow.set_tag(
            "validation_rows",
            len(X_val),
        )

        mlflow.set_tag(
            "test_rows",
            len(X_test),
        )

        mlflow.set_tag(
            "split_strategy",
            "64/16/20 train/validation/test",
        )

        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)

        # -----------------------------------------------------
        # Train model
        # -----------------------------------------------------

        fit_start = time.perf_counter()

        if model_type == "random_forest":

            model = train_random_forest(
                X_train,
                y_train,
                config,
            )

            params = config["random_forest"]

        elif model_type == "xgboost":

            model = train_xgboost(
                X_train,
                y_train,
                config,
                model_config_key="xgboost",
                X_val=X_val,
                y_val=y_val,
            )

            params = config["xgboost"]

        elif model_type == "xgboost_tuned":

            model = train_xgboost(
                X_train,
                y_train,
                config,
                model_config_key="xgboost_tuned",
                X_val=X_val,
                y_val=y_val,
            )

            params = config["xgboost_tuned"]

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        training_time_seconds = time.perf_counter() - fit_start

        mlflow.log_metric(
            "training_time_seconds",
            round(training_time_seconds, 3),
        )

        # -----------------------------------------------------
        # Log hyperparameters
        # -----------------------------------------------------

        mlflow.log_params(params)

        mlflow.log_params(
            {
                "train_rows": len(X_train),
                "validation_rows": len(X_val),
                "test_rows": len(X_test),
                "feature_count": X.shape[1],
            }
        )

        if extra_params:
            mlflow.log_params(extra_params)

        # -----------------------------------------------------
        # Validation evaluation
        # -----------------------------------------------------

        val_metrics, _ = compute_metrics(
            model,
            X_val,
            y_val,
        )

        for key, value in val_metrics.items():
            mlflow.log_metric(
                f"val_{key}",
                value,
            )

        # -----------------------------------------------------
        # Final test evaluation
        # -----------------------------------------------------
        #
        # IMPORTANT:
        # The final test set is kept separate from validation.
        #

        test_metrics, y_proba = compute_metrics(
            model,
            X_test,
            y_test,
        )

        for key, value in test_metrics.items():
            mlflow.log_metric(
                key,
                value,
            )

        # -----------------------------------------------------
        # Log test artifacts
        # -----------------------------------------------------

        fig_cm = plot_confusion_matrix(
            y_test,
            model.predict(X_test),
        )

        fig_roc = plot_roc_curve(
            y_test,
            y_proba,
        )

        mlflow.log_figure(
            fig_cm,
            "confusion_matrix.png",
        )

        mlflow.log_figure(
            fig_roc,
            "roc_curve.png",
        )

        # -----------------------------------------------------
        # Registration gate
        # -----------------------------------------------------
        #
        # TRAIN -> EVALUATE -> GATE -> (register | run only)
        #
        # The run is always tracked. Only a model that clears the
        # acceptance threshold earns a Model Registry version.
        #

        registered_model_name = config["model_name"] if register else None

        gate_auc = (
            min_registration_auc
            if min_registration_auc is not None
            else get_min_registration_auc(config)
        )

        gate_passed = should_register_model(
            test_metrics["roc_auc"],
            gate_auc,
            registered_model_name,
        )

        mlflow.log_param(
            "registration_gate_auc",
            gate_auc,
        )

        mlflow.set_tag(
            "registration_status",
            "registered" if gate_passed else "not_registered",
        )

        # -----------------------------------------------------
        # Log model
        # -----------------------------------------------------

        log_model = (
            mlflow.sklearn.log_model if model_type == "random_forest" else mlflow.xgboost.log_model
        )

        log_model(
            model,
            "model",
            registered_model_name=(registered_model_name if gate_passed else None),
        )

        if gate_passed:
            logger.info(
                "Registration gate PASSED " "(roc_auc=%.4f >= %.4f) — registered as '%s'",
                test_metrics["roc_auc"],
                gate_auc,
                registered_model_name,
            )
        else:
            logger.info(
                "Registration gate NOT passed "
                "(roc_auc=%.4f < %.4f) — run tracked, no registry version",
                test_metrics["roc_auc"],
                gate_auc,
            )

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------

        logger.info(
            "Run %s complete | " "Test AUC=%.4f | Test F1=%.4f | " "Test Recall=%.4f",
            run.info.run_id[:8],
            test_metrics["roc_auc"],
            test_metrics["f1"],
            test_metrics["recall"],
        )

        logger.info(
            "Validation | " "AUC=%.4f | F1=%.4f | Recall=%.4f",
            val_metrics["roc_auc"],
            val_metrics["f1"],
            val_metrics["recall"],
        )

        return run.info.run_id
