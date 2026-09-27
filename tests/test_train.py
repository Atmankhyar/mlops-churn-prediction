"""Unit tests for src/models/train.py.

The registration-gate functions are pure and tested directly. run_training()
is tested with mlflow entirely mocked (src.models.train.mlflow) -- these
tests check the actual decision logic (register / don't register), not
MLflow's own plumbing, and don't require a running MLflow server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    get_min_registration_auc,
    prepare_features,
    run_training,
    should_register_model,
    split_train_validation_test,
)

MODEL_CONFIG = {
    "experiment_name": "churn-prediction-test",
    "model_name": "churn-predictor",
    "features": {
        "numerical": [
            "tenure_months",
            "total_spending_3m",
            "avg_monthly_charge",
            "payment_failure_rate",
        ],
        "categorical": ["contract_type"],
    },
    "target": "churned",
    "registration": {"min_roc_auc": 0.66},
    "train_test_split": {
        "test_size": 0.2,
        "validation_size": 0.2,
        "random_state": 42,
        "stratify": True,
    },
    "random_forest": {
        "n_estimators": 20,
        "max_depth": 4,
        "random_state": 42,
    },
}


def _synthetic_dataset(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    churned = rng.integers(0, 2, size=n)
    return pd.DataFrame(
        {
            "tenure_months": rng.integers(1, 72, size=n),
            "total_spending_3m": rng.normal(150, 40, size=n),
            "avg_monthly_charge": rng.normal(70, 15, size=n),
            "payment_failure_rate": rng.uniform(0, 0.3, size=n),
            "contract_type": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
            # Make churn weakly separable so the model has *some* signal.
            "churned": churned,
        }
    )


class TestGetMinRegistrationAuc:
    def test_reads_configured_threshold(self):
        config = {"registration": {"min_roc_auc": 0.7}}
        assert get_min_registration_auc(config) == 0.7

    def test_falls_back_to_default_when_missing(self):
        assert get_min_registration_auc({}) == 0.66


class TestShouldRegisterModel:
    def test_registers_when_auc_meets_threshold(self):
        assert should_register_model(0.70, 0.66, "churn-predictor") is True

    def test_registers_at_exact_threshold(self):
        assert should_register_model(0.66, 0.66, "churn-predictor") is True

    def test_does_not_register_below_threshold(self):
        assert should_register_model(0.6555, 0.66, "churn-predictor") is False

    def test_does_not_register_when_name_is_none(self):
        # register=False path: no registered_model_name regardless of AUC.
        assert should_register_model(0.99, 0.66, None) is False


class TestPrepareFeatures:
    def test_one_hot_encodes_categorical_columns(self):
        df = _synthetic_dataset(30)
        X, y = prepare_features(df, MODEL_CONFIG)

        assert "contract_type_Month-to-month" in X.columns
        assert "contract_type_One year" in X.columns
        assert "contract_type_Two year" in X.columns
        assert "contract_type" not in X.columns

    def test_includes_all_numerical_features(self):
        df = _synthetic_dataset(30)
        X, _ = prepare_features(df, MODEL_CONFIG)

        for col in MODEL_CONFIG["features"]["numerical"]:
            assert col in X.columns

    def test_target_is_cast_to_int(self):
        df = _synthetic_dataset(30)
        df["churned"] = df["churned"].astype(bool)

        _, y = prepare_features(df, MODEL_CONFIG)

        assert y.dtype == int
        assert set(y.unique()) <= {0, 1}


class TestSplitTrainValidationTest:
    def test_split_proportions_match_configured_sizes(self):
        n = 500
        test_size = MODEL_CONFIG["train_test_split"]["test_size"]
        validation_size = MODEL_CONFIG["train_test_split"]["validation_size"]

        df = _synthetic_dataset(n)
        X, y = prepare_features(df, MODEL_CONFIG)

        X_train, X_val, X_test, y_train, y_val, y_test = split_train_validation_test(
            X, y, MODEL_CONFIG
        )

        total = len(X_train) + len(X_val) + len(X_test)
        assert total == n
        assert len(X_test) == pytest.approx(n * test_size, abs=2)
        assert len(X_val) == pytest.approx(n * validation_size, abs=5)
        assert len(X_train) == pytest.approx(n * (1 - test_size - validation_size), abs=5)

    def test_no_row_overlap_between_splits(self):
        df = _synthetic_dataset(200)
        X, y = prepare_features(df, MODEL_CONFIG)

        X_train, X_val, X_test, *_ = split_train_validation_test(X, y, MODEL_CONFIG)

        assert set(X_train.index) & set(X_val.index) == set()
        assert set(X_train.index) & set(X_test.index) == set()
        assert set(X_val.index) & set(X_test.index) == set()


@pytest.fixture
def mock_mlflow_run():
    """Mock the entire mlflow module and config load used inside src.models.train."""
    with patch("src.models.train.mlflow") as mock_mlflow, patch(
        "src.models.train.load_config", return_value=MODEL_CONFIG
    ):
        run_ctx = MagicMock()
        run_ctx.info.run_id = "0123456789abcdef"
        mock_mlflow.start_run.return_value.__enter__.return_value = run_ctx
        mock_mlflow.start_run.return_value.__exit__.return_value = False
        yield mock_mlflow


class TestRunTrainingRegistrationGate:
    def test_registers_when_gate_passes(self, mock_mlflow_run):
        df = _synthetic_dataset(120)

        run_id = run_training(
            df,
            model_type="random_forest",
            register=True,
            min_registration_auc=0.0,  # any real AUC clears this
        )

        assert run_id == "0123456789abcdef"
        _, kwargs = mock_mlflow_run.sklearn.log_model.call_args
        assert kwargs["registered_model_name"] == "churn-predictor"

    def test_does_not_register_when_gate_fails(self, mock_mlflow_run):
        df = _synthetic_dataset(120)

        run_training(
            df,
            model_type="random_forest",
            register=True,
            min_registration_auc=1.1,  # impossible to reach (AUC max is 1.0)
        )

        _, kwargs = mock_mlflow_run.sklearn.log_model.call_args
        assert kwargs["registered_model_name"] is None

    def test_register_false_never_registers_regardless_of_auc(self, mock_mlflow_run):
        df = _synthetic_dataset(120)

        run_training(
            df,
            model_type="random_forest",
            register=False,
            min_registration_auc=0.0,  # would otherwise clear the gate easily
        )

        _, kwargs = mock_mlflow_run.sklearn.log_model.call_args
        assert kwargs["registered_model_name"] is None

    def test_unknown_model_type_raises(self, mock_mlflow_run):
        df = _synthetic_dataset(120)
        with pytest.raises(ValueError, match="Unknown model_type"):
            run_training(df, model_type="not_a_real_model")
