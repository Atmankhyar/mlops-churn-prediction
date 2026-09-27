"""Unit tests for model training and evaluation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models.evaluate import compute_metrics
from src.models.predict import predict_churn
from src.models.train import (
    DEFAULT_MIN_REGISTRATION_AUC,
    get_min_registration_auc,
    load_config,
    should_register_model,
)


@pytest.fixture
def trained_rf_model():
    """Tiny RF model trained on dummy data."""
    X = pd.DataFrame(
        {
            "tenure_months": [12, 24, 6, 48, 3, 36],
            "total_spending_3m": [300, 200, 400, 150, 500, 100],
            "avg_monthly_charge": [100, 66, 133, 50, 166, 33],
            "payment_failure_rate": [0.1, 0.0, 0.3, 0.0, 0.5, 0.0],
            "transaction_count_3m": [3, 3, 3, 3, 3, 3],
            "support_tickets_3m": [2, 0, 4, 0, 5, 0],
            "avg_resolution_days": [4.0, 2.0, 7.0, 1.0, 9.0, 1.0],
            "avg_sentiment": [-0.3, 0.5, -0.8, 0.8, -1.0, 0.9],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 0])
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


class TestComputeMetrics:
    def test_returns_all_metrics(self, trained_rf_model):
        model, X, y = trained_rf_model
        metrics, proba = compute_metrics(model, X, y)
        for key in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
            assert key in metrics

    def test_metrics_in_range(self, trained_rf_model):
        model, X, y = trained_rf_model
        metrics, proba = compute_metrics(model, X, y)
        for val in metrics.values():
            assert 0.0 <= val <= 1.0

    def test_proba_shape(self, trained_rf_model):
        model, X, y = trained_rf_model
        _, proba = compute_metrics(model, X, y)
        assert len(proba) == len(y)
        assert all(0.0 <= p <= 1.0 for p in proba)


# ---------------------------------------------------------------------------
# Registration gate
# ---------------------------------------------------------------------------
class TestRegistrationGate:
    """TRAIN -> EVALUATE -> GATE -> (register | run only).

    A model only earns a Model Registry version when it clears the acceptance
    threshold declared in configs/model_config.yaml.
    """

    MODEL = "churn-predictor"

    def test_auc_above_threshold_is_registered(self):
        assert should_register_model(0.6618, 0.66, self.MODEL) is True

    def test_auc_below_threshold_is_not_registered(self):
        assert should_register_model(0.6555, 0.66, self.MODEL) is False

    def test_auc_exactly_at_threshold_is_registered(self):
        """The gate is inclusive: >= passes."""
        assert should_register_model(0.66, 0.66, self.MODEL) is True

    def test_registration_can_be_disabled_even_when_gate_passes(self):
        """register=False -> registered_model_name is None -> never registers."""
        assert should_register_model(0.99, 0.66, None) is False

    def test_threshold_comes_from_config(self):
        config = load_config()
        assert "registration" in config, "model_config.yaml must declare the gate"
        assert get_min_registration_auc(config) == pytest.approx(0.66)

    def test_threshold_falls_back_when_config_omits_it(self):
        assert get_min_registration_auc({}) == pytest.approx(DEFAULT_MIN_REGISTRATION_AUC)


# ---------------------------------------------------------------------------
# Inference contract
# ---------------------------------------------------------------------------
class _FakePyfunc:
    """Mimics MLflow's pyfunc wrapper: predict() yields hard class labels."""

    def __init__(self, estimator):
        self._model_impl = SimpleNamespace(sklearn_model=estimator)

    def predict(self, features):
        return self._model_impl.sklearn_model.predict(features)


class TestPredictChurnProbability:
    def test_returns_probabilities_not_class_labels(self, trained_rf_model):
        """Regression: pyfunc.predict() returns 0/1, so churn_probability was
        collapsing to exactly 0.0 or 1.0 instead of a real probability."""
        model, X, _ = trained_rf_model
        _, probas = predict_churn(_FakePyfunc(model), X)

        expected = model.predict_proba(X)[:, 1]
        assert np.allclose(probas, expected)
        assert not set(np.unique(probas)).issubset({0.0, 1.0})

    def test_predictions_follow_the_threshold(self, trained_rf_model):
        model, X, _ = trained_rf_model
        preds, probas = predict_churn(_FakePyfunc(model), X, threshold=0.5)
        assert np.array_equal(preds, (probas >= 0.5).astype(int))

    def test_probabilities_in_unit_range(self, trained_rf_model):
        model, X, _ = trained_rf_model
        _, probas = predict_churn(_FakePyfunc(model), X)
        assert ((probas >= 0.0) & (probas <= 1.0)).all()

    def test_accepts_bare_estimator(self, trained_rf_model):
        """A raw sklearn estimator exposes predict_proba directly."""
        model, X, _ = trained_rf_model
        _, probas = predict_churn(model, X)
        assert np.allclose(probas, model.predict_proba(X)[:, 1])
