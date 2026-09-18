"""Unit tests for model training and evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models.evaluate import compute_metrics


@pytest.fixture
def trained_rf_model():
    """Tiny RF model trained on dummy data."""
    X = pd.DataFrame({
        "tenure_months": [12, 24, 6, 48, 3, 36],
        "total_spending_3m": [300, 200, 400, 150, 500, 100],
        "avg_monthly_charge": [100, 66, 133, 50, 166, 33],
        "payment_failure_rate": [0.1, 0.0, 0.3, 0.0, 0.5, 0.0],
        "transaction_count_3m": [3, 3, 3, 3, 3, 3],
        "support_tickets_3m": [2, 0, 4, 0, 5, 0],
        "avg_resolution_days": [4.0, 2.0, 7.0, 1.0, 9.0, 1.0],
        "avg_sentiment": [-0.3, 0.5, -0.8, 0.8, -1.0, 0.9],
    })
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
