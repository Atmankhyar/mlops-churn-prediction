"""Integration tests for the FastAPI endpoints.

These tests use httpx's TestClient and mock the model loading so no
MLflow registry connection is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def mock_model():
    """A mock MLflow pyfunc model that returns deterministic predictions."""
    model = MagicMock()
    model.predict.return_value = np.array([[0.2, 0.8]])   # 80% churn prob
    return model


@pytest.fixture
def client(mock_model):
    with patch("api.main._model", mock_model), \
         patch("api.main._model_version", "test-v1"):
        with TestClient(app) as c:
            yield c


SAMPLE_REQUEST = {
    "customer_id": "test-customer-001",
    "tenure_months": 24,
    "contract_type": "Month-to-month",
    "total_spending_3m": 287.5,
    "avg_monthly_charge": 95.8,
    "payment_failure_rate": 0.1,
    "transaction_count_3m": 3,
    "support_tickets_3m": 2,
    "avg_resolution_days": 4.2,
    "avg_sentiment": -0.3,
}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_ok(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        resp = client.post("/predict", json=SAMPLE_REQUEST)
        assert resp.status_code == 200

    def test_predict_has_required_fields(self, client):
        resp = client.post("/predict", json=SAMPLE_REQUEST)
        body = resp.json()
        assert "prediction" in body
        assert "churn_probability" in body
        assert "model_version" in body
        assert "timestamp" in body

    def test_predict_probability_in_range(self, client):
        resp = client.post("/predict", json=SAMPLE_REQUEST)
        prob = resp.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_rejects_invalid_sentiment(self, client):
        bad_request = {**SAMPLE_REQUEST, "avg_sentiment": 999.0}
        resp = client.post("/predict", json=bad_request)
        assert resp.status_code == 422   # Pydantic validation error


class TestBatchPredictEndpoint:
    def test_batch_predict_returns_200(self, client):
        resp = client.post("/predict/batch", json={"customers": [SAMPLE_REQUEST]})
        assert resp.status_code == 200

    def test_batch_count_matches_input(self, client):
        resp = client.post("/predict/batch", json={"customers": [SAMPLE_REQUEST, SAMPLE_REQUEST]})
        assert resp.json()["total"] == 2

