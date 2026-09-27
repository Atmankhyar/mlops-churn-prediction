"""Integration tests for the FastAPI endpoints.

These tests use httpx's TestClient and mock the model loading so no
MLflow registry connection is required.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def mock_model():
    """A mock MLflow pyfunc model that returns deterministic predictions.

    predict_churn() (src/models/predict.py) always prefers predict_proba()
    when the model exposes it, which a bare MagicMock does for any attribute
    name -- so predict_proba, not predict, must be the one configured here.
    """
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8]])   # 80% churn prob
    return model


@pytest.fixture
def client(mock_model, tmp_path, monkeypatch):
    # Redirect inference logging to a throwaway DB so these mocked requests
    # (fake model, fixed 80% probability) never pollute the real
    # monitoring/inference_log.db used for actual production drift analysis.
    db_path = tmp_path / "test_inference_log.db"
    monkeypatch.setenv("INFERENCE_LOG_DB", str(db_path))
    # TestClient(app) triggers the startup event, which calls
    # load_model_from_registry() for real. Left unpatched, that clobbers the
    # _model patch below the moment a real MLflow server is reachable --
    # silently turning every "mocked" test here into one that depends on
    # live infrastructure. Force it to fail so the startup except-branch
    # leaves our patched mock in place, as the fixture actually intends.
    with patch("api.main._model", mock_model), \
         patch("api.main._model_version", "test-v1"), \
         patch("api.main.load_model_from_registry", side_effect=RuntimeError("mocked: no registry in tests")):
        with TestClient(app) as c:
            c.inference_db_path = db_path
            yield c


def _inference_log_row_count(client) -> int:
    conn = sqlite3.connect(str(client.inference_db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM inference_logs").fetchone()[0]
    finally:
        conn.close()


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


class TestInferenceLogging:
    def test_successful_predict_is_logged(self, client):
        resp = client.post("/predict", json=SAMPLE_REQUEST)
        assert resp.status_code == 200
        assert _inference_log_row_count(client) == 1

    def test_successful_batch_logs_each_prediction(self, client):
        resp = client.post(
            "/predict/batch", json={"customers": [SAMPLE_REQUEST, SAMPLE_REQUEST, SAMPLE_REQUEST]}
        )
        assert resp.status_code == 200
        assert _inference_log_row_count(client) == 3

    def test_invalid_request_is_not_logged(self, client):
        bad_request = {**SAMPLE_REQUEST, "avg_sentiment": 999.0}
        resp = client.post("/predict", json=bad_request)
        assert resp.status_code == 422
        assert _inference_log_row_count(client) == 0

    def test_batch_failure_logs_nothing_for_the_whole_batch(self, client, mock_model):
        # First call in the batch succeeds, second raises -> the endpoint
        # returns 500 for the whole batch and must not have logged the
        # customer(s) that succeeded before the failure.
        mock_model.predict_proba.side_effect = [
            np.array([[0.2, 0.8]]),
            RuntimeError("simulated inference failure"),
        ]
        resp = client.post(
            "/predict/batch", json={"customers": [SAMPLE_REQUEST, SAMPLE_REQUEST]}
        )
        assert resp.status_code == 500
        assert _inference_log_row_count(client) == 0

    def test_logging_failure_does_not_break_predict_response(self, client):
        with patch("api.main.log_prediction", side_effect=RuntimeError("db unavailable")):
            resp = client.post("/predict", json=SAMPLE_REQUEST)
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["churn_probability"] <= 1.0

    def test_logged_record_excludes_customer_id(self, client):
        client.post("/predict", json=SAMPLE_REQUEST)
        conn = sqlite3.connect(str(client.inference_db_path))
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(inference_logs)")}
        finally:
            conn.close()
        assert "customer_id" not in columns

