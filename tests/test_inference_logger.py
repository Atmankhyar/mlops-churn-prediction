"""Unit tests for src/monitoring/inference_logger.py.

Uses a temporary SQLite database per test (via the INFERENCE_LOG_DB env
override) so these tests never touch the real monitoring/inference_log.db.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.monitoring import inference_logger as log_mod

SAMPLE_FEATURES = {
    "tenure_months": 24,
    "total_spending_3m": 287.5,
    "avg_monthly_charge": 95.8,
    "payment_failure_rate": 0.1,
    "avg_sentiment": -0.3,
}


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point every test at a throwaway database file."""
    db_path = tmp_path / "test_inference_log.db"
    monkeypatch.setenv("INFERENCE_LOG_DB", str(db_path))
    return db_path


def _table_columns(db_path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("PRAGMA table_info(inference_logs)").fetchall()
    finally:
        conn.close()
    return {row[1] for row in rows}


class TestInitializeDatabase:
    def test_creates_table_and_index(self, isolated_db):
        log_mod.initialize_database()

        conn = sqlite3.connect(str(isolated_db))
        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='inference_logs'"
            ).fetchall()
            indexes = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name='idx_inference_logs_timestamp'"
            ).fetchall()
        finally:
            conn.close()

        assert len(tables) == 1
        assert len(indexes) == 1


class TestLogPrediction:
    def test_successful_insert_stores_exactly_one_row(self, isolated_db):
        log_mod.initialize_database()
        log_mod.log_prediction(
            features=SAMPLE_FEATURES,
            probability=0.63,
            prediction=1,
            model_version="ca13ab72",
        )

        conn = sqlite3.connect(str(isolated_db))
        try:
            count = conn.execute("SELECT COUNT(*) FROM inference_logs").fetchone()[0]
        finally:
            conn.close()
        assert count == 1

    def test_rejects_invalid_probability(self, isolated_db):
        log_mod.initialize_database()
        with pytest.raises(ValueError):
            log_mod.log_prediction(
                features=SAMPLE_FEATURES,
                probability=1.5,
                prediction=1,
                model_version="ca13ab72",
            )

    def test_rejects_invalid_prediction_label(self, isolated_db):
        log_mod.initialize_database()
        with pytest.raises(ValueError):
            log_mod.log_prediction(
                features=SAMPLE_FEATURES,
                probability=0.5,
                prediction=2,
                model_version="ca13ab72",
            )


class TestPrivacy:
    def test_no_customer_id_or_raw_fields_in_schema(self, isolated_db):
        log_mod.initialize_database()
        columns = _table_columns(isolated_db)

        assert columns == {
            "id",
            "timestamp",
            "model_version",
            "tenure_months",
            "total_spending_3m",
            "avg_monthly_charge",
            "payment_failure_rate",
            "avg_sentiment",
            "churn_probability",
            "predicted_churn",
        }
        assert "customer_id" not in columns

    def test_extra_feature_keys_are_ignored_not_persisted(self, isolated_db):
        log_mod.initialize_database()
        features_with_extra = {**SAMPLE_FEATURES, "customer_id": "should-not-be-stored"}
        log_mod.log_prediction(
            features=features_with_extra,
            probability=0.4,
            prediction=0,
            model_version="ca13ab72",
        )
        columns = _table_columns(isolated_db)
        assert "customer_id" not in columns


class TestLogPredictions:
    def _record(self, **overrides) -> dict:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": "ca13ab72",
            "churn_probability": 0.5,
            "predicted_churn": 0,
            **SAMPLE_FEATURES,
        }
        record.update(overrides)
        return record

    def test_batch_insert_commits_all_records_together(self, isolated_db):
        log_mod.initialize_database()
        records = [self._record(), self._record(), self._record()]
        log_mod.log_predictions(records)

        conn = sqlite3.connect(str(isolated_db))
        try:
            count = conn.execute("SELECT COUNT(*) FROM inference_logs").fetchone()[0]
        finally:
            conn.close()
        assert count == 3

    def test_empty_batch_inserts_nothing(self, isolated_db):
        """Mirrors the API behavior: if a batch fails before logging is
        reached, main.py never calls log_predictions with partial records —
        it simply never calls it, or calls it with []. Either way, nothing
        should be persisted."""
        log_mod.initialize_database()
        log_mod.log_predictions([])

        conn = sqlite3.connect(str(isolated_db))
        try:
            count = conn.execute("SELECT COUNT(*) FROM inference_logs").fetchone()[0]
        finally:
            conn.close()
        assert count == 0


class TestReadRecentInference:
    def test_returns_only_records_within_window(self, isolated_db):
        log_mod.initialize_database()
        now = datetime.now(timezone.utc)

        log_mod.log_prediction(
            features=SAMPLE_FEATURES,
            probability=0.5,
            prediction=0,
            model_version="v1",
            timestamp=now - timedelta(days=2),
        )
        log_mod.log_prediction(
            features=SAMPLE_FEATURES,
            probability=0.6,
            prediction=1,
            model_version="v1",
            timestamp=now - timedelta(days=10),
        )

        recent = log_mod.read_recent_inference(window_days=7)

        assert isinstance(recent, pd.DataFrame)
        assert len(recent) == 1
        assert recent.iloc[0]["churn_probability"] == 0.5

    def test_empty_history_returns_empty_dataframe(self, isolated_db):
        log_mod.initialize_database()
        recent = log_mod.read_recent_inference(window_days=7)
        assert isinstance(recent, pd.DataFrame)
        assert len(recent) == 0


class TestRetentionCleanup:
    def test_deletes_records_older_than_retention_period(self, isolated_db):
        log_mod.initialize_database()
        now = datetime.now(timezone.utc)

        log_mod.log_prediction(
            features=SAMPLE_FEATURES,
            probability=0.5,
            prediction=0,
            model_version="v1",
            timestamp=now - timedelta(days=45),
        )
        log_mod.log_prediction(
            features=SAMPLE_FEATURES,
            probability=0.6,
            prediction=1,
            model_version="v1",
            timestamp=now - timedelta(days=5),
        )

        deleted = log_mod.cleanup_old_records(retention_days=30)

        conn = sqlite3.connect(str(isolated_db))
        try:
            remaining = conn.execute("SELECT COUNT(*) FROM inference_logs").fetchone()[0]
        finally:
            conn.close()

        assert deleted == 1
        assert remaining == 1
