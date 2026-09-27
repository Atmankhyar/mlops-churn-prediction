"""Tests for scripts/run_drift_monitoring.py.

Uses small synthetic DataFrames and an isolated monitoring_config.yaml
(via a patched CONFIG_PATH) so these tests never touch the real
monitoring/reports/ directory or the real inference log.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import run_drift_monitoring as runner
from src.monitoring import drift as drift_mod

FEATURES = [
    "tenure_months",
    "total_spending_3m",
    "avg_monthly_charge",
    "payment_failure_rate",
    "avg_sentiment",
]


def _make_dataframe(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tenure_months": rng.integers(1, 72, size=n).astype(float),
            "total_spending_3m": rng.normal(150, 40, size=n),
            "avg_monthly_charge": rng.normal(70, 15, size=n),
            "payment_failure_rate": rng.uniform(0, 0.3, size=n),
            "avg_sentiment": rng.normal(0, 0.3, size=n),
        }
    )


def _make_current(n: int, seed: int = 1) -> pd.DataFrame:
    df = _make_dataframe(n, seed)
    df["model_version"] = "test-v1"
    df["churn_probability"] = np.random.default_rng(seed).uniform(0, 1, size=n)
    df["predicted_churn"] = (df["churn_probability"] >= 0.5).astype(int)
    return df


def _write_config(tmp_path, reports_dir, recent_window_days: int = 7) -> Path:
    config = {
        "drift": {
            "p_value_threshold": 0.05,
            "features_to_monitor": FEATURES,
            "prediction_column": "churn_probability",
            "target_column": "churned",
        },
        "reports": {"output_dir": str(reports_dir), "frequency": "weekly"},
        "inference_logging": {
            "enabled": True,
            "database_path": "monitoring/inference_log.db",
            "retention_days": 30,
            "recent_window_days": recent_window_days,
        },
    }
    config_path = tmp_path / "monitoring_config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


@pytest.fixture
def reference_df():
    return _make_dataframe(200, seed=0)


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point drift.load_config() at a throwaway config with an isolated reports dir."""
    reports_dir = tmp_path / "reports"
    config_path = _write_config(tmp_path, reports_dir)
    monkeypatch.setattr(drift_mod, "CONFIG_PATH", config_path)
    return reports_dir


class TestNormalOperation:
    def test_generates_report_with_sufficient_data(self, reference_df, isolated_config):
        current = _make_current(40)
        with patch.object(
            runner, "read_recent_inference", return_value=current
        ), patch.object(runner, "load_reference_features", return_value=reference_df):
            result = runner.run_monitoring()

        assert result["status"] == "completed"
        assert result["sample_size"] == 40
        assert result["report_path"] is not None
        assert result["drift_results"] is not None
        assert set(result["drift_results"].keys()) == set(FEATURES)

    def test_report_written_to_configured_reports_dir(self, reference_df, isolated_config):
        current = _make_current(40)
        with patch.object(
            runner, "read_recent_inference", return_value=current
        ), patch.object(runner, "load_reference_features", return_value=reference_df):
            result = runner.run_monitoring()

        report_path = Path(result["report_path"])
        assert report_path.exists()
        assert report_path.parent == isolated_config
        assert report_path.name.startswith("drift_report_production_")


class TestInsufficientData:
    def test_zero_predictions_skips_cleanly(self, reference_df, isolated_config):
        empty = pd.DataFrame(columns=FEATURES)
        with patch.object(
            runner, "read_recent_inference", return_value=empty
        ), patch.object(runner, "load_reference_features", return_value=reference_df):
            result = runner.run_monitoring()

        assert result["status"] == "skipped_no_data"
        assert result["report_path"] is None
        assert result["drift_results"] is None

    def test_too_few_predictions_skips_cleanly(self, reference_df, isolated_config):
        current = _make_current(5)
        with patch.object(
            runner, "read_recent_inference", return_value=current
        ), patch.object(runner, "load_reference_features", return_value=reference_df):
            result = runner.run_monitoring(min_sample_size=30)

        assert result["status"] == "skipped_insufficient_data"
        assert result["sample_size"] == 5
        assert result["report_path"] is None

    def test_exactly_at_threshold_runs(self, reference_df, isolated_config):
        current = _make_current(30)
        with patch.object(
            runner, "read_recent_inference", return_value=current
        ), patch.object(runner, "load_reference_features", return_value=reference_df):
            result = runner.run_monitoring(min_sample_size=30)

        assert result["status"] == "completed"


class TestConfiguredWindow:
    def test_recent_window_days_is_passed_through(self, tmp_path, monkeypatch, reference_df):
        reports_dir = tmp_path / "reports"
        config_path = _write_config(tmp_path, reports_dir, recent_window_days=14)
        monkeypatch.setattr(drift_mod, "CONFIG_PATH", config_path)

        current = _make_current(40)
        with patch.object(
            runner, "read_recent_inference", return_value=current
        ) as mock_read, patch.object(runner, "load_reference_features", return_value=reference_df):
            runner.run_monitoring()

        mock_read.assert_called_once_with(window_days=14)

    def test_default_window_is_seven_days(self, tmp_path, monkeypatch, reference_df):
        reports_dir = tmp_path / "reports"
        config_path = _write_config(tmp_path, reports_dir, recent_window_days=7)
        monkeypatch.setattr(drift_mod, "CONFIG_PATH", config_path)

        current = _make_current(40)
        with patch.object(
            runner, "read_recent_inference", return_value=current
        ) as mock_read, patch.object(runner, "load_reference_features", return_value=reference_df):
            runner.run_monitoring()

        mock_read.assert_called_once_with(window_days=7)
