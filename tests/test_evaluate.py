"""Unit tests for src/models/evaluate.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")  # headless backend for CI/test environments

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve


def _mock_model(y_pred, y_proba_positive):
    model = MagicMock()
    model.predict.return_value = np.array(y_pred)
    model.predict_proba.return_value = np.column_stack(
        [1 - np.array(y_proba_positive), y_proba_positive]
    )
    return model


class TestComputeMetrics:
    def test_returns_all_expected_metric_keys(self):
        model = _mock_model(y_pred=[0, 1, 1, 1], y_proba_positive=[0.1, 0.6, 0.8, 0.9])
        y_test = pd.Series([0, 0, 1, 1])

        metrics, y_proba = compute_metrics(model, pd.DataFrame({"x": [1, 2, 3, 4]}), y_test)

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "roc_auc" in metrics

    def test_metrics_match_hand_computed_values(self):
        # y_test = [0, 0, 1, 1], y_pred = [0, 1, 1, 1]
        # TP=2 (idx 2,3), FP=1 (idx 1), TN=1 (idx 0), FN=0
        # accuracy = 3/4, precision = 2/3, recall = 2/2, f1 = 2PR/(P+R)
        model = _mock_model(y_pred=[0, 1, 1, 1], y_proba_positive=[0.1, 0.6, 0.8, 0.9])
        y_test = pd.Series([0, 0, 1, 1])

        metrics, _ = compute_metrics(model, pd.DataFrame({"x": [1, 2, 3, 4]}), y_test)

        assert metrics["accuracy"] == 0.75
        assert metrics["precision"] == round(2 / 3, 4)
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == round(2 * (2 / 3) / (2 / 3 + 1), 4)

    def test_roc_auc_matches_sklearn_direct_call(self):
        y_proba_positive = [0.1, 0.6, 0.8, 0.9]
        model = _mock_model(y_pred=[0, 1, 1, 1], y_proba_positive=y_proba_positive)
        y_test = pd.Series([0, 0, 1, 1])

        metrics, y_proba = compute_metrics(model, pd.DataFrame({"x": [1, 2, 3, 4]}), y_test)

        assert metrics["roc_auc"] == round(roc_auc_score(y_test, y_proba_positive), 4)
        assert list(y_proba) == y_proba_positive

    def test_perfect_predictions_score_1_0(self):
        model = _mock_model(y_pred=[0, 0, 1, 1], y_proba_positive=[0.05, 0.1, 0.9, 0.95])
        y_test = pd.Series([0, 0, 1, 1])

        metrics, _ = compute_metrics(model, pd.DataFrame({"x": [1, 2, 3, 4]}), y_test)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_no_positive_predictions_precision_is_zero_not_error(self):
        # zero_division=0 must prevent a ZeroDivisionError/warning-as-error
        model = _mock_model(y_pred=[0, 0, 0, 0], y_proba_positive=[0.1, 0.2, 0.3, 0.4])
        y_test = pd.Series([0, 0, 1, 1])

        metrics, _ = compute_metrics(model, pd.DataFrame({"x": [1, 2, 3, 4]}), y_test)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0


class TestPlots:
    def test_plot_confusion_matrix_returns_figure(self):
        fig = plot_confusion_matrix(y_true=[0, 0, 1, 1], y_pred=np.array([0, 1, 1, 1]))
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_plot_roc_curve_returns_figure(self):
        fig = plot_roc_curve(y_true=[0, 0, 1, 1], y_proba=np.array([0.1, 0.6, 0.8, 0.9]))
        assert fig is not None
        assert hasattr(fig, "savefig")
