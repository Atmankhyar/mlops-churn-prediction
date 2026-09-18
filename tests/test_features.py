"""Unit tests for feature engineering."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.engineering import (
    build_transaction_features,
    build_support_features,
    build_full_feature_table,
)


@pytest.fixture
def sample_customers():
    return pd.DataFrame({
        "customer_id": ["c1", "c2", "c3"],
        "tenure_months": [12, 24, 6],
        "contract_type": ["Month-to-month", "One year", "Two year"],
    })


@pytest.fixture
def sample_transactions():
    return pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3", "t4", "t5"],
        "customer_id": ["c1", "c1", "c2", "c3", "c3"],
        "transaction_date": pd.to_datetime([
            "2024-02-01", "2024-03-01",
            "2024-02-15",
            "2024-01-10", "2024-03-20",
        ]),
        "amount": [80.0, 90.0, 100.0, 60.0, 70.0],
        "payment_status": ["paid", "failed", "paid", "late", "paid"],
    })


@pytest.fixture
def sample_interactions():
    return pd.DataFrame({
        "interaction_id": ["i1", "i2", "i3"],
        "customer_id": ["c1", "c2", "c3"],
        "ticket_date": pd.to_datetime(["2024-02-01", "2024-03-01", "2024-03-15"]),
        "resolution_days": [3, 5, 7],
        "sentiment_score": [-0.5, 0.2, 0.8],
    })


@pytest.fixture
def sample_labels():
    return pd.DataFrame({
        "customer_id": ["c1", "c2", "c3"],
        "churned": [True, False, True],
    })


class TestTransactionFeatures:
    def test_output_columns(self, sample_transactions):
        result = build_transaction_features(sample_transactions)
        assert "customer_id" in result.columns
        assert "total_spending_3m" in result.columns
        assert "payment_failure_rate" in result.columns

    def test_no_duplicate_customers(self, sample_transactions):
        result = build_transaction_features(sample_transactions)
        assert result["customer_id"].nunique() == result.shape[0]

    def test_payment_failure_rate_in_range(self, sample_transactions):
        result = build_transaction_features(sample_transactions)
        assert (result["payment_failure_rate"] >= 0).all()
        assert (result["payment_failure_rate"] <= 1).all()


class TestSupportFeatures:
    def test_output_columns(self, sample_interactions):
        result = build_support_features(sample_interactions)
        assert "support_tickets_3m" in result.columns
        assert "avg_sentiment" in result.columns

    def test_avg_sentiment_in_range(self, sample_interactions):
        result = build_support_features(sample_interactions)
        assert (result["avg_sentiment"] >= -1.0).all()
        assert (result["avg_sentiment"] <= 1.0).all()


class TestFullFeatureTable:
    def test_shape(
        self,
        sample_customers,
        sample_transactions,
        sample_interactions,
        sample_labels,
    ):
        result = build_full_feature_table(
            sample_customers, sample_transactions, sample_interactions, sample_labels
        )
        assert len(result) == 3

    def test_has_target(
        self,
        sample_customers,
        sample_transactions,
        sample_interactions,
        sample_labels,
    ):
        result = build_full_feature_table(
            sample_customers, sample_transactions, sample_interactions, sample_labels
        )
        assert "churned" in result.columns
