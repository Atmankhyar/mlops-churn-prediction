"""Unit tests for src/features/spark_engineering.py.

Uses a local[1] SparkSession against small in-memory CSV fixtures. Delta
Lake itself (delta-spark) isn't installed in this environment, so
build_and_save_feature_table()'s write step is out of scope here -- that's
covered as an integration check by src/features/validate_spark_output.py
against the real pipeline instead.
"""

from __future__ import annotations

import pytest

from src.features.spark_engineering import (
    build_support_features_spark,
    build_transaction_features_spark,
)


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[1]")
        .appName("test-spark-engineering")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def _write_csv(tmp_path, name: str, header: str, rows: list[str]) -> str:
    path = tmp_path / name
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(path)


class TestBuildTransactionFeaturesSpark:
    def test_aggregates_per_customer(self, spark, tmp_path):
        path = _write_csv(
            tmp_path,
            "transactions.csv",
            "customer_id,transaction_id,transaction_date,amount,payment_status",
            [
                "1,t1,2026-09-01,100.0,success",
                "1,t2,2026-09-15,50.0,failed",
                "2,t3,2026-09-10,200.0,success",
            ],
        )

        result = {
            row["customer_id"]: row.asDict()
            for row in build_transaction_features_spark(spark, path).collect()
        }

        assert result[1]["total_spending_3m"] == 150.0
        assert result[1]["transaction_count_3m"] == 2
        assert result[1]["payment_failure_rate"] == 0.5
        assert result[2]["total_spending_3m"] == 200.0
        assert result[2]["payment_failure_rate"] == 0.0

    def test_excludes_transactions_outside_lookback_window(self, spark, tmp_path):
        path = _write_csv(
            tmp_path,
            "transactions.csv",
            "customer_id,transaction_id,transaction_date,amount,payment_status",
            [
                "1,t1,2026-09-01,100.0,success",  # within 3 months of max date
                "1,t2,2026-01-01,999.0,success",  # more than 3 months before max date
            ],
        )

        result = {
            row["customer_id"]: row.asDict()
            for row in build_transaction_features_spark(spark, path, lookback_months=3).collect()
        }

        # Only the September transaction should count; the January one predates
        # the 3-month lookback window from the latest transaction date (Sept 1).
        assert result[1]["transaction_count_3m"] == 1
        assert result[1]["total_spending_3m"] == 100.0


class TestBuildSupportFeaturesSpark:
    def test_aggregates_per_customer(self, spark, tmp_path):
        path = _write_csv(
            tmp_path,
            "interactions.csv",
            "customer_id,interaction_id,ticket_date,resolution_days,sentiment_score",
            [
                "1,i1,2026-09-01,2.0,0.5",
                "1,i2,2026-09-10,4.0,-0.5",
                "2,i3,2026-09-05,1.0,1.0",
            ],
        )

        result = {
            row["customer_id"]: row.asDict()
            for row in build_support_features_spark(spark, path).collect()
        }

        assert result[1]["support_tickets_3m"] == 2
        assert result[1]["avg_resolution_days"] == 3.0
        assert result[1]["avg_sentiment"] == 0.0
        assert result[2]["support_tickets_3m"] == 1

    def test_excludes_interactions_outside_lookback_window(self, spark, tmp_path):
        """Regression test for the bug where this function computed a lookback
        cutoff but never applied it, aggregating over all history instead of
        the configured window."""
        path = _write_csv(
            tmp_path,
            "interactions.csv",
            "customer_id,interaction_id,ticket_date,resolution_days,sentiment_score",
            [
                "1,i1,2026-09-01,2.0,0.0",  # within 3 months of max date
                "1,i2,2026-01-01,10.0,0.0",  # more than 3 months before max date
            ],
        )

        result = {
            row["customer_id"]: row.asDict()
            for row in build_support_features_spark(spark, path, lookback_months=3).collect()
        }

        assert result[1]["support_tickets_3m"] == 1
        assert result[1]["avg_resolution_days"] == 2.0
