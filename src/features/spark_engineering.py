"""PySpark-based feature engineering for Databricks.

This module mirrors `engineering.py` but uses PySpark for distributed
processing of large datasets (1M+ rows).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


def get_spark() -> "SparkSession":
    """Return an active SparkSession (works both locally and on Databricks)."""
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder
        .appName("churn-feature-engineering")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def build_transaction_features_spark(
    spark: "SparkSession",
    transactions_path: str,
    lookback_months: int = 3,
) -> "DataFrame":
    """Distributed transaction feature aggregation using PySpark.

    Args:
        spark: Active SparkSession.
        transactions_path: Path to raw transactions (CSV, Parquet, or Delta).
        lookback_months: Rolling window in months.

    Returns:
        Spark DataFrame with one row per customer_id.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType

    df = spark.read.format("delta" if "delta" in transactions_path else "csv") \
        .option("header", True) \
        .option("inferSchema", True) \
        .load(transactions_path)

    # Filter to lookback window
    max_date = df.agg(F.max("transaction_date")).collect()[0][0]
    cutoff = F.add_months(F.lit(max_date), -lookback_months)
    recent = df.filter(F.col("transaction_date") >= cutoff)

    agg = recent.groupBy("customer_id").agg(
        F.sum("amount").alias("total_spending_3m"),
        F.avg("amount").alias("avg_monthly_charge"),
        F.count("transaction_id").alias("transaction_count_3m"),
        (
            F.sum((F.col("payment_status") == "failed").cast("int")) /
            F.count("transaction_id")
        ).cast(DoubleType()).alias("payment_failure_rate"),
    )

    logger.info("Spark: transaction feature agg complete")
    return agg


def build_support_features_spark(
    spark: "SparkSession",
    interactions_path: str,
) -> "DataFrame":
    """Distributed support interaction feature aggregation."""
    from pyspark.sql import functions as F

    df = spark.read.format("delta" if "delta" in interactions_path else "csv") \
        .option("header", True) \
        .option("inferSchema", True) \
        .load(interactions_path)

    cutoff = df.agg(F.max("ticket_date")).collect()[0][0]

    agg = df.groupBy("customer_id").agg(
        F.count("interaction_id").alias("support_tickets_3m"),
        F.avg("resolution_days").alias("avg_resolution_days"),
        F.avg("sentiment_score").alias("avg_sentiment"),
    )
    return agg


def build_and_save_feature_table(
    customers_path: str,
    transactions_path: str,
    interactions_path: str,
    labels_path: str,
    output_path: str,
) -> None:
    """End-to-end PySpark feature engineering pipeline.

    Reads raw data, engineers features, and writes output as a Delta table.

    Args:
        customers_path: Path to raw customers data.
        transactions_path: Path to raw transactions data.
        interactions_path: Path to raw support interactions data.
        labels_path: Path to churn labels.
        output_path: Delta Lake path to write the feature table.
    """
    from pyspark.sql import functions as F

    spark = get_spark()

    customers = spark.read.csv(customers_path, header=True, inferSchema=True)
    labels = spark.read.csv(labels_path, header=True, inferSchema=True)

    tx_feats = build_transaction_features_spark(spark, transactions_path)
    sup_feats = build_support_features_spark(spark, interactions_path)

    features = (
        customers.select("customer_id", "tenure_months", "contract_type")
        .join(tx_feats, on="customer_id", how="left")
        .join(sup_feats, on="customer_id", how="left")
        .join(labels.select("customer_id", "churned"), on="customer_id", how="left")
    )

    # Record count for CV / README documentation
    row_count = features.count()
    logger.info("Feature table row count: %d", row_count)
    print(f"[INFO] Feature table written to {output_path} | Rows: {row_count:,}")

    features.write.format("delta").mode("overwrite").save(output_path)
