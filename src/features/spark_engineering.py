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

    Reads raw data, engineers features, handles missing values,
    validates the final feature table, and writes it as Delta Lake.
    """
    from pyspark.sql import functions as F

    spark = get_spark()

    # ============================================================
    # 1. Load customers and churn labels
    # ============================================================

    customers = spark.read.csv(
        customers_path,
        header=True,
        inferSchema=True,
    )

    labels = spark.read.csv(
        labels_path,
        header=True,
        inferSchema=True,
    )

    # ============================================================
    # 2. Build transaction and support features
    # ============================================================

    tx_feats = build_transaction_features_spark(
        spark,
        transactions_path,
    )

    sup_feats = build_support_features_spark(
        spark,
        interactions_path,
    )

    # ============================================================
    # 3. Join all features
    # ============================================================

    features = (
        customers
        .select(
            "customer_id",
            "tenure_months",
            "contract_type",
        )
        .join(
            tx_feats,
            on="customer_id",
            how="left",
        )
        .join(
            sup_feats,
            on="customer_id",
            how="left",
        )
        .join(
            labels.select(
                "customer_id",
                "churned",
            ),
            on="customer_id",
            how="left",
        )
    )

    # ============================================================
    # 4. Handle NULL values created by LEFT JOINs
    # ============================================================
    #
    # Customers without transactions have NULL transaction features.
    # Customers without support interactions have NULL support features.
    #
    # These customers must NOT be removed.
    # We replace missing activity with 0.
    # ============================================================

    features = features.select(
        "customer_id",
        "tenure_months",
        "contract_type",

        # Transaction features
        F.coalesce(
            F.col("total_spending_3m"),
            F.lit(0.0),
        ).alias("total_spending_3m"),

        F.coalesce(
            F.col("avg_monthly_charge"),
            F.lit(0.0),
        ).alias("avg_monthly_charge"),

        F.coalesce(
            F.col("transaction_count_3m"),
            F.lit(0),
        ).alias("transaction_count_3m"),

        F.coalesce(
            F.col("payment_failure_rate"),
            F.lit(0.0),
        ).alias("payment_failure_rate"),

        # Support features
        F.coalesce(
            F.col("support_tickets_3m"),
            F.lit(0),
        ).alias("support_tickets_3m"),

        F.coalesce(
            F.col("avg_resolution_days"),
            F.lit(0.0),
        ).alias("avg_resolution_days"),

        F.coalesce(
            F.col("avg_sentiment"),
            F.lit(0.0),
        ).alias("avg_sentiment"),

        # Target
        "churned",
    )

    # ============================================================
    # 5. Validate row count
    # ============================================================

    row_count = features.count()

    logger.info(
        "Final feature table row count: %d",
        row_count,
    )

    print(
        f"[INFO] Final feature table rows: {row_count:,}"
    )

    # ============================================================
    # 6. Validate NULL values
    # ============================================================

    null_counts = features.select(
        [
            F.sum(
                F.col(column).isNull().cast("int")
            ).alias(column)
            for column in features.columns
        ]
    ).collect()[0].asDict()

    total_nulls = sum(
        value or 0
        for value in null_counts.values()
    )

    print("\n[NULL VALIDATION]")

    for column, count in null_counts.items():
        print(f"  {column}: {count}")

    if total_nulls > 0:
        raise ValueError(
            f"Feature table still contains {total_nulls} NULL values."
        )

    print("[PASS] No NULL values detected.")

    # ============================================================
    # 7. Write final feature table to Delta Lake
    # ============================================================

    features.write \
        .format("delta") \
        .mode("overwrite") \
        .save(output_path)

    print(
        f"\n[PASS] Feature table written to Delta Lake: {output_path}"
    )

    print(
        f"[INFO] Rows written: {row_count:,}"
    )