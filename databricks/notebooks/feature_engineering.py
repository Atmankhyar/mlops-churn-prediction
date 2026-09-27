# Databricks notebook source
# ============================================================
# Feature Engineering Pipeline — Self-Contained Upload Version
#
# Upload this file directly to your Databricks workspace:
#   Workspace → Import → select this .py file
#
# Attach to: Serverless compute  OR  Runtime 13.3 LTS ML +
# ============================================================

# COMMAND ----------
# MAGIC %md
# MAGIC # Feature Engineering Pipeline
# MAGIC
# MAGIC Reads raw customer, transaction, and support-interaction data,
# MAGIC engineers a validated feature table, and writes it as Delta Lake.
# MAGIC
# MAGIC | Step | Description |
# MAGIC |------|-------------|
# MAGIC | 1 | Load raw CSVs |
# MAGIC | 2 | Transaction features — 3-month rolling aggregation |
# MAGIC | 3 | Support interaction features |
# MAGIC | 4 | LEFT JOIN all sources on `customer_id` |
# MAGIC | 5 | Impute NULLs → 0 via `coalesce` |
# MAGIC | 6 | Validate row count + assert zero NULLs |
# MAGIC | 7 | Write to Delta Lake |
# MAGIC | 8 | Spot-check output |

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cell 1 · Feature-engineering functions
# MAGIC
# MAGIC Sourced from `src/features/spark_engineering.py` in the local repo.
# MAGIC When you update that file locally, paste the updated version here
# MAGIC (or switch to Databricks Repos to sync automatically).

# COMMAND ----------

"""
PySpark feature engineering functions.
Source: src/features/spark_engineering.py
"""
from __future__ import annotations
import logging
from pyspark.sql import functions as F, SparkSession, DataFrame
from pyspark.sql.types import DoubleType

logger = logging.getLogger(__name__)


def build_transaction_features_spark(
    spark: SparkSession,
    transactions_path: str,
    lookback_months: int = 3,
) -> DataFrame:
    """Distributed transaction feature aggregation.

    Returns one row per customer_id with 3-month rolling metrics.
    """
    df = (
        spark.read
        .format("delta" if "delta" in transactions_path else "csv")
        .option("header", True)
        .option("inferSchema", True)
        .load(transactions_path)
    )

    # Filter to the lookback window
    max_date = df.agg(F.max("transaction_date")).collect()[0][0]
    cutoff   = F.add_months(F.lit(max_date), -lookback_months)
    recent   = df.filter(F.col("transaction_date") >= cutoff)

    agg = recent.groupBy("customer_id").agg(
        F.sum("amount").alias("total_spending_3m"),
        F.avg("amount").alias("avg_monthly_charge"),
        F.count("transaction_id").alias("transaction_count_3m"),
        (
            F.sum((F.col("payment_status") == "failed").cast("int")) /
            F.count("transaction_id")
        ).cast(DoubleType()).alias("payment_failure_rate"),
    )

    logger.info("Transaction feature aggregation complete.")
    return agg


def build_support_features_spark(
    spark: SparkSession,
    interactions_path: str,
) -> DataFrame:
    """Distributed support-interaction feature aggregation.

    Returns one row per customer_id.
    """
    df = (
        spark.read
        .format("delta" if "delta" in interactions_path else "csv")
        .option("header", True)
        .option("inferSchema", True)
        .load(interactions_path)
    )

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

    Steps:
        1. Load customers + churn labels
        2. Build transaction features  (3-month lookback)
        3. Build support features
        4. LEFT JOIN all sources on customer_id
        5. Impute NULLs → 0 via coalesce
        6. Validate: row count + zero-null assertion
        7. Write to Delta Lake
    """
    # On Databricks the SparkSession is already available as `spark`.
    # getOrCreate() returns the existing session without creating a new one.
    _spark = SparkSession.builder.getOrCreate()

    # ------------------------------------------------------------------
    # 1. Load customers and churn labels
    # ------------------------------------------------------------------
    customers = _spark.read.csv(customers_path, header=True, inferSchema=True)
    labels    = _spark.read.csv(labels_path,    header=True, inferSchema=True)

    # ------------------------------------------------------------------
    # 2 & 3. Build feature groups
    # ------------------------------------------------------------------
    tx_feats  = build_transaction_features_spark(_spark, transactions_path)
    sup_feats = build_support_features_spark(_spark, interactions_path)

    # ------------------------------------------------------------------
    # 4. JOIN
    # ------------------------------------------------------------------
    features = (
        customers.select("customer_id", "tenure_months", "contract_type")
        .join(tx_feats,  on="customer_id", how="left")
        .join(sup_feats, on="customer_id", how="left")
        .join(labels.select("customer_id", "churned"), on="customer_id", how="left")
    )

    # ------------------------------------------------------------------
    # 5. Impute NULLs — customers with no activity in window → 0
    # ------------------------------------------------------------------
    features = features.select(
        "customer_id",
        "tenure_months",
        "contract_type",
        F.coalesce(F.col("total_spending_3m"),   F.lit(0.0)).alias("total_spending_3m"),
        F.coalesce(F.col("avg_monthly_charge"),   F.lit(0.0)).alias("avg_monthly_charge"),
        F.coalesce(F.col("transaction_count_3m"), F.lit(0)  ).alias("transaction_count_3m"),
        F.coalesce(F.col("payment_failure_rate"), F.lit(0.0)).alias("payment_failure_rate"),
        F.coalesce(F.col("support_tickets_3m"),   F.lit(0)  ).alias("support_tickets_3m"),
        F.coalesce(F.col("avg_resolution_days"),  F.lit(0.0)).alias("avg_resolution_days"),
        F.coalesce(F.col("avg_sentiment"),        F.lit(0.0)).alias("avg_sentiment"),
        "churned",
    )

    # ------------------------------------------------------------------
    # 6a. Validate — row count
    # ------------------------------------------------------------------
    row_count = features.count()
    print(f"[INFO] Feature table rows: {row_count:,}")

    # ------------------------------------------------------------------
    # 6b. Validate — zero NULLs
    # ------------------------------------------------------------------
    null_counts = features.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in features.columns]
    ).collect()[0].asDict()

    total_nulls = sum(v or 0 for v in null_counts.values())
    print("\n[NULL VALIDATION]")
    for col, cnt in null_counts.items():
        print(f"  {col}: {cnt}")

    if total_nulls > 0:
        raise ValueError(
            f"Feature table still contains {total_nulls} NULL values after imputation."
        )
    print("[PASS] No NULL values detected.")

    # ------------------------------------------------------------------
    # 7. Write to Delta Lake
    # ------------------------------------------------------------------
    features.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(output_path)

    print(f"\n[PASS] Feature table written to: {output_path}")
    print(f"[INFO] Rows written: {row_count:,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cell 2 · Configuration
# MAGIC
# MAGIC Edit the paths below to match your DBFS mount or Unity Catalog Volume.

# COMMAND ----------

# ------------------------------------------------------------------
# DBFS mount paths (classic clusters)
# ------------------------------------------------------------------
RAW_CUSTOMERS_PATH    = "/mnt/data/raw/customers.csv"
RAW_TRANSACTIONS_PATH = "/mnt/data/raw/transactions.csv"
RAW_INTERACTIONS_PATH = "/mnt/data/raw/support_interactions.csv"
RAW_LABELS_PATH       = "/mnt/data/raw/churn_labels.csv"
DELTA_OUTPUT_PATH     = "/mnt/delta/customer_features"

# ------------------------------------------------------------------
# Unity Catalog Volume paths — uncomment if using UC Serverless
# ------------------------------------------------------------------
# RAW_CUSTOMERS_PATH    = "/Volumes/mlops/raw/customers/customers.csv"
# RAW_TRANSACTIONS_PATH = "/Volumes/mlops/raw/transactions/transactions.csv"
# RAW_INTERACTIONS_PATH = "/Volumes/mlops/raw/support/support_interactions.csv"
# RAW_LABELS_PATH       = "/Volumes/mlops/raw/labels/churn_labels.csv"
# DELTA_OUTPUT_PATH     = "/Volumes/mlops/features/customer_features"

print("Paths:")
print(f"  customers    → {RAW_CUSTOMERS_PATH}")
print(f"  transactions → {RAW_TRANSACTIONS_PATH}")
print(f"  interactions → {RAW_INTERACTIONS_PATH}")
print(f"  labels       → {RAW_LABELS_PATH}")
print(f"  output       → {DELTA_OUTPUT_PATH}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cell 3 · Run the pipeline

# COMMAND ----------

build_and_save_feature_table(
    customers_path    = RAW_CUSTOMERS_PATH,
    transactions_path = RAW_TRANSACTIONS_PATH,
    interactions_path = RAW_INTERACTIONS_PATH,
    labels_path       = RAW_LABELS_PATH,
    output_path       = DELTA_OUTPUT_PATH,
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cell 4 · Spot-check the Delta output

# COMMAND ----------

feature_df = spark.read.format("delta").load(DELTA_OUTPUT_PATH)

print(f"Rows : {feature_df.count():,}")
print(f"Cols : {len(feature_df.columns)}")
feature_df.printSchema()

# COMMAND ----------

# Null audit
null_audit = feature_df.select(
    [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in feature_df.columns]
)
print("Null counts per column (all must be 0):")
null_audit.show(vertical=True)

# COMMAND ----------

# Duplicate key check
dup_count = (
    feature_df.groupBy("customer_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)
assert dup_count == 0, f"Found {dup_count:,} duplicate customer_id rows!"
print(f"Duplicate customer_id rows: {dup_count}  [PASS]")

# COMMAND ----------

# Summary statistics
display(
    feature_df.select(
        "total_spending_3m", "avg_monthly_charge", "transaction_count_3m",
        "payment_failure_rate", "support_tickets_3m", "avg_resolution_days",
        "avg_sentiment",
    ).summary()
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cell 5 · (Optional) Register in Unity Catalog

# COMMAND ----------

# UC_CATALOG = "mlops"
# UC_SCHEMA  = "features"
# UC_TABLE   = "customer_features"
#
# spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")
# spark.sql(f"""
#     CREATE TABLE IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}.{UC_TABLE}
#     USING DELTA
#     LOCATION '{DELTA_OUTPUT_PATH}'
# """)
# display(spark.sql(f"SELECT * FROM {UC_CATALOG}.{UC_SCHEMA}.{UC_TABLE} LIMIT 5"))
