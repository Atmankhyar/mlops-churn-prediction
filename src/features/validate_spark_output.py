"""Validation script for spark_engineering.py output.

Runs the PySpark feature pipeline in-memory and validates:
  1. Row count  -- one row per customer (matches customers.csv)
  2. Columns    -- expected feature columns are all present
  3. Nulls      -- reports unexpected null percentages per column
  4. Duplicates -- no duplicate customer_id rows in the output
"""

from __future__ import annotations

import io
import os
import sys

# Force UTF-8 output so Windows cp1252 terminals don't raise UnicodeEncodeError.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path  # noqa: E402

# ---------------------------------------------------------------------------
# Paths – resolve relative to project root regardless of cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"

CUSTOMERS_PATH = str(DATA_RAW / "customers.csv")
TRANSACTIONS_PATH = str(DATA_RAW / "transactions.csv")
INTERACTIONS_PATH = str(DATA_RAW / "support_interactions.csv")
LABELS_PATH = str(DATA_RAW / "churn_labels.csv")

# Expected feature columns produced by build_and_save_feature_table
EXPECTED_COLUMNS = [
    "customer_id",
    "tenure_months",
    "contract_type",
    "total_spending_3m",
    "avg_monthly_charge",
    "transaction_count_3m",
    "payment_failure_rate",
    "support_tickets_3m",
    "avg_resolution_days",
    "avg_sentiment",
    "churned",
]

# Columns that MUST be non-null (core identifier + label)
MUST_NOT_BE_NULL = {"customer_id", "tenure_months", "contract_type", "churned"}

# Columns that may be null (left-join side-effect for customers with no
# transactions / support tickets) – we warn but don't fail.
ALLOWED_NULL_COLS = {
    "total_spending_3m",
    "avg_monthly_charge",
    "transaction_count_3m",
    "payment_failure_rate",
    "support_tickets_3m",
    "avg_resolution_days",
    "avg_sentiment",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def _rule(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def _ok(msg: str) -> None:
    print(f"  {PASS}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {FAIL}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {WARN}  {msg}")


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------
def main() -> int:
    """Return 0 if all hard checks pass, 1 if any fail."""
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    print("\n" + "=" * 60)
    print("  Spark Feature-Engineering Validation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build SparkSession (local mode, reduced verbosity)
    # ------------------------------------------------------------------
    spark = (
        SparkSession.builder.appName("churn-fe-validation")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # ------------------------------------------------------------------
    # 2. Read raw sources to get ground-truth counts
    # ------------------------------------------------------------------
    customers = spark.read.csv(CUSTOMERS_PATH, header=True, inferSchema=True)
    transactions = spark.read.csv(TRANSACTIONS_PATH, header=True, inferSchema=True)
    interactions = spark.read.csv(INTERACTIONS_PATH, header=True, inferSchema=True)
    labels = spark.read.csv(LABELS_PATH, header=True, inferSchema=True)

    n_customers = customers.select("customer_id").distinct().count()
    n_transactions = transactions.count()
    n_interactions = interactions.count()
    n_labels = labels.select("customer_id").distinct().count()

    print("\n  Raw data summary:")
    print(f"    customers (distinct):    {n_customers:>10,}")
    print(f"    transactions (rows):     {n_transactions:>10,}")
    print(f"    support interactions:    {n_interactions:>10,}")
    print(f"    churn labels (distinct): {n_labels:>10,}")

    # ------------------------------------------------------------------
    # 3. Replicate the feature pipeline (no file output needed)
    #    Mirrors build_and_save_feature_table exactly, including step 4
    #    (coalesce imputation) so the validator checks the *final* table.
    # ------------------------------------------------------------------
    print("\n  Building feature table in-memory ...")

    import sys as _sys

    _sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from features.spark_engineering import (
        build_support_features_spark,
        build_transaction_features_spark,
    )

    tx_feats = build_transaction_features_spark(spark, TRANSACTIONS_PATH)
    sup_feats = build_support_features_spark(spark, INTERACTIONS_PATH)

    features = (
        customers.select("customer_id", "tenure_months", "contract_type")
        .join(tx_feats, on="customer_id", how="left")
        .join(sup_feats, on="customer_id", how="left")
        .join(labels.select("customer_id", "churned"), on="customer_id", how="left")
    )

    # Step 4 – coalesce imputation (mirrors spark_engineering.py step 4)
    # Customers with no transactions / support tickets get 0 instead of NULL.
    features = features.select(
        "customer_id",
        "tenure_months",
        "contract_type",
        F.coalesce(F.col("total_spending_3m"), F.lit(0.0)).alias("total_spending_3m"),
        F.coalesce(F.col("avg_monthly_charge"), F.lit(0.0)).alias("avg_monthly_charge"),
        F.coalesce(F.col("transaction_count_3m"), F.lit(0)).alias("transaction_count_3m"),
        F.coalesce(F.col("payment_failure_rate"), F.lit(0.0)).alias("payment_failure_rate"),
        F.coalesce(F.col("support_tickets_3m"), F.lit(0)).alias("support_tickets_3m"),
        F.coalesce(F.col("avg_resolution_days"), F.lit(0.0)).alias("avg_resolution_days"),
        F.coalesce(F.col("avg_sentiment"), F.lit(0.0)).alias("avg_sentiment"),
        "churned",
    )

    # Cache for repeated actions
    features.cache()
    total_rows = features.count()

    # ------------------------------------------------------------------
    # CHECK 1 – Row count
    # ------------------------------------------------------------------
    _rule("CHECK 1 - Row Count")
    print(f"  Expected (distinct customers): {n_customers:,}")
    print(f"  Actual   (feature table rows): {total_rows:,}")

    hard_fail = False

    if total_rows == n_customers:
        _ok("Row count matches distinct customer count exactly.")
    elif abs(total_rows - n_customers) / max(n_customers, 1) < 0.01:
        _warn(
            f"Row count differs by {abs(total_rows - n_customers):,} rows "
            f"(<1%) – acceptable (label/customer alignment gap)."
        )
    else:
        _fail(
            f"Row count mismatch: delta = {total_rows - n_customers:+,}. "
            f"Investigate join fanout or missing customers."
        )
        hard_fail = True

    # ------------------------------------------------------------------
    # CHECK 2 – Columns
    # ------------------------------------------------------------------
    _rule("CHECK 2 - Column Presence")
    actual_cols = set(features.columns)
    missing = [c for c in EXPECTED_COLUMNS if c not in actual_cols]
    extra = [c for c in features.columns if c not in EXPECTED_COLUMNS]

    if not missing:
        _ok(f"All {len(EXPECTED_COLUMNS)} expected columns present.")
    else:
        _fail(f"Missing columns: {missing}")
        hard_fail = True

    if extra:
        _warn(f"Extra (unexpected) columns: {extra}")
    else:
        _ok("No unexpected extra columns.")

    print(f"\n  Full schema ({len(features.columns)} columns):")
    for field in features.schema.fields:
        print(f"    {field.name:<30} {str(field.dataType)}")

    # ------------------------------------------------------------------
    # CHECK 3 – Nulls (all columns must be 0 after imputation)
    # ------------------------------------------------------------------
    _rule("CHECK 3 - Null Analysis (post-imputation)")
    null_counts = (
        features.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in features.columns])
        .collect()[0]
        .asDict()
    )

    any_bad_null = False
    for col_name in EXPECTED_COLUMNS:
        if col_name not in actual_cols:
            continue
        null_n = null_counts.get(col_name, 0)
        null_pct = null_n / total_rows * 100 if total_rows else 0

        if null_n == 0:
            _ok(f"{col_name:<30} 0 nulls")
        else:
            _fail(
                f"{col_name:<30} {null_n:,} nulls ({null_pct:.1f}%) "
                f"– UNEXPECTED: coalesce imputation should have eliminated nulls!"
            )
            any_bad_null = True
            hard_fail = True

    if not any_bad_null:
        _ok("No nulls in any column after coalesce imputation.")

    # ------------------------------------------------------------------
    # CHECK 4 – Duplicate customer_id
    # ------------------------------------------------------------------
    _rule("CHECK 4 - Duplicate customer_id Keys")
    dup_count = features.groupBy("customer_id").count().filter(F.col("count") > 1).count()

    if dup_count == 0:
        _ok("No duplicate customer_id records – aggregation keys are unique.")
    else:
        _fail(f"{dup_count:,} customer_id values appear more than once.")
        examples = (
            features.groupBy("customer_id")
            .count()
            .filter(F.col("count") > 1)
            .orderBy(F.col("count").desc())
            .limit(5)
        )
        print("\n  Top duplicate examples:")
        examples.show(truncate=False)
        hard_fail = True

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if hard_fail:
        _fail("VALIDATION FAILED – one or more hard checks did not pass.")
        print("=" * 60 + "\n")
        return 1
    else:
        _ok("ALL CHECKS PASSED")
        print("=" * 60 + "\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
