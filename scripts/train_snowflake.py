"""Train the churn model on features read straight from Snowflake (Phase 3).

Mirrors scripts/train_local.py, but the feature table comes from
MLOPS2_DB.ANALYTICS.CUSTOMER_FEATURES instead of data/raw/*.csv, and the run is
tagged with its Snowflake provenance so an MLflow run says not just *what* was
trained but *what it was trained on*.

Usage:
    python scripts/train_snowflake.py
    python scripts/train_snowflake.py --model xgboost --version v2.0
    python scripts/train_snowflake.py --model xgboost --feature-month 2026-09-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.cleaning import full_clean_pipeline  # noqa: E402
from src.data.ingestion import load_customer_features_from_snowflake  # noqa: E402
from src.models.train import run_training  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Identifiers and metadata: present in the warehouse table, never predictors.
NON_FEATURE_COLUMNS = ["customer_id", "feature_month", "created_at", "pipeline_version"]


def main(
    model_type: str,
    version: str,
    feature_month: str | None,
    table: str,
) -> None:
    logger.info("Loading features from Snowflake ...")

    df, provenance = load_customer_features_from_snowflake(
        table=table,
        feature_month=feature_month,
    )

    if df.empty:
        raise SystemExit(
            f"No rows returned from {provenance['database']}."
            f"{provenance['schema']}.{provenance['table']}"
            + (f" for feature_month={feature_month}" if feature_month else "")
        )

    loaded_month = str(df["feature_month"].iloc[0])

    logger.info(
        "Loaded %d rows x %d columns from %s.%s.%s (feature_month=%s)",
        len(df),
        df.shape[1],
        provenance["database"],
        provenance["schema"],
        provenance["table"],
        loaded_month,
    )
    logger.info("Churn rate: %.2f%%", df["churned"].astype(int).mean() * 100)

    # Clean before dropping customer_id — dedup keys on it.
    df = full_clean_pipeline(df)

    training_rows = len(df)

    # Separate target from predictors: drop identifiers and date fields.
    dropped = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    df_model = df.drop(columns=dropped)
    logger.info("Dropped non-feature columns: %s", ", ".join(dropped))
    logger.info("Training on %d rows", training_rows)

    tags = {
        **provenance,
        "feature_month": loaded_month,
        "training_pipeline": "snowflake",
    }

    extra_params = {
        "training_rows": training_rows,
        "source_table": (f"{provenance['database']}.{provenance['schema']}.{provenance['table']}"),
        "feature_month": loaded_month,
    }

    logger.info(
        "Starting MLflow training run (model=%s, version=%s) ...",
        model_type,
        version,
    )

    run_id = run_training(
        df=df_model,
        model_type=model_type,
        dataset_version=version,
        tags=tags,
        extra_params=extra_params,
    )

    logger.info("Training complete! MLflow run_id: %s", run_id)
    logger.info("View results at http://localhost:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the churn model on Snowflake features.")
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["random_forest", "xgboost", "xgboost_tuned"],
    )
    parser.add_argument("--version", default="v2.0")
    parser.add_argument(
        "--feature-month",
        default=None,
        help="YYYY-MM-DD snapshot to train on (default: latest in the table).",
    )
    parser.add_argument("--table", default="CUSTOMER_FEATURES")
    args = parser.parse_args()
    main(args.model, args.version, args.feature_month, args.table)
