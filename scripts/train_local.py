"""End-to-end local training script (Phase 1 + 2 milestone).

Usage:
    python scripts/train_local.py
    python scripts/train_local.py --model xgboost --version v1.1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.cleaning import full_clean_pipeline
from src.features.engineering import build_full_feature_table
from src.models.train import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")


def main(model_type: str, version: str) -> None:
    logger.info("Loading raw data from %s ...", DATA_DIR)

    customers = pd.read_csv(DATA_DIR / "customers.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    interactions = pd.read_csv(DATA_DIR / "support_interactions.csv")
    labels = pd.read_csv(DATA_DIR / "churn_labels.csv")

    logger.info(
        "Loaded: customers=%d, transactions=%d, interactions=%d",
        len(customers), len(transactions), len(interactions),
    )

    logger.info("Building feature table ...")
    features = build_full_feature_table(customers, transactions, interactions, labels)
    features = full_clean_pipeline(features)

    logger.info("Feature table shape: %s", features.shape)
    logger.info("Starting MLflow training run (model=%s, version=%s) ...", model_type, version)

    run_id = run_training(
        df=features,
        model_type=model_type,
        dataset_version=version,
    )
    logger.info("Training complete! MLflow run_id: %s", run_id)
    logger.info("View results at http://localhost:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "xgboost"])
    parser.add_argument("--version", default="v1.0")
    args = parser.parse_args()
    main(args.model, args.version)
