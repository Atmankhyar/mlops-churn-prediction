"""Phase A validation of the Evidently drift-monitoring pipeline.

Reference data is the real training feature table (data/raw -> the same
build_full_feature_table + full_clean_pipeline used by scripts/train_local.py).

There is no production inference log yet (the API does not persist request
data), so the "current" batch here is SYNTHETIC: a sample of the reference
data with a couple of features deliberately perturbed to simulate drift.
This validates the pipeline only -- it is not a real production drift result.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.cleaning import full_clean_pipeline  # noqa: E402
from src.features.engineering import build_full_feature_table  # noqa: E402
from src.monitoring.drift import check_drift_threshold, generate_drift_report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")


def main() -> None:
    logger.info(
        "Loading raw data and building the reference feature table (real training data) ..."
    )
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    interactions = pd.read_csv(DATA_DIR / "support_interactions.csv")
    labels = pd.read_csv(DATA_DIR / "churn_labels.csv")

    reference = build_full_feature_table(customers, transactions, interactions, labels)
    reference = full_clean_pipeline(reference)
    logger.info("Reference table shape: %s", reference.shape)

    logger.warning(
        "SYNTHETIC TEST DATA: 'current' batch below is a perturbed sample of the "
        "reference data, built only to exercise the drift pipeline. It is NOT "
        "real production inference data and must not be reported as such."
    )
    current = reference.sample(frac=0.3, random_state=42).copy()
    # Deliberately shift two features to simulate drift; leave the rest untouched
    # as a control group that should NOT be flagged.
    current["avg_monthly_charge"] = current["avg_monthly_charge"] * 1.6 + 25
    current["payment_failure_rate"] = (current["payment_failure_rate"] + 0.5).clip(upper=1.0)

    report_path = generate_drift_report(reference, current)
    logger.info("HTML drift report saved to: %s", report_path)

    results = check_drift_threshold(reference, current)
    logger.info("Per-feature drift results (SYNTHETIC input): %s", results)


if __name__ == "__main__":
    main()
