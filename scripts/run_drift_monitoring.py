"""Scheduled drift-monitoring runner.

Compares real, recently-logged inference requests
(src/monitoring/inference_logger.py) against the training reference
distribution, using the existing Evidently integration in
src/monitoring/drift.py.

This never substitutes synthetic data for missing production history: if
there isn't enough real inference data yet, monitoring is skipped and
reported as such rather than producing a report that looks authoritative
but isn't backed by real predictions.

Usage:
    python scripts/run_drift_monitoring.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.cleaning import full_clean_pipeline  # noqa: E402
from src.features.engineering import build_full_feature_table  # noqa: E402
from src.monitoring.drift import (  # noqa: E402
    check_drift_threshold,
    generate_drift_report,
    load_config,
)
from src.monitoring.inference_logger import read_recent_inference  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")

# Below this many recent predictions, Evidently's per-feature drift tests
# aren't statistically meaningful -- skip rather than produce a report that
# looks authoritative but isn't.
MIN_SAMPLE_SIZE = 30


def load_reference_features() -> pd.DataFrame:
    """Build the same feature table the model was trained on."""
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")
    interactions = pd.read_csv(DATA_DIR / "support_interactions.csv")
    labels = pd.read_csv(DATA_DIR / "churn_labels.csv")

    reference = build_full_feature_table(customers, transactions, interactions, labels)
    return full_clean_pipeline(reference)


def run_monitoring(min_sample_size: int = MIN_SAMPLE_SIZE) -> dict:
    """Run one drift-monitoring cycle against real logged inference data.

    Returns a status dict with keys: status, sample_size, report_path,
    drift_results. status is one of "skipped_no_data",
    "skipped_insufficient_data", or "completed".
    """
    config = load_config()
    window_days = int(config.get("inference_logging", {}).get("recent_window_days", 7))

    current = read_recent_inference(window_days=window_days)

    if len(current) == 0:
        logger.warning(
            "No recent inference data found in the last %d day(s). Monitoring skipped.",
            window_days,
        )
        return {
            "status": "skipped_no_data",
            "sample_size": 0,
            "report_path": None,
            "drift_results": None,
        }

    if len(current) < min_sample_size:
        logger.warning(
            "Only %d recent prediction(s) in the last %d day(s) (minimum %d required). "
            "Monitoring skipped -- a drift comparison on this few samples would not "
            "be meaningful.",
            len(current),
            window_days,
            min_sample_size,
        )
        return {
            "status": "skipped_insufficient_data",
            "sample_size": len(current),
            "report_path": None,
            "drift_results": None,
        }

    reference = load_reference_features()

    report_path = generate_drift_report(reference, current, label="production")
    drift_results = check_drift_threshold(reference, current)

    logger.info(
        "Monitoring complete: %d recent prediction(s) compared against %d reference row(s).",
        len(current),
        len(reference),
    )
    logger.info("Drift results: %s", drift_results)

    return {
        "status": "completed",
        "sample_size": len(current),
        "report_path": str(report_path),
        "drift_results": drift_results,
    }


def main() -> None:
    result = run_monitoring()
    logger.info("MONITORING_STATUS: %s", result["status"])


if __name__ == "__main__":
    main()
