"""Drift detection using Evidently AI.

Generates HTML reports comparing reference (training) data distribution
to current (production) data distribution.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "monitoring_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: str | None = None,
) -> Path:
    """Generate an Evidently data drift HTML report.

    Args:
        reference: Training dataset (baseline distribution).
        current: Recent production prediction requests.
        output_dir: Directory to save the HTML report.

    Returns:
        Path to the generated HTML report.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    except ImportError as exc:
        raise ImportError(
            "evidentlyai is required: pip install evidentlyai"
        ) from exc

    config = load_config()
    features = config["drift"]["features_to_monitor"]

    # Select only monitored features present in both DataFrames. Cast to
    # plain float64: pandas nullable dtypes (e.g. Int64) upcast DataFrame.values
    # to object arrays, which breaks Evidently's internal numpy correlation
    # calculations (np.corrcoef/np.cov) in evidently==0.4.30.
    ref = reference[[c for c in features if c in reference.columns]].astype("float64")
    cur = current[[c for c in features if c in current.columns]].astype("float64")

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=ref, current_data=cur)

    out_dir = Path(output_dir or config["reports"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"drift_report_{date.today().isoformat()}.html"
    report.save_html(str(report_path))

    logger.info("Drift report saved to %s", report_path)
    return report_path


def check_drift_threshold(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict[str, bool]:
    """Check whether drift threshold is exceeded for each monitored feature.

    Returns a dict mapping feature_name -> drift_detected (bool).
    """
    try:
        from evidently.report import Report
        from evidently.metrics import ColumnDriftMetric
    except ImportError as exc:
        raise ImportError("evidentlyai required") from exc

    config = load_config()
    features = config["drift"]["features_to_monitor"]

    results = {}
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        report = Report(metrics=[ColumnDriftMetric(column_name=feature)])
        report.run(reference_data=reference, current_data=current)
        report_dict = report.as_dict()
        drift_detected = report_dict["metrics"][0]["result"]["drift_detected"]
        results[feature] = drift_detected
        if drift_detected:
            logger.warning("DRIFT DETECTED in feature: %s", feature)
    return results
