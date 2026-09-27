"""Privacy-conscious SQLite logging for successful churn predictions."""
from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("monitoring/inference_log.db")

MONITORED_FEATURES = [
    "tenure_months",
    "total_spending_3m",
    "avg_monthly_charge",
    "payment_failure_rate",
    "avg_sentiment",
]

LOG_COLUMNS = [
    "timestamp",
    "model_version",
    *MONITORED_FEATURES,
    "churn_probability",
    "predicted_churn",
]


def get_db_path() -> Path:
    """Resolve the database path, allowing Docker/environment overrides."""
    configured = os.getenv("INFERENCE_LOG_DB", str(DEFAULT_DB_PATH))
    return Path(configured)


def _connect() -> sqlite3.Connection:
    """Open a new SQLite connection with WAL mode and a busy timeout."""
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


@contextmanager
def _connection():
    """Short-lived connection that commits/rolls back and always closes.

    sqlite3's own ``with conn:`` only manages the transaction, not the
    connection lifecycle -- it never closes the connection. Left as-is,
    a long-running API process would leak a connection per call.
    """
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database() -> None:
    """Create the inference log table if it does not exist."""
    with _connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_version TEXT NOT NULL,
                tenure_months REAL NOT NULL,
                total_spending_3m REAL NOT NULL,
                avg_monthly_charge REAL NOT NULL,
                payment_failure_rate REAL NOT NULL,
                avg_sentiment REAL NOT NULL,
                churn_probability REAL NOT NULL,
                predicted_churn INTEGER NOT NULL
                    CHECK (predicted_churn IN (0, 1))
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_inference_logs_timestamp
            ON inference_logs(timestamp)
            """
        )


def _utc_timestamp(value: datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp."""
    value = value or datetime.now(timezone.utc)

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc).isoformat()


def log_prediction(
    features: dict,
    probability: float,
    prediction: int,
    model_version: str,
    timestamp: datetime | None = None,
) -> None:
    """Persist one successful prediction without customer identifiers."""
    record = {
        "timestamp": _utc_timestamp(timestamp),
        "model_version": str(model_version),
        **{name: float(features[name]) for name in MONITORED_FEATURES},
        "churn_probability": float(probability),
        "predicted_churn": int(prediction),
    }

    if not 0.0 <= record["churn_probability"] <= 1.0:
        raise ValueError("Invalid churn probability")

    if record["predicted_churn"] not in (0, 1):
        raise ValueError("Invalid prediction label")

    with _connection() as conn:
        conn.execute(
            """
            INSERT INTO inference_logs (
                timestamp,
                model_version,
                tenure_months,
                total_spending_3m,
                avg_monthly_charge,
                payment_failure_rate,
                avg_sentiment,
                churn_probability,
                predicted_churn
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(record[column] for column in LOG_COLUMNS),
        )


def log_predictions(records: list[dict]) -> None:
    """Insert a batch atomically, after the API batch succeeds."""
    if not records:
        return

    rows = [tuple(record[column] for column in LOG_COLUMNS) for record in records]

    with _connection() as conn:
        conn.executemany(
            """
            INSERT INTO inference_logs (
                timestamp,
                model_version,
                tenure_months,
                total_spending_3m,
                avg_monthly_charge,
                payment_failure_rate,
                avg_sentiment,
                churn_probability,
                predicted_churn
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def cleanup_old_records(retention_days: int = 30) -> int:
    """Delete records older than the configured retention period."""
    if retention_days < 1:
        raise ValueError("retention_days must be at least 1")

    cutoff = _utc_timestamp(datetime.now(timezone.utc) - timedelta(days=retention_days))

    with _connection() as conn:
        cursor = conn.execute(
            "DELETE FROM inference_logs WHERE timestamp < ?",
            (cutoff,),
        )
        return cursor.rowcount


def read_recent_inference(window_days: int = 7) -> pd.DataFrame:
    """Return recent inference features and predictions."""
    if window_days < 1:
        raise ValueError("window_days must be at least 1")

    cutoff = _utc_timestamp(datetime.now(timezone.utc) - timedelta(days=window_days))

    with _connection() as conn:
        return pd.read_sql_query(
            """
            SELECT
                timestamp,
                model_version,
                tenure_months,
                total_spending_3m,
                avg_monthly_charge,
                payment_failure_rate,
                avg_sentiment,
                churn_probability,
                predicted_churn
            FROM inference_logs
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
            """,
            conn,
            params=(cutoff,),
        )
