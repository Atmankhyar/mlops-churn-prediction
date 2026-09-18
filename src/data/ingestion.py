"""Data ingestion utilities.

Supports loading data from:
  - Local CSV / Parquet files
  - Snowflake (via snowflake-connector-python)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path, **kwargs)


def load_parquet(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame."""
    return pd.read_parquet(path, **kwargs)


def load_from_snowflake(
    query: str,
    account: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    warehouse: Optional[str] = None,
) -> pd.DataFrame:
    """Execute a SQL query against Snowflake and return a DataFrame.

    Credentials are read from environment variables when not supplied
    explicitly.
    """
    try:
        import snowflake.connector
    except ImportError as exc:
        raise ImportError(
            "snowflake-connector-python is required: pip install snowflake-connector-python"
        ) from exc

    conn = snowflake.connector.connect(
        account=account or os.environ["SNOWFLAKE_ACCOUNT"],
        user=user or os.environ["SNOWFLAKE_USER"],
        password=password or os.environ["SNOWFLAKE_PASSWORD"],
        database=database or os.environ.get("SNOWFLAKE_DATABASE", "MLOPS_DB"),
        schema=schema or os.environ.get("SNOWFLAKE_SCHEMA", "ANALYTICS"),
        warehouse=warehouse or os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    )
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df


def load_customer_features_from_snowflake() -> pd.DataFrame:
    """Convenience function: load the CUSTOMER_FEATURES table."""
    query = """
        SELECT
            customer_id,
            tenure_months,
            contract_type,
            total_spending_3m,
            avg_monthly_charge,
            payment_failure_rate,
            transaction_count_3m,
            support_tickets_3m,
            avg_resolution_days,
            avg_sentiment,
            churned
        FROM MLOPS_DB.ANALYTICS.CUSTOMER_FEATURES
        WHERE feature_month = (
            SELECT MAX(feature_month) FROM MLOPS_DB.ANALYTICS.CUSTOMER_FEATURES
        )
    """
    return load_from_snowflake(query)
