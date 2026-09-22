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
    role: Optional[str] = None,
    lowercase_columns: bool = False,
) -> pd.DataFrame:
    """Execute a SQL query against Snowflake and return a DataFrame.

    Credentials are read from environment variables when not supplied
    explicitly.

    Args:
        query: SQL to execute.
        lowercase_columns:
            Snowflake returns unquoted identifiers in UPPER CASE. Set this to
            True to fold them to lower case, which is the convention the rest
            of the codebase (configs/model_config.yaml, src.data.cleaning)
            expects.
    """
    try:
        import snowflake.connector
        from snowflake.connector.errors import NotSupportedError
    except ImportError as exc:
        raise ImportError(
            "snowflake-connector-python is required: pip install snowflake-connector-python"
        ) from exc

    conn = snowflake.connector.connect(
        account=account or os.environ["SNOWFLAKE_ACCOUNT"],
        user=user or os.environ["SNOWFLAKE_USER"],
        password=password or os.environ["SNOWFLAKE_PASSWORD"],
        database=database or os.environ.get("SNOWFLAKE_DATABASE", "mlops2_db"),
        schema=schema or os.environ.get("SNOWFLAKE_SCHEMA", "ANALYTICS"),
        warehouse=warehouse or os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        role=role or os.environ.get("SNOWFLAKE_ROLE"),
    )
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            # Arrow-based fetch is dramatically faster than pd.read_sql for the
            # 100K-row feature table; fall back when the arrow extra is absent.
            try:
                df = cursor.fetch_pandas_all()
            except NotSupportedError:
                df = pd.DataFrame(
                    cursor.fetchall(),
                    columns=[c[0] for c in cursor.description],
                )
        finally:
            cursor.close()
    finally:
        conn.close()

    if lowercase_columns:
        df.columns = [c.lower() for c in df.columns]

    return df


def load_customer_features_from_snowflake(
    database: Optional[str] = None,
    schema: Optional[str] = None,
    table: str = "CUSTOMER_FEATURES",
    feature_month: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load the CUSTOMER_FEATURES table for a single feature month.

    Defaults to the most recent ``feature_month`` present in the table so that
    training never mixes snapshots from different months.

    Returns:
        Tuple of (DataFrame with lower-cased columns, provenance dict). The
        provenance dict carries the database / schema / table actually read,
        for logging as MLflow tags.
    """
    database = database or os.environ.get("SNOWFLAKE_DATABASE", "mlops2_db")
    schema = schema or os.environ.get("SNOWFLAKE_SCHEMA", "ANALYTICS")
    fqn = f"{database}.{schema}.{table}".upper()

    if feature_month:
        month_predicate = f"feature_month = '{feature_month}'"
    else:
        month_predicate = f"feature_month = (SELECT MAX(feature_month) FROM {fqn})"

    query = f"""
        SELECT
            customer_id,
            feature_month,
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
        FROM {fqn}
        WHERE {month_predicate}
    """

    df = load_from_snowflake(query, lowercase_columns=True)

    provenance = {
        "data_source": "Snowflake",
        "account": os.environ.get("SNOWFLAKE_ACCOUNT", ""),
        "database": database.upper(),
        "schema": schema.upper(),
        "table": table.upper(),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", ""),
    }
    return df, provenance
