"""Data cleaning utilities."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def drop_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    logger.info("Dropped %d duplicate rows", before - len(df))
    return df


def impute_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values.

    - Numerical columns: median imputation.
    - Categorical columns: mode imputation.
    """
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            logger.info("Imputed %d nulls in '%s' with median=%.4f", null_count, col, fill_val)
        else:
            fill_val = df[col].mode().iloc[0]
            df[col] = df[col].fillna(fill_val)
            logger.info("Imputed %d nulls in '%s' with mode='%s'", null_count, col, fill_val)
    return df


def clip_outliers(df: pd.DataFrame, columns: list[str], n_std: float = 4.0) -> pd.DataFrame:
    """Clip extreme values beyond n_std standard deviations from the mean."""
    for col in columns:
        if col not in df.columns:
            continue
        mean, std = df[col].mean(), df[col].std()
        lower, upper = mean - n_std * std, mean + n_std * std
        clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        if clipped > 0:
            logger.info("Clipped %d outliers in '%s'", clipped, col)
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct dtypes for known columns."""
    bool_cols = ["churned"]
    int_cols = ["tenure_months", "transaction_count_3m", "support_tickets_3m"]
    float_cols = [
        "total_spending_3m",
        "avg_monthly_charge",
        "payment_failure_rate",
        "avg_resolution_days",
        "avg_sentiment",
    ]

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    return df


def full_clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline in sequence."""
    numerical_cols = [
        "total_spending_3m",
        "avg_monthly_charge",
        "payment_failure_rate",
        "transaction_count_3m",
        "support_tickets_3m",
        "avg_resolution_days",
        "avg_sentiment",
        "tenure_months",
    ]
    df = drop_duplicates(df, subset=["customer_id"])
    df = impute_nulls(df)
    df = clip_outliers(df, numerical_cols)
    df = cast_types(df)
    return df
