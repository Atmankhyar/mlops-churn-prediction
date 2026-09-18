"""Local (pandas-based) feature engineering.

Builds the CUSTOMER_FEATURES table from raw transactions, customers,
and support interactions.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_transaction_features(
    transactions: pd.DataFrame,
    lookback_months: int = 3,
) -> pd.DataFrame:
    """Aggregate transaction-level data to customer-level features.

    Args:
        transactions: Raw transactions DataFrame with columns:
            customer_id, transaction_date, amount, payment_status
        lookback_months: Window size for rolling aggregation.

    Returns:
        DataFrame with one row per customer_id.
    """
    transactions = transactions.copy()
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    cutoff = transactions["transaction_date"].max() - pd.DateOffset(months=lookback_months)
    recent = transactions[transactions["transaction_date"] >= cutoff]

    agg = (
        recent.groupby("customer_id")
        .agg(
            total_spending_3m=("amount", "sum"),
            avg_monthly_charge=("amount", "mean"),
            transaction_count_3m=("transaction_id", "count"),
            payment_failure_rate=(
                "payment_status",
                lambda s: (s == "failed").mean(),
            ),
        )
        .reset_index()
    )
    logger.info("Built transaction features for %d customers", len(agg))
    return agg


def build_support_features(interactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate support interaction data to customer-level features.

    Args:
        interactions: Raw interactions DataFrame with columns:
            customer_id, ticket_date, resolution_days, sentiment_score

    Returns:
        DataFrame with one row per customer_id.
    """
    interactions = interactions.copy()
    interactions["ticket_date"] = pd.to_datetime(interactions["ticket_date"])

    cutoff = interactions["ticket_date"].max() - pd.DateOffset(months=3)
    recent = interactions[interactions["ticket_date"] >= cutoff]

    agg = (
        recent.groupby("customer_id")
        .agg(
            support_tickets_3m=("interaction_id", "count"),
            avg_resolution_days=("resolution_days", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
        )
        .reset_index()
    )
    logger.info("Built support features for %d customers", len(agg))
    return agg


def build_full_feature_table(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    interactions: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Join all feature tables and attach churn labels.

    Returns the final feature matrix ready for ML training.
    """
    tx_feats = build_transaction_features(transactions)
    sup_feats = build_support_features(interactions)

    df = customers[["customer_id", "tenure_months", "contract_type"]].copy()
    df = df.merge(tx_feats, on="customer_id", how="left")
    df = df.merge(sup_feats, on="customer_id", how="left")
    df = df.merge(labels[["customer_id", "churned"]], on="customer_id", how="left")

    # Encode categorical
    df = pd.get_dummies(df, columns=["contract_type"], drop_first=False)

    logger.info(
        "Feature table shape: %s | Churn rate: %.2f%%",
        df.shape,
        df["churned"].mean() * 100 if "churned" in df.columns else float("nan"),
    )
    return df
