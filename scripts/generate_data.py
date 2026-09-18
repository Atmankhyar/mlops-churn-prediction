"""Synthetic data generator.

Generates 100K customers, 1M+ transactions, 500K+ support interactions,
and churn labels. Outputs CSV files to data/raw/.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --customers 50000 --transactions 500000
"""
from __future__ import annotations

import argparse
import logging
import random
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
fake = Faker()
Faker.seed(RANDOM_SEED)

OUTPUT_DIR = Path("data/raw")


def generate_customers(n: int = 100_000) -> pd.DataFrame:
    logger.info("Generating %d customers ...", n)
    contract_types = ["Month-to-month", "One year", "Two year"]
    payment_methods = ["Credit card", "Bank transfer", "Electronic check", "Mailed check"]

    customer_ids = [str(uuid.uuid4()) for _ in range(n)]
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": np.random.randint(18, 80, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "tenure_months": np.random.randint(1, 72, n),
        "contract_type": np.random.choice(contract_types, n, p=[0.55, 0.25, 0.20]),
        "payment_method": np.random.choice(payment_methods, n),
        "monthly_charges": np.round(np.random.uniform(20, 120, n), 2),
    })
    df["total_charges"] = np.round(df["monthly_charges"] * df["tenure_months"], 2)
    logger.info("Customers generated: %d rows", len(df))
    return df


def generate_transactions(
    customer_ids: list[str],
    n: int = 1_000_000,
    start_date: date = date(2023, 1, 1),
    end_date: date = date(2024, 3, 31),
) -> pd.DataFrame:
    logger.info("Generating %d transactions ...", n)
    date_range = (end_date - start_date).days
    payment_statuses = ["paid", "late", "failed"]

    df = pd.DataFrame({
        "transaction_id": [str(uuid.uuid4()) for _ in range(n)],
        "customer_id": np.random.choice(customer_ids, n),
        "transaction_date": [
            start_date + timedelta(days=random.randint(0, date_range))
            for _ in range(n)
        ],
        "amount": np.round(np.random.uniform(20, 130, n), 2),
        "payment_status": np.random.choice(payment_statuses, n, p=[0.82, 0.12, 0.06]),
    })
    logger.info("Transactions generated: %d rows", len(df))
    return df


def generate_support_interactions(
    customer_ids: list[str],
    n: int = 500_000,
    start_date: date = date(2023, 1, 1),
    end_date: date = date(2024, 3, 31),
) -> pd.DataFrame:
    logger.info("Generating %d support interactions ...", n)
    date_range = (end_date - start_date).days
    categories = ["billing", "technical", "general"]

    df = pd.DataFrame({
        "interaction_id": [str(uuid.uuid4()) for _ in range(n)],
        "customer_id": np.random.choice(customer_ids, n),
        "ticket_date": [
            start_date + timedelta(days=random.randint(0, date_range))
            for _ in range(n)
        ],
        "category": np.random.choice(categories, n),
        "resolution_days": np.random.randint(1, 15, n),
        "sentiment_score": np.round(np.random.uniform(-1.0, 1.0, n), 3),
    })
    logger.info("Support interactions generated: %d rows", len(df))
    return df


def generate_churn_labels(customers: pd.DataFrame) -> pd.DataFrame:
    """Generate churn labels with realistic correlations.

    Higher churn probability for:
    - Month-to-month contracts
    - Short tenure
    - High monthly charges
    """
    logger.info("Generating churn labels ...")
    df = customers[["customer_id", "contract_type", "tenure_months", "monthly_charges"]].copy()

    contract_churn_prob = {
        "Month-to-month": 0.30,
        "One year": 0.10,
        "Two year": 0.03,
    }
    base_prob = df["contract_type"].map(contract_churn_prob)

    # Tenure modifier: longer tenure -> lower churn
    tenure_mod = np.clip(1 - df["tenure_months"] / 72, 0, 1) * 0.15

    # Charge modifier: higher charges -> higher churn
    charge_mod = (df["monthly_charges"] - 20) / (120 - 20) * 0.10

    churn_prob = np.clip(base_prob + tenure_mod + charge_mod, 0, 1)
    churned = np.random.binomial(1, churn_prob).astype(bool)

    labels = pd.DataFrame({
        "customer_id": df["customer_id"],
        "label_month": date(2024, 3, 1),
        "churned": churned,
    })
    churn_rate = churned.mean() * 100
    logger.info("Churn labels generated | Churn rate: %.1f%%", churn_rate)
    return labels


def main(n_customers: int, n_transactions: int, n_interactions: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    customers = generate_customers(n_customers)
    customers.to_csv(OUTPUT_DIR / "customers.csv", index=False)
    logger.info("Saved customers.csv")

    customer_ids = customers["customer_id"].tolist()

    transactions = generate_transactions(customer_ids, n_transactions)
    transactions.to_csv(OUTPUT_DIR / "transactions.csv", index=False)
    logger.info("Saved transactions.csv")

    interactions = generate_support_interactions(customer_ids, n_interactions)
    interactions.to_csv(OUTPUT_DIR / "support_interactions.csv", index=False)
    logger.info("Saved support_interactions.csv")

    labels = generate_churn_labels(customers)
    labels.to_csv(OUTPUT_DIR / "churn_labels.csv", index=False)
    logger.info("Saved churn_labels.csv")

    print("\n" + "=" * 50)
    print("Data generation complete!")
    print(f"  Customers:    {len(customers):>10,}")
    print(f"  Transactions: {len(transactions):>10,}")
    print(f"  Interactions: {len(interactions):>10,}")
    print(f"  Churn rate:   {labels['churned'].mean()*100:>9.1f}%")
    print(f"  Output dir:   {OUTPUT_DIR.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic MLOps dataset")
    parser.add_argument("--customers", type=int, default=100_000)
    parser.add_argument("--transactions", type=int, default=1_000_000)
    parser.add_argument("--interactions", type=int, default=500_000)
    args = parser.parse_args()
    main(args.customers, args.transactions, args.interactions)
