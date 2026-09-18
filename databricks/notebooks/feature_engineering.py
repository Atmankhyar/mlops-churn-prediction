# Databricks Notebook: Feature Engineering Pipeline
# ============================================================
# Run this notebook on Databricks with a PySpark cluster.
# It reads raw data, engineers features at scale (1M+ rows),
# and writes the result as a Delta Lake table.
#
# Attach to a cluster with: Runtime 13.x ML, Python 3.10+
# ============================================================

# COMMAND ----------
# %md
# ## 1. Configuration

import logging
from datetime import date
from dateutil.relativedelta import relativedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths — update for your DBFS mount or Unity Catalog volume
RAW_CUSTOMERS_PATH    = "/mnt/data/raw/customers.csv"
RAW_TRANSACTIONS_PATH = "/mnt/data/raw/transactions.csv"
RAW_INTERACTIONS_PATH = "/mnt/data/raw/support_interactions.csv"
RAW_LABELS_PATH       = "/mnt/data/raw/churn_labels.csv"
DELTA_OUTPUT_PATH     = "/mnt/delta/customer_features"

LOOKBACK_MONTHS = 3

# COMMAND ----------
# %md
# ## 2. Load Raw Data

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

customers = (
    spark.read.csv(RAW_CUSTOMERS_PATH, header=True, inferSchema=True)
    .cache()
)
transactions = (
    spark.read.csv(RAW_TRANSACTIONS_PATH, header=True, inferSchema=True)
    .withColumn("transaction_date", F.to_date("transaction_date"))
    .cache()
)
interactions = (
    spark.read.csv(RAW_INTERACTIONS_PATH, header=True, inferSchema=True)
    .withColumn("ticket_date", F.to_date("ticket_date"))
    .cache()
)
labels = spark.read.csv(RAW_LABELS_PATH, header=True, inferSchema=True)

print(f"Customers:    {customers.count():>10,}")
print(f"Transactions: {transactions.count():>10,}")
print(f"Interactions: {interactions.count():>10,}")

# COMMAND ----------
# %md
# ## 3. Transaction Features (distributed aggregation)

max_tx_date = transactions.agg(F.max("transaction_date")).collect()[0][0]
cutoff_date = F.add_months(F.lit(max_tx_date), -LOOKBACK_MONTHS)

recent_tx = transactions.filter(F.col("transaction_date") >= cutoff_date)

tx_features = recent_tx.groupBy("customer_id").agg(
    F.sum("amount").alias("total_spending_3m"),
    F.avg("amount").alias("avg_monthly_charge"),
    F.count("transaction_id").alias("transaction_count_3m"),
    (
        F.sum((F.col("payment_status") == "failed").cast("int")) /
        F.count("transaction_id")
    ).cast(DoubleType()).alias("payment_failure_rate"),
)

print(f"Transaction features computed for {tx_features.count():,} customers")

# COMMAND ----------
# %md
# ## 4. Support Interaction Features

max_ticket_date = interactions.agg(F.max("ticket_date")).collect()[0][0]
cutoff_ticket   = F.add_months(F.lit(max_ticket_date), -LOOKBACK_MONTHS)

recent_si = interactions.filter(F.col("ticket_date") >= cutoff_ticket)

support_features = recent_si.groupBy("customer_id").agg(
    F.count("interaction_id").alias("support_tickets_3m"),
    F.avg("resolution_days").alias("avg_resolution_days"),
    F.avg("sentiment_score").alias("avg_sentiment"),
)

print(f"Support features computed for {support_features.count():,} customers")

# COMMAND ----------
# %md
# ## 5. Join and Build Final Feature Table

feature_table = (
    customers.select("customer_id", "tenure_months", "contract_type")
    .join(tx_features,      on="customer_id", how="left")
    .join(support_features, on="customer_id", how="left")
    .join(
        labels.select("customer_id", "churned"),
        on="customer_id",
        how="left",
    )
    .fillna({
        "total_spending_3m":    0.0,
        "avg_monthly_charge":   0.0,
        "payment_failure_rate": 0.0,
        "transaction_count_3m": 0,
        "support_tickets_3m":   0,
        "avg_resolution_days":  0.0,
        "avg_sentiment":        0.0,
    })
    .withColumn("feature_month", F.lit(str(max_tx_date)))
)

total_rows = feature_table.count()
churn_rate = (
    feature_table
    .agg(F.avg(F.col("churned").cast("int")))
    .collect()[0][0]
)

print(f"\n{'='*50}")
print(f"Feature table row count: {total_rows:,}")
print(f"Churn rate:              {churn_rate*100:.1f}%")
print(f"Output path:             {DELTA_OUTPUT_PATH}")
print(f"{'='*50}")

# COMMAND ----------
# %md
# ## 6. Write as Delta Lake

(
    feature_table
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(DELTA_OUTPUT_PATH)
)

print(f"Delta table written to {DELTA_OUTPUT_PATH}")

# Verify
delta_count = spark.read.format("delta").load(DELTA_OUTPUT_PATH).count()
print(f"Verified row count in Delta: {delta_count:,}")

# COMMAND ----------
# %md
# ## 7. (Optional) Register as Databricks Table

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS default.customer_features
    USING DELTA
    LOCATION '{DELTA_OUTPUT_PATH}'
""")

display(spark.sql("DESCRIBE TABLE default.customer_features"))
