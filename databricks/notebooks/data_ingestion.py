# Databricks Notebook: Data Ingestion
# ============================================================
# Uploads local CSV files to DBFS and optionally writes them
# to Snowflake using the Spark-Snowflake connector.
# ============================================================

# COMMAND ----------
# %md
# ## 1. Upload CSVs to DBFS
# Run this once to get your local data into Databricks.

# Upload via Databricks CLI (run locally, not in notebook):
#   databricks fs cp data/raw/customers.csv dbfs:/mnt/data/raw/customers.csv
#   databricks fs cp data/raw/transactions.csv dbfs:/mnt/data/raw/transactions.csv
#   databricks fs cp data/raw/support_interactions.csv dbfs:/mnt/data/raw/support_interactions.csv
#   databricks fs cp data/raw/churn_labels.csv dbfs:/mnt/data/raw/churn_labels.csv

# COMMAND ----------
# %md
# ## 2. Validate uploaded files

import os

files = [
    "/mnt/data/raw/customers.csv",
    "/mnt/data/raw/transactions.csv",
    "/mnt/data/raw/support_interactions.csv",
    "/mnt/data/raw/churn_labels.csv",
]

for path in files:
    try:
        dbutils.fs.ls(path)
        print(f"✓  {path}")
    except Exception:
        print(f"✗  {path} — NOT FOUND")

# COMMAND ----------
# %md
# ## 3. Quick data profiling

from pyspark.sql import functions as F

customers = spark.read.csv("/mnt/data/raw/customers.csv", header=True, inferSchema=True)
transactions = spark.read.csv("/mnt/data/raw/transactions.csv", header=True, inferSchema=True)

print("Customers schema:")
customers.printSchema()
print(f"\nRow count: {customers.count():,}")

print("\nTransactions schema:")
transactions.printSchema()
print(f"\nRow count: {transactions.count():,}")

# COMMAND ----------
# %md
# ## 4. (Optional) Write to Snowflake
# Requires the spark-snowflake connector JAR installed on the cluster.
# Install via: cluster libraries > Maven > net.snowflake:spark-snowflake_2.12:2.15.0-spark_3.4

import os

SNOWFLAKE_OPTIONS = {
    "sfURL":       os.getenv("SNOWFLAKE_ACCOUNT"),
    "sfUser":      os.getenv("SNOWFLAKE_USER"),
    "sfPassword":  os.getenv("SNOWFLAKE_PASSWORD"),
    "sfDatabase":  "mlops2_db",
    "sfSchema":    "RAW",
    "sfWarehouse": "COMPUTE_WH",
}

def write_to_snowflake(df, table_name: str) -> None:
    (
        df.write
        .format("net.snowflake.spark.snowflake")
        .options(**SNOWFLAKE_OPTIONS)
        .option("dbtable", table_name)
        .mode("overwrite")
        .save()
    )
    print(f"Written to Snowflake table: {table_name}")


# Uncomment to execute:
# write_to_snowflake(customers, "CUSTOMERS")
# write_to_snowflake(transactions, "TRANSACTIONS")
