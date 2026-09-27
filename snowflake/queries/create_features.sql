-- =============================================================================
-- Feature Materialization: Build CUSTOMER_FEATURES from RAW tables
-- =============================================================================
-- Run monthly (or after each Databricks pipeline run) to refresh
-- the ANALYTICS.CUSTOMER_FEATURES table.
-- =============================================================================

USE DATABASE mlops2_db;
USE SCHEMA ANALYTICS;

-- Set the feature window: last 3 calendar months
SET feature_month = DATE_TRUNC('MONTH', CURRENT_DATE());
SET lookback_start = DATEADD('MONTH', -3, $feature_month);

-- -----------------------------------------------------------------------------
-- Step 1: Transaction features per customer (3-month window)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TEMPORARY TABLE tmp_transaction_features AS
SELECT
    customer_id,
    SUM(amount)                                         AS total_spending_3m,
    AVG(amount)                                         AS avg_monthly_charge,
    COUNT(transaction_id)                               AS transaction_count_3m,
    ROUND(
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) /
        NULLIF(COUNT(transaction_id), 0),
        4
    )                                                   AS payment_failure_rate
FROM mlops2_db.RAW.TRANSACTIONS
WHERE transaction_date >= $lookback_start
  AND transaction_date <  $feature_month
GROUP BY customer_id;

-- -----------------------------------------------------------------------------
-- Step 2: Support interaction features per customer (3-month window)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TEMPORARY TABLE tmp_support_features AS
SELECT
    customer_id,
    COUNT(interaction_id)       AS support_tickets_3m,
    AVG(resolution_days)        AS avg_resolution_days,
    AVG(sentiment_score)        AS avg_sentiment
FROM mlops2_db.RAW.SUPPORT_INTERACTIONS
WHERE ticket_date >= $lookback_start
  AND ticket_date <  $feature_month
GROUP BY customer_id;

-- -----------------------------------------------------------------------------
-- Step 3: Join and insert into CUSTOMER_FEATURES
-- -----------------------------------------------------------------------------
INSERT INTO mlops2_db.ANALYTICS.CUSTOMER_FEATURES (
    customer_id,
    feature_month,
    total_spending_3m,
    avg_monthly_charge,
    payment_failure_rate,
    transaction_count_3m,
    support_tickets_3m,
    avg_resolution_days,
    avg_sentiment,
    tenure_months,
    contract_type,
    churned,
    pipeline_version
)
SELECT
    c.customer_id,
    $feature_month                          AS feature_month,
    COALESCE(t.total_spending_3m, 0)        AS total_spending_3m,
    COALESCE(t.avg_monthly_charge, 0)       AS avg_monthly_charge,
    COALESCE(t.payment_failure_rate, 0)     AS payment_failure_rate,
    COALESCE(t.transaction_count_3m, 0)     AS transaction_count_3m,
    COALESCE(s.support_tickets_3m, 0)       AS support_tickets_3m,
    COALESCE(s.avg_resolution_days, 0)      AS avg_resolution_days,
    COALESCE(s.avg_sentiment, 0)            AS avg_sentiment,
    c.tenure_months,
    c.contract_type,
    COALESCE(l.churned, FALSE)              AS churned,
    'v1.0'                                  AS pipeline_version
FROM mlops2_db.RAW.CUSTOMERS c
LEFT JOIN tmp_transaction_features t ON c.customer_id = t.customer_id
LEFT JOIN tmp_support_features     s ON c.customer_id = s.customer_id
LEFT JOIN mlops2_db.RAW.CHURN_LABELS l
    ON c.customer_id = l.customer_id
    AND l.label_month = $feature_month;

-- Result summary
SELECT
    COUNT(*)                                AS total_customers,
    AVG(total_spending_3m)                  AS avg_spending,
    ROUND(AVG(churned::INT) * 100, 2)       AS churn_rate_pct,
    $feature_month                          AS feature_month
FROM mlops2_db.ANALYTICS.CUSTOMER_FEATURES
WHERE feature_month = $feature_month;
