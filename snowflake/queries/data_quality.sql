-- =============================================================================
-- Data Quality Checks
-- =============================================================================
-- Run these checks after each pipeline run to detect data anomalies
-- before feeding data into model training.
-- =============================================================================

USE DATABASE mlops2_db;

-- -----------------------------------------------------------------------------
-- Check 1: Null rates in CUSTOMER_FEATURES
-- -----------------------------------------------------------------------------
SELECT
    'CUSTOMER_FEATURES' AS table_name,
    COUNT(*)            AS total_rows,
    SUM(CASE WHEN total_spending_3m   IS NULL THEN 1 ELSE 0 END) AS null_spending,
    SUM(CASE WHEN avg_monthly_charge  IS NULL THEN 1 ELSE 0 END) AS null_avg_charge,
    SUM(CASE WHEN tenure_months       IS NULL THEN 1 ELSE 0 END) AS null_tenure,
    SUM(CASE WHEN churned             IS NULL THEN 1 ELSE 0 END) AS null_label
FROM mlops2_db.ANALYTICS.CUSTOMER_FEATURES;

-- -----------------------------------------------------------------------------
-- Check 2: Class balance in churn labels
-- -----------------------------------------------------------------------------
SELECT
    churned,
    COUNT(*)                             AS count,
    ROUND(COUNT(*) * 100.0 /
        SUM(COUNT(*)) OVER (), 2)        AS pct
FROM mlops2_db.ANALYTICS.CUSTOMER_FEATURES
WHERE feature_month = (
    SELECT MAX(feature_month) FROM mlops2_db.ANALYTICS.CUSTOMER_FEATURES
)
GROUP BY churned;

-- -----------------------------------------------------------------------------
-- Check 3: Feature distribution statistics (spot drift early)
-- -----------------------------------------------------------------------------
SELECT
    ROUND(AVG(total_spending_3m), 2)     AS avg_spending,
    ROUND(STDDEV(total_spending_3m), 2)  AS std_spending,
    MIN(total_spending_3m)               AS min_spending,
    MAX(total_spending_3m)               AS max_spending,
    ROUND(AVG(avg_monthly_charge), 2)    AS avg_charge,
    ROUND(AVG(payment_failure_rate), 4)  AS avg_fail_rate,
    ROUND(AVG(avg_sentiment), 4)         AS avg_sentiment
FROM mlops2_db.ANALYTICS.CUSTOMER_FEATURES
WHERE feature_month = (
    SELECT MAX(feature_month) FROM mlops2_db.ANALYTICS.CUSTOMER_FEATURES
);

-- -----------------------------------------------------------------------------
-- Check 4: Orphaned transactions (no matching customer)
-- -----------------------------------------------------------------------------
SELECT COUNT(*) AS orphaned_transactions
FROM mlops2_db.RAW.TRANSACTIONS t
LEFT JOIN mlops2_db.RAW.CUSTOMERS c ON t.customer_id = c.customer_id
WHERE c.customer_id IS NULL;

-- -----------------------------------------------------------------------------
-- Check 5: Recent prediction log — volume and distribution
-- -----------------------------------------------------------------------------
SELECT
    DATE_TRUNC('DAY', request_timestamp)  AS prediction_date,
    COUNT(*)                               AS predictions,
    ROUND(AVG(churn_probability), 4)       AS avg_churn_prob,
    SUM(prediction)                        AS churn_flags
FROM mlops2_db.ML.PREDICTION_LOG
WHERE request_timestamp >= DATEADD('DAY', -7, CURRENT_TIMESTAMP())
GROUP BY prediction_date
ORDER BY prediction_date DESC;
