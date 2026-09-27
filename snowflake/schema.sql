-- =============================================================================
-- Snowflake DDL: End-to-End MLOps — Customer Churn Prediction
-- =============================================================================
-- Run this script once to set up the full database, schemas, and tables.
-- Execute as SYSADMIN or ACCOUNTADMIN.
-- =============================================================================

-- Databases & Schemas
-- -----------------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS mlops2_db;

USE DATABASE mlops2_db;

CREATE SCHEMA IF NOT EXISTS RAW;
CREATE SCHEMA IF NOT EXISTS ANALYTICS;
CREATE SCHEMA IF NOT EXISTS ML;

-- =============================================================================
-- RAW LAYER  (raw ingested data, unchanged from source)
-- =============================================================================
USE SCHEMA mlops2_db.RAW;

CREATE OR REPLACE TABLE CUSTOMERS (
    customer_id        VARCHAR(36)   NOT NULL PRIMARY KEY,
    age                INTEGER,
    gender             VARCHAR(10),
    tenure_months      INTEGER,
    contract_type      VARCHAR(20),   -- Month-to-month / One year / Two year
    payment_method     VARCHAR(30),
    monthly_charges    FLOAT,
    total_charges      FLOAT,
    created_at         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE OR REPLACE TABLE TRANSACTIONS (
    transaction_id     VARCHAR(36)   NOT NULL PRIMARY KEY,
    customer_id        VARCHAR(36)   NOT NULL,
    transaction_date   DATE,
    amount             FLOAT,
    payment_status     VARCHAR(20),   -- paid / late / failed
    created_at         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE OR REPLACE TABLE SUPPORT_INTERACTIONS (
    interaction_id     VARCHAR(36)   NOT NULL PRIMARY KEY,
    customer_id        VARCHAR(36)   NOT NULL,
    ticket_date        DATE,
    category           VARCHAR(30),   -- billing / technical / general
    resolution_days    INTEGER,
    sentiment_score    FLOAT,         -- -1.0 to 1.0
    created_at         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE OR REPLACE TABLE CHURN_LABELS (
    customer_id        VARCHAR(36)   NOT NULL,
    label_month        DATE          NOT NULL,
    churned            BOOLEAN       NOT NULL,
    created_at         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (customer_id, label_month)
);

-- =============================================================================
-- ANALYTICS LAYER  (cleaned, aggregated, feature-engineered)
-- =============================================================================
USE SCHEMA mlops2_db.ANALYTICS;

CREATE OR REPLACE TABLE CUSTOMER_FEATURES (
    customer_id            VARCHAR(36)   NOT NULL,
    feature_month          DATE          NOT NULL,
    -- Transactional features (3-month rolling window)
    total_spending_3m      FLOAT,
    avg_monthly_charge     FLOAT,
    payment_failure_rate   FLOAT,
    transaction_count_3m   INTEGER,
    -- Engagement / support features
    support_tickets_3m     INTEGER,
    avg_resolution_days    FLOAT,
    avg_sentiment          FLOAT,
    -- Contract / demographic features
    tenure_months          INTEGER,
    contract_type          VARCHAR(20),
    -- Label
    churned                BOOLEAN,
    -- Metadata
    pipeline_version       VARCHAR(20),
    created_at             TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (customer_id, feature_month)
);

-- =============================================================================
-- ML LAYER  (training splits, prediction logs, model metadata)
-- =============================================================================
USE SCHEMA mlops2_db.ML;

CREATE OR REPLACE TABLE TRAINING_DATASETS (
    dataset_version    VARCHAR(20)   NOT NULL,
    customer_id        VARCHAR(36)   NOT NULL,
    split              VARCHAR(10)   NOT NULL,   -- train / test / validation
    feature_month      DATE,
    created_at         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (dataset_version, customer_id)
);

CREATE OR REPLACE TABLE PREDICTION_LOG (
    log_id                 VARCHAR(36)   DEFAULT UUID_STRING() PRIMARY KEY,
    customer_id            VARCHAR(36),
    request_timestamp      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    churn_probability      FLOAT,
    prediction             INTEGER,      -- 0 or 1
    model_version          VARCHAR(50),
    -- Input features (stored for drift monitoring)
    tenure_months          INTEGER,
    total_spending_3m      FLOAT,
    avg_monthly_charge     FLOAT,
    payment_failure_rate   FLOAT,
    transaction_count_3m   INTEGER,
    support_tickets_3m     INTEGER,
    avg_resolution_days    FLOAT,
    avg_sentiment          FLOAT
);

CREATE OR REPLACE TABLE MODEL_REGISTRY_LOG (
    log_id          VARCHAR(36)   DEFAULT UUID_STRING() PRIMARY KEY,
    model_name      VARCHAR(100),
    model_version   VARCHAR(50),
    stage           VARCHAR(20),  -- Staging / Production / Archived
    mlflow_run_id   VARCHAR(64),
    roc_auc         FLOAT,
    f1_score        FLOAT,
    registered_at   TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);
