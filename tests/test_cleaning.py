"""Unit tests for src/data/cleaning.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.cleaning import (
    cast_types,
    clip_outliers,
    drop_duplicates,
    full_clean_pipeline,
    impute_nulls,
)


class TestDropDuplicates:
    def test_removes_duplicate_rows(self):
        df = pd.DataFrame({"customer_id": [1, 1, 2], "value": [10, 10, 20]})
        result = drop_duplicates(df)
        assert len(result) == 2

    def test_respects_subset(self):
        df = pd.DataFrame({"customer_id": [1, 1, 2], "value": [10, 99, 20]})
        result = drop_duplicates(df, subset=["customer_id"])
        assert len(result) == 2
        assert result["customer_id"].tolist() == [1, 2]

    def test_no_duplicates_is_a_noop(self):
        df = pd.DataFrame({"customer_id": [1, 2, 3]})
        result = drop_duplicates(df)
        assert len(result) == 3


class TestImputeNulls:
    def test_numeric_column_filled_with_median(self):
        df = pd.DataFrame({"amount": [10.0, 20.0, np.nan, 30.0]})
        result = impute_nulls(df)
        assert result["amount"].isna().sum() == 0
        assert result["amount"].iloc[2] == 20.0  # median of 10, 20, 30

    def test_categorical_column_filled_with_mode(self):
        df = pd.DataFrame({"plan": ["basic", "basic", None, "premium"]})
        result = impute_nulls(df)
        assert result["plan"].isna().sum() == 0
        assert result["plan"].iloc[2] == "basic"

    def test_column_with_no_nulls_is_unchanged(self):
        df = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})
        result = impute_nulls(df)
        assert result["amount"].tolist() == [1.0, 2.0, 3.0]


class TestClipOutliers:
    def test_clips_values_beyond_n_std(self):
        # 9 normal values around 10, plus one extreme outlier.
        original = pd.Series([10.0] * 9 + [10_000.0])
        expected_upper = original.mean() + 2.0 * original.std()

        df = pd.DataFrame({"amount": original.copy()})
        result = clip_outliers(df, columns=["amount"], n_std=2.0)

        assert result["amount"].max() == expected_upper

    def test_values_within_range_are_unchanged(self):
        df = pd.DataFrame({"amount": [8.0, 9.0, 10.0, 11.0, 12.0]})
        result = clip_outliers(df, columns=["amount"], n_std=4.0)
        assert result["amount"].tolist() == [8.0, 9.0, 10.0, 11.0, 12.0]

    def test_missing_column_is_skipped_not_errored(self):
        df = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})
        result = clip_outliers(df, columns=["does_not_exist"], n_std=4.0)
        assert result["amount"].tolist() == [1.0, 2.0, 3.0]


class TestCastTypes:
    def test_bool_column_cast(self):
        df = pd.DataFrame({"churned": [1, 0, 1]})
        result = cast_types(df)
        assert result["churned"].dtype == bool
        assert result["churned"].tolist() == [True, False, True]

    def test_int_column_cast_to_nullable_int64(self):
        df = pd.DataFrame({"tenure_months": [1.6, 2.4, 3.5]})
        result = cast_types(df)
        assert str(result["tenure_months"].dtype) == "Int64"
        # .round() -> 2, 2, 4 (banker's rounding on 3.5 rounds to 4)
        assert result["tenure_months"].tolist() == [2, 2, 4]

    def test_float_column_cast(self):
        df = pd.DataFrame({"avg_sentiment": ["0.5", "-0.3", "1.0"]})
        result = cast_types(df)
        assert result["avg_sentiment"].dtype == float
        assert result["avg_sentiment"].tolist() == [0.5, -0.3, 1.0]

    def test_invalid_numeric_value_coerced_to_nan(self):
        df = pd.DataFrame({"avg_sentiment": ["0.5", "not_a_number", "1.0"]})
        result = cast_types(df)
        assert pd.isna(result["avg_sentiment"].iloc[1])

    def test_missing_columns_are_skipped_not_errored(self):
        df = pd.DataFrame({"customer_id": [1, 2, 3]})
        result = cast_types(df)
        assert result["customer_id"].tolist() == [1, 2, 3]


class TestFullCleanPipeline:
    def test_runs_end_to_end_without_error(self):
        df = pd.DataFrame(
            {
                "customer_id": [1, 1, 2, 3],
                "total_spending_3m": [100.0, 100.0, np.nan, 150.0],
                "avg_monthly_charge": [50.0, 50.0, 60.0, 70.0],
                "payment_failure_rate": [0.0, 0.0, 0.1, 0.2],
                "transaction_count_3m": [3, 3, 4, 5],
                "support_tickets_3m": [0, 0, 1, 2],
                "avg_resolution_days": [1.0, 1.0, 2.0, 3.0],
                "avg_sentiment": [0.1, 0.1, -0.2, 0.3],
                "tenure_months": [12, 12, 24, 36],
                "churned": [0, 0, 1, 0],
            }
        )
        result = full_clean_pipeline(df)

        # Duplicate customer_id=1 row removed.
        assert len(result) == 3
        # No remaining nulls anywhere.
        assert result.isna().sum().sum() == 0
        # Types were cast as expected.
        assert result["churned"].dtype == bool
        assert str(result["tenure_months"].dtype) == "Int64"
