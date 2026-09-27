"""Unit tests for src/data/ingestion.py.

Snowflake is mocked throughout -- these are unit tests of our own query
construction and result handling, not integration tests against a live
Snowflake account.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.ingestion import (
    load_csv,
    load_customer_features_from_snowflake,
    load_from_snowflake,
    load_parquet,
)


class TestLoadCsv:
    def test_loads_csv_into_dataframe(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2\n3,4\n")

        df = load_csv(path)

        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_passes_through_kwargs(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a;b\n1;2\n")

        df = load_csv(path, sep=";")

        assert list(df.columns) == ["a", "b"]


class TestLoadParquet:
    def test_loads_parquet_into_dataframe(self, tmp_path):
        path = tmp_path / "data.parquet"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_parquet(path)

        df = load_parquet(path)

        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2


def _mock_snowflake_connection(fetch_pandas_result=None, fetch_pandas_side_effect=None):
    """Build a mock connection whose cursor supports the fetch_pandas_all path."""
    cursor = MagicMock()
    if fetch_pandas_side_effect is not None:
        cursor.fetch_pandas_all.side_effect = fetch_pandas_side_effect
    else:
        cursor.fetch_pandas_all.return_value = fetch_pandas_result

    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


class TestLoadFromSnowflake:
    def test_uses_explicit_credentials_over_env(self, monkeypatch):
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "env-account")
        conn, _ = _mock_snowflake_connection(fetch_pandas_result=pd.DataFrame({"X": [1]}))

        with patch("snowflake.connector.connect", return_value=conn) as mock_connect:
            load_from_snowflake("SELECT 1", account="explicit-account", user="u", password="p")

        assert mock_connect.call_args.kwargs["account"] == "explicit-account"

    def test_falls_back_to_env_credentials(self, monkeypatch):
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "env-account")
        monkeypatch.setenv("SNOWFLAKE_USER", "env-user")
        monkeypatch.setenv("SNOWFLAKE_PASSWORD", "env-password")
        conn, _ = _mock_snowflake_connection(fetch_pandas_result=pd.DataFrame({"X": [1]}))

        with patch("snowflake.connector.connect", return_value=conn) as mock_connect:
            load_from_snowflake("SELECT 1")

        assert mock_connect.call_args.kwargs["account"] == "env-account"
        assert mock_connect.call_args.kwargs["user"] == "env-user"
        assert mock_connect.call_args.kwargs["password"] == "env-password"

    def test_missing_credentials_raise(self, monkeypatch):
        monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
        with pytest.raises(KeyError):
            load_from_snowflake("SELECT 1", user="u", password="p")

    def test_uses_arrow_fetch_when_available(self, monkeypatch):
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "a")
        monkeypatch.setenv("SNOWFLAKE_USER", "u")
        monkeypatch.setenv("SNOWFLAKE_PASSWORD", "p")
        expected = pd.DataFrame({"CUSTOMER_ID": [1, 2]})
        conn, cursor = _mock_snowflake_connection(fetch_pandas_result=expected)

        with patch("snowflake.connector.connect", return_value=conn):
            result = load_from_snowflake("SELECT * FROM t")

        cursor.execute.assert_called_once_with("SELECT * FROM t")
        pd.testing.assert_frame_equal(result, expected)

    def test_falls_back_to_fetchall_when_arrow_unsupported(self, monkeypatch):
        from snowflake.connector.errors import NotSupportedError

        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "a")
        monkeypatch.setenv("SNOWFLAKE_USER", "u")
        monkeypatch.setenv("SNOWFLAKE_PASSWORD", "p")
        conn, cursor = _mock_snowflake_connection(fetch_pandas_side_effect=NotSupportedError())
        cursor.fetchall.return_value = [(1, "a"), (2, "b")]
        cursor.description = [("ID",), ("NAME",)]

        with patch("snowflake.connector.connect", return_value=conn):
            result = load_from_snowflake("SELECT * FROM t")

        assert list(result.columns) == ["ID", "NAME"]
        assert len(result) == 2

    def test_lowercases_columns_when_requested(self, monkeypatch):
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "a")
        monkeypatch.setenv("SNOWFLAKE_USER", "u")
        monkeypatch.setenv("SNOWFLAKE_PASSWORD", "p")
        conn, _ = _mock_snowflake_connection(
            fetch_pandas_result=pd.DataFrame({"CUSTOMER_ID": [1], "TENURE_MONTHS": [5]})
        )

        with patch("snowflake.connector.connect", return_value=conn):
            result = load_from_snowflake("SELECT 1", lowercase_columns=True)

        assert list(result.columns) == ["customer_id", "tenure_months"]

    def test_closes_cursor_and_connection(self, monkeypatch):
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "a")
        monkeypatch.setenv("SNOWFLAKE_USER", "u")
        monkeypatch.setenv("SNOWFLAKE_PASSWORD", "p")
        conn, cursor = _mock_snowflake_connection(fetch_pandas_result=pd.DataFrame({"X": [1]}))

        with patch("snowflake.connector.connect", return_value=conn):
            load_from_snowflake("SELECT 1")

        cursor.close.assert_called_once()
        conn.close.assert_called_once()

    def test_closes_connection_even_if_query_fails(self, monkeypatch):
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "a")
        monkeypatch.setenv("SNOWFLAKE_USER", "u")
        monkeypatch.setenv("SNOWFLAKE_PASSWORD", "p")
        conn, cursor = _mock_snowflake_connection()
        cursor.execute.side_effect = RuntimeError("query failed")

        with patch("snowflake.connector.connect", return_value=conn):
            with pytest.raises(RuntimeError):
                load_from_snowflake("SELECT 1")

        cursor.close.assert_called_once()
        conn.close.assert_called_once()


class TestLoadCustomerFeaturesFromSnowflake:
    def test_queries_latest_feature_month_by_default(self, monkeypatch):
        monkeypatch.delenv("SNOWFLAKE_DATABASE", raising=False)
        monkeypatch.delenv("SNOWFLAKE_SCHEMA", raising=False)
        fake_df = pd.DataFrame({"customer_id": [1]})

        with patch("src.data.ingestion.load_from_snowflake", return_value=fake_df) as mock_load:
            df, provenance = load_customer_features_from_snowflake()

        query = mock_load.call_args.args[0]
        assert "MAX(feature_month)" in query
        assert "MLOPS2_DB.ANALYTICS.CUSTOMER_FEATURES" in query
        assert df is fake_df

    def test_queries_specific_feature_month_when_given(self):
        fake_df = pd.DataFrame({"customer_id": [1]})

        with patch("src.data.ingestion.load_from_snowflake", return_value=fake_df) as mock_load:
            load_customer_features_from_snowflake(feature_month="2026-09-01")

        query = mock_load.call_args.args[0]
        assert "feature_month = '2026-09-01'" in query
        assert "MAX(feature_month)" not in query

    def test_provenance_reflects_actual_source(self, monkeypatch):
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "my-account")
        monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "MY_WH")
        fake_df = pd.DataFrame({"customer_id": [1]})

        with patch("src.data.ingestion.load_from_snowflake", return_value=fake_df):
            _, provenance = load_customer_features_from_snowflake(
                database="custom_db", schema="custom_schema"
            )

        assert provenance == {
            "data_source": "Snowflake",
            "account": "my-account",
            "database": "CUSTOM_DB",
            "schema": "CUSTOM_SCHEMA",
            "table": "CUSTOMER_FEATURES",
            "warehouse": "MY_WH",
        }

    def test_uses_lowercase_columns(self):
        fake_df = pd.DataFrame({"customer_id": [1]})

        with patch("src.data.ingestion.load_from_snowflake", return_value=fake_df) as mock_load:
            load_customer_features_from_snowflake()

        assert mock_load.call_args.kwargs["lowercase_columns"] is True
