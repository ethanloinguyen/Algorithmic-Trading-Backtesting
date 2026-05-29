"""
Unit tests for app/services/portfolio_service.py

Run with:
    python -m pytest tests/portfolio_service_tests.py -v

All deps are imported inside functions in portfolio_service.py, so every
patch target must be the SOURCE module, not portfolio_service itself.

  get_bq_client  → app.core.bigquery.get_bq_client
  get_settings   → app.core.config.get_settings
  get_quality_picks          → app.services.bigquery_services.get_quality_picks
  get_signal_recommendations → app.services.portfolio_engine.get_signal_recommendations
  get_independent_*          → app.services.portfolio_engine.get_independent_recommendations
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

import app.services.portfolio_service as svc


# Helpers 

def _make_network_df(tickers=("AAPL", "MSFT")) -> pd.DataFrame:
    rows = []
    tickers = list(tickers)
    for i in range(len(tickers) - 1):
        rows.append(dict(
            ticker_i=tickers[i], ticker_j=tickers[i + 1],
            signal_strength=70.0, best_lag=2, mean_dcor=0.35,
            oos_sharpe_net=0.8, predicted_sharpe=0.9,
            sector_i="Technology", sector_j="Technology",
            centrality_i=0.8, centrality_j=0.6,
            frequency=0.6, half_life=45.0, sharpness=0.4,
            variance_dcor=0.01, oos_dcor=0.3, rank=1,
            as_of_date="2024-01-01",
        ))
    return pd.DataFrame(rows)


def _make_universe_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": "AAPL", "sector": "Technology", "centrality": 0.8},
        {"ticker": "MSFT", "sector": "Technology", "centrality": 0.7},
        {"ticker": "XOM",  "sector": "Energy",     "centrality": 0.5},
        {"ticker": "JPM",  "sector": "Finance",    "centrality": 0.6},
    ])


def _make_settings():
    s = MagicMock()
    s.gcp_project_id = "proj"
    s.bq_dataset = "ds"
    return s


def _make_bq_client(date_val="2024-01-01", net_df=None):
    """BQ client that returns aod_df on first call, net_df on second."""
    client = MagicMock()
    aod_job = MagicMock()
    aod_job.to_dataframe.return_value = pd.DataFrame({"latest": [date_val]})
    net_job = MagicMock()
    net_job.to_dataframe.return_value = net_df if net_df is not None else _make_network_df()
    client.query.side_effect = [aod_job, net_job]
    return client


# _TABLE_NAMES 

class TestTableNames:
    def test_broad_market_maps_to_final_network(self):
        assert svc._TABLE_NAMES["broad_market"] == "final_network"

    def test_in_sector_maps_to_sector_final_network(self):
        assert svc._TABLE_NAMES["in_sector"] == "sector_final_network"

    def test_ttl_is_one_hour(self):
        assert svc._AS_OF_TTL_SECONDS == 3600


# _get_latest_as_of_date 

class TestGetLatestAsOfDate:
    def setup_method(self):
        svc._AS_OF_CACHE.clear()

    @patch("app.core.config.get_settings")
    @patch("app.core.bigquery.get_bq_client")
    def test_returns_date_string(self, mock_bq, mock_settings):
        mock_settings.return_value = _make_settings()
        mock_bq.return_value = _make_bq_client("2024-06-01")
        result = svc._get_latest_as_of_date("final_network")
        assert result == "2024-06-01"

    @patch("app.core.config.get_settings")
    @patch("app.core.bigquery.get_bq_client")
    def test_caches_result_avoids_second_bq_call(self, mock_bq, mock_settings):
        mock_settings.return_value = _make_settings()
        client = MagicMock()
        client.query.return_value.to_dataframe.return_value = pd.DataFrame({"latest": ["2024-01-01"]})
        mock_bq.return_value = client

        svc._get_latest_as_of_date("final_network")
        svc._get_latest_as_of_date("final_network")
        assert client.query.call_count == 1  # second call served from cache

    @patch("app.core.config.get_settings")
    @patch("app.core.bigquery.get_bq_client")
    def test_stale_cache_triggers_refresh(self, mock_bq, mock_settings):
        mock_settings.return_value = _make_settings()
        client = MagicMock()
        client.query.return_value.to_dataframe.return_value = pd.DataFrame({"latest": ["2024-01-01"]})
        mock_bq.return_value = client

        svc._AS_OF_CACHE["final_network"] = {
            "date": "2023-01-01",
            "fetched_at": time.monotonic() - svc._AS_OF_TTL_SECONDS - 10,
        }
        svc._get_latest_as_of_date("final_network")
        assert client.query.call_count == 1  # stale cache forced a refresh

    @patch("app.core.config.get_settings")
    @patch("app.core.bigquery.get_bq_client")
    def test_fresh_cache_skips_bq(self, mock_bq, mock_settings):
        mock_settings.return_value = _make_settings()
        client = MagicMock()
        mock_bq.return_value = client

        svc._AS_OF_CACHE["final_network"] = {
            "date": "2024-06-01",
            "fetched_at": time.monotonic(),  # just set — definitely fresh
        }
        result = svc._get_latest_as_of_date("final_network")
        assert result == "2024-06-01"
        client.query.assert_not_called()

    @patch("app.core.config.get_settings")
    @patch("app.core.bigquery.get_bq_client")
    def test_raises_on_null_date(self, mock_bq, mock_settings):
        mock_settings.return_value = _make_settings()
        client = MagicMock()
        client.query.return_value.to_dataframe.return_value = pd.DataFrame({"latest": [None]})
        mock_bq.return_value = client

        with pytest.raises(ValueError, match="no valid as_of_date"):
            svc._get_latest_as_of_date("final_network")

    @patch("app.core.config.get_settings")
    @patch("app.core.bigquery.get_bq_client")
    def test_separate_caches_per_table(self, mock_bq, mock_settings):
        mock_settings.return_value = _make_settings()
        client = MagicMock()
        client.query.return_value.to_dataframe.return_value = pd.DataFrame({"latest": ["2024-01-01"]})
        mock_bq.return_value = client

        svc._get_latest_as_of_date("final_network")
        svc._get_latest_as_of_date("sector_final_network")
        assert client.query.call_count == 2  # one per distinct table


# get_final_network

class TestGetFinalNetwork:
    def setup_method(self):
        svc._AS_OF_CACHE.clear()

    def test_empty_tickers_returns_empty_df(self):
        result = svc.get_final_network([])
        assert isinstance(result, pd.DataFrame) and result.empty

    def test_none_tickers_returns_empty_df(self):
        result = svc.get_final_network(None)
        assert result.empty

    @patch("app.core.config.get_settings")
    @patch("app.core.bigquery.get_bq_client")
    def test_uses_final_network_for_broad_market(self, mock_bq, mock_settings):
        mock_settings.return_value = _make_settings()
        mock_bq.return_value = _make_bq_client()
        svc.get_final_network(["AAPL"], analysis_mode="broad_market")
        queries = [c.args[0] for c in mock_bq.return_value.query.call_args_list]
        assert any("final_network" in q for q in queries)

    @patch("app.core.config.get_settings")
    @patch("app.core.bigquery.get_bq_client")
    def test_uses_sector_table_for_in_sector(self, mock_bq, mock_settings):
        mock_settings.return_value = _make_settings()
        mock_bq.return_value = _make_bq_client()
        svc.get_final_network(["AAPL"], analysis_mode="in_sector")
        queries = [c.args[0] for c in mock_bq.return_value.query.call_args_list]
        assert any("sector_final_network" in q for q in queries)

    @patch("app.core.config.get_settings")
    @patch("app.core.bigquery.get_bq_client")
    def test_returns_dataframe(self, mock_bq, mock_settings):
        mock_settings.return_value = _make_settings()
        mock_bq.return_value = _make_bq_client()
        result = svc.get_final_network(["AAPL"])
        assert isinstance(result, pd.DataFrame)

    @patch("app.core.config.get_settings")
    @patch("app.core.bigquery.get_bq_client")
    def test_numeric_columns_coerced_from_strings(self, mock_bq, mock_settings):
        mock_settings.return_value = _make_settings()
        net_df = _make_network_df()
        net_df["signal_strength"] = net_df["signal_strength"].astype(str)
        mock_bq.return_value = _make_bq_client(net_df=net_df)
        result = svc.get_final_network(["AAPL"])
        assert result["signal_strength"].dtype in (float, "float64")


# run_portfolio_analysis 

class TestRunPortfolioAnalysis:
    def setup_method(self):
        svc._AS_OF_CACHE.clear()

    def _base_patches(self, net_df=None, universe_df=None):
        """The three patches needed for every run_portfolio_analysis test."""
        return [
            patch("app.services.portfolio_service.get_final_network",
                  return_value=net_df if net_df is not None else _make_network_df()),
            patch("app.services.portfolio_service._get_latest_as_of_date",
                  return_value="2024-01-01"),
            patch("app.services.portfolio_service._get_universe_metadata",
                  return_value=universe_df if universe_df is not None else _make_universe_df()),
            patch("app.services.bigquery_services.get_quality_picks",
                  return_value=[]),
        ]

    def test_empty_tickers_returns_empty_structure(self):
        result = svc.run_portfolio_analysis([])
        assert result["tickers_analyzed"]            == []
        assert result["overlaps"]                    == []
        assert result["signal_recommendations"]      == []
        assert result["independent_recommendations"] == []
        assert result["holdings_sectors"]            == {}

    def test_whitespace_only_tickers_treated_as_empty(self):
        result = svc.run_portfolio_analysis(["  ", ""])
        assert result["tickers_analyzed"] == []

    def test_returns_all_expected_keys(self):
        with patch("app.services.portfolio_service.get_final_network",
                   return_value=_make_network_df()), \
             patch("app.services.portfolio_service._get_latest_as_of_date",
                   return_value="2024-01-01"), \
             patch("app.services.portfolio_service._get_universe_metadata",
                   return_value=_make_universe_df()), \
             patch("app.services.bigquery_services.get_quality_picks",
                   return_value=[]):
            result = svc.run_portfolio_analysis(["AAPL", "MSFT"])

        for key in ("tickers_analyzed", "unknown_tickers", "overlaps",
                    "signal_recommendations", "independent_recommendations",
                    "quality_picks", "holdings_sectors"):
            assert key in result, f"Missing key: {key}"

    def test_unknown_tickers_identified(self):
        with patch("app.services.portfolio_service.get_final_network",
                   return_value=_make_network_df(["AAPL", "MSFT"])), \
             patch("app.services.portfolio_service._get_latest_as_of_date",
                   return_value="2024-01-01"), \
             patch("app.services.portfolio_service._get_universe_metadata",
                   return_value=_make_universe_df()), \
             patch("app.services.bigquery_services.get_quality_picks",
                   return_value=[]):
            result = svc.run_portfolio_analysis(["AAPL", "FAKE"])
        assert "FAKE" in result["unknown_tickers"]

    def test_known_tickers_in_tickers_analyzed(self):
        with patch("app.services.portfolio_service.get_final_network",
                   return_value=_make_network_df(["AAPL", "MSFT"])), \
             patch("app.services.portfolio_service._get_latest_as_of_date",
                   return_value="2024-01-01"), \
             patch("app.services.portfolio_service._get_universe_metadata",
                   return_value=_make_universe_df()), \
             patch("app.services.bigquery_services.get_quality_picks",
                   return_value=[]):
            result = svc.run_portfolio_analysis(["AAPL", "MSFT"])
        assert "AAPL" in result["tickers_analyzed"]
        assert "MSFT" in result["tickers_analyzed"]

    def test_overlaps_are_dicts(self):
        with patch("app.services.portfolio_service.get_final_network",
                   return_value=_make_network_df(["AAPL", "MSFT"])), \
             patch("app.services.portfolio_service._get_latest_as_of_date",
                   return_value="2024-01-01"), \
             patch("app.services.portfolio_service._get_universe_metadata",
                   return_value=_make_universe_df()), \
             patch("app.services.bigquery_services.get_quality_picks",
                   return_value=[]):
            result = svc.run_portfolio_analysis(["AAPL", "MSFT"])
        assert all(isinstance(o, dict) for o in result["overlaps"])

    def test_signal_recommendations_are_dicts(self):
        with patch("app.services.portfolio_service.get_final_network",
                   return_value=_make_network_df(["AAPL", "MSFT"])), \
             patch("app.services.portfolio_service._get_latest_as_of_date",
                   return_value="2024-01-01"), \
             patch("app.services.portfolio_service._get_universe_metadata",
                   return_value=_make_universe_df()), \
             patch("app.services.bigquery_services.get_quality_picks",
                   return_value=[]):
            result = svc.run_portfolio_analysis(["AAPL", "MSFT"])
        assert all(isinstance(r, dict) for r in result["signal_recommendations"])

    def test_independent_recommendations_are_dicts(self):
        with patch("app.services.portfolio_service.get_final_network",
                   return_value=_make_network_df(["AAPL", "MSFT"])), \
             patch("app.services.portfolio_service._get_latest_as_of_date",
                   return_value="2024-01-01"), \
             patch("app.services.portfolio_service._get_universe_metadata",
                   return_value=_make_universe_df()), \
             patch("app.services.bigquery_services.get_quality_picks",
                   return_value=[]):
            result = svc.run_portfolio_analysis(["AAPL", "MSFT"])
        assert all(isinstance(r, dict) for r in result["independent_recommendations"])

    def test_quality_picks_exception_returns_empty_list(self):
        with patch("app.services.portfolio_service.get_final_network",
                   return_value=_make_network_df()), \
             patch("app.services.portfolio_service._get_latest_as_of_date",
                   return_value="2024-01-01"), \
             patch("app.services.portfolio_service._get_universe_metadata",
                   return_value=_make_universe_df()), \
             patch("app.services.bigquery_services.get_quality_picks",
                   side_effect=Exception("BQ timeout")):
            result = svc.run_portfolio_analysis(["AAPL", "MSFT"])
        assert result["quality_picks"] == []

    def test_tickers_normalized_before_processing(self):
        with patch("app.services.portfolio_service.get_final_network",
                   return_value=_make_network_df()) as mock_gfn, \
             patch("app.services.portfolio_service._get_latest_as_of_date",
                   return_value="2024-01-01"), \
             patch("app.services.portfolio_service._get_universe_metadata",
                   return_value=_make_universe_df()), \
             patch("app.services.bigquery_services.get_quality_picks",
                   return_value=[]):
            svc.run_portfolio_analysis(["aapl", " msft "])
        called_tickers = mock_gfn.call_args.args[0]
        assert "AAPL" in called_tickers
        assert "MSFT" in called_tickers

    def test_in_sector_uses_zero_min_signal(self):
        # get_signal_recommendations is imported inside run_portfolio_analysis
        # via `from app.services.portfolio_engine import ...` — patch at source
        with patch("app.services.portfolio_service.get_final_network",
                   return_value=_make_network_df()), \
             patch("app.services.portfolio_service._get_latest_as_of_date",
                   return_value="2024-01-01"), \
             patch("app.services.portfolio_service._get_universe_metadata",
                   return_value=_make_universe_df()), \
             patch("app.services.bigquery_services.get_quality_picks",
                   return_value=[]), \
             patch("app.services.portfolio_engine.get_signal_recommendations",
                   return_value=[]) as mock_sig:
            svc.run_portfolio_analysis(["AAPL"], analysis_mode="in_sector")
        mock_sig.assert_called_once()
        assert mock_sig.call_args.kwargs.get("min_signal_strength") == 0.0

    def test_broad_market_uses_configured_min_signal(self):
        with patch("app.services.portfolio_service.get_final_network",
                   return_value=_make_network_df()), \
             patch("app.services.portfolio_service._get_latest_as_of_date",
                   return_value="2024-01-01"), \
             patch("app.services.portfolio_service._get_universe_metadata",
                   return_value=_make_universe_df()), \
             patch("app.services.bigquery_services.get_quality_picks",
                   return_value=[]), \
             patch("app.services.portfolio_engine.get_signal_recommendations",
                   return_value=[]) as mock_sig:
            svc.run_portfolio_analysis(["AAPL"], analysis_mode="broad_market", min_signal=70.0)
        mock_sig.assert_called_once()
        assert mock_sig.call_args.kwargs.get("min_signal_strength") == 70.0

    def test_top_n_forwarded_to_recommendations(self):
        with patch("app.services.portfolio_service.get_final_network",
                   return_value=_make_network_df()), \
             patch("app.services.portfolio_service._get_latest_as_of_date",
                   return_value="2024-01-01"), \
             patch("app.services.portfolio_service._get_universe_metadata",
                   return_value=_make_universe_df()), \
             patch("app.services.bigquery_services.get_quality_picks",
                   return_value=[]), \
             patch("app.services.portfolio_engine.get_signal_recommendations",
                   return_value=[]) as mock_sig, \
             patch("app.services.portfolio_engine.get_independent_recommendations",
                   return_value=[]) as mock_ind:
            svc.run_portfolio_analysis(["AAPL"], top_n=5)
        assert mock_sig.call_args.kwargs.get("top_n") == 5
        assert mock_ind.call_args.kwargs.get("top_n") == 5