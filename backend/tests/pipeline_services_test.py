"""
Unit tests for app/services/pipeline_service.py

Run with:
    python -m pytest tests/pipeline_services_test.py -v

pipeline_service.py runs importlib file-loading and sys.path manipulation
at IMPORT TIME, which crashes in tests because the repo paths don't exist
relative to the test runner.

Fix: inject stub modules into sys.modules and mock importlib.util BEFORE
the module is imported. Then replace run_clustering / run_portfolio_risk
on the live module object for each test via the fresh_mocks fixture.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch, call
import pandas as pd
import pytest


# ── One-time setup: block the file-system imports before the module loads ─────

def _inject_stubs():
    """Put stub modules in sys.modules so Python never touches the filesystem."""
    if "hierarchical" not in sys.modules:
        m = types.ModuleType("hierarchical")
        m.run_clustering = MagicMock()
        sys.modules["hierarchical"] = m

    if "mc_engine" not in sys.modules:
        m = types.ModuleType("mc_engine")
        m.run_portfolio_risk = MagicMock()
        sys.modules["mc_engine"] = m


_inject_stubs()

# Patch importlib so spec_from_file_location / module_from_spec never hit disk
_fake_spec = MagicMock()
_fake_hier_mod = sys.modules["hierarchical"]
_fake_spec.loader.exec_module = lambda mod: setattr(mod, "run_clustering", MagicMock())

sys.modules.pop("app.services.pipeline_service", None)

with patch("importlib.util.spec_from_file_location", return_value=_fake_spec), \
     patch("importlib.util.module_from_spec", return_value=_fake_hier_mod):
    import app.services.pipeline_service as _ps

# Make sure the module-level names point at our stubs
_ps.run_clustering    = sys.modules["hierarchical"].run_clustering
_ps.run_portfolio_risk = sys.modules["mc_engine"].run_portfolio_risk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clustering_df(tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ["NVDA", "JPM", "XOM"]
    n = len(tickers)
    all_sectors = ["Tech", "Finance", "Energy", "Consumer", "Health", "Energy2", "Utilities"]
    return pd.DataFrame({
        "sector":                all_sectors[:n],
        "stock":                 tickers,
        "cluster":               list(range(n)),
        "is_medoid":             [True] * n,
        "avg_dcor_to_portfolio": [0.1] * n,
        "mean_intra_dist":       [0.2] * n,
        "n_sector_candidates":   [5] * n,
        "cluster_size":          [3] * n,
    })


def _risk_result(tickers: list[str], portfolio_var: float = -0.04) -> dict:
    return {
        "tickers":       tickers,
        "missing":       [],
        "weights":       {t: 1 / len(tickers) for t in tickers},
        "horizon_days":  63,
        "n_simulations": 1000,
        "per_stock": {
            t: {"var_95": -0.05, "cvar_95": -0.07, "var_99": -0.09, "cvar_99": -0.12}
            for t in tickers
        },
        "portfolio": {"var_95": portfolio_var, "cvar_95": -0.06,
                      "diversification_benefit_95": 0.01},
    }


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_mocks():
    """Replace the two callables on the imported module with fresh MagicMocks."""
    mock_clust = MagicMock()
    mock_mc    = MagicMock()
    _ps.run_clustering     = mock_clust
    _ps.run_portfolio_risk = mock_mc
    yield mock_clust, mock_mc
    # Reset so stale call counts don't bleed between tests
    _ps.run_clustering     = MagicMock()
    _ps.run_portfolio_risk = MagicMock()


# ── Happy path ────────────────────────────────────────────────────────────────

class TestRunRiskPipelineSuccess:

    def test_returns_three_top_level_keys(self, fresh_mocks):
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df()
        mock_mc.side_effect = [_risk_result(["NVDA", "JPM", "XOM"]),
                                _risk_result(["AAPL"])]
        result = _ps.run_risk_pipeline(["AAPL"])
        assert set(result.keys()) == {"user_portfolio", "recommendations", "risk"}

    def test_user_portfolio_passed_through_unchanged(self, fresh_mocks):
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df()
        mock_mc.side_effect = [_risk_result(["NVDA", "JPM", "XOM"]),
                                _risk_result(["AAPL", "MSFT"])]
        result = _ps.run_risk_pipeline(["AAPL", "MSFT"])
        assert result["user_portfolio"] == ["AAPL", "MSFT"]

    def test_recommendations_serialized_as_list_of_dicts(self, fresh_mocks):
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df(["NVDA", "JPM"])
        mock_mc.side_effect = [_risk_result(["NVDA", "JPM"]),
                                _risk_result(["AAPL"])]
        result = _ps.run_risk_pipeline(["AAPL"])
        assert isinstance(result["recommendations"], list)
        assert result["recommendations"][0]["stock"] == "NVDA"

    def test_portfolio_risk_comes_from_user_run_not_rec_run(self, fresh_mocks):
        """portfolio key must be sourced from the second run_portfolio_risk call."""
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df(["NVDA"])
        rec_risk  = _risk_result(["NVDA"],          portfolio_var=-0.99)
        user_risk = _risk_result(["AAPL", "MSFT"],  portfolio_var=-0.01)
        mock_mc.side_effect = [rec_risk, user_risk]
        result = _ps.run_risk_pipeline(["AAPL", "MSFT"])
        assert result["risk"]["portfolio"]["var_95"] == -0.01

    def test_per_stock_comes_from_rec_run(self, fresh_mocks):
        """per_stock must be sourced from the first run_portfolio_risk call."""
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df(["NVDA"])
        rec_risk  = _risk_result(["NVDA"])
        user_risk = _risk_result(["AAPL"])
        rec_risk["per_stock"]["NVDA"]["var_95"] = -0.42
        mock_mc.side_effect = [rec_risk, user_risk]
        result = _ps.run_risk_pipeline(["AAPL"])
        assert result["risk"]["per_stock"]["NVDA"]["var_95"] == -0.42

    def test_clustering_called_once_with_user_portfolio(self, fresh_mocks):
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df()
        mock_mc.side_effect = [_risk_result(["NVDA", "JPM", "XOM"]),
                                _risk_result(["AAPL"])]
        _ps.run_risk_pipeline(["AAPL"])
        mock_clust.assert_called_once()
        assert mock_clust.call_args.kwargs["user_portfolio"] == ["AAPL"]

    def test_run_portfolio_risk_called_exactly_twice(self, fresh_mocks):
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df()
        mock_mc.side_effect = [_risk_result(["NVDA", "JPM", "XOM"]),
                                _risk_result(["AAPL"])]
        _ps.run_risk_pipeline(["AAPL"])
        assert mock_mc.call_count == 2

    def test_first_mc_call_uses_rec_tickers(self, fresh_mocks):
        mock_clust, mock_mc = fresh_mocks
        rec_tickers = ["NVDA", "JPM", "XOM"]
        mock_clust.return_value = _clustering_df(rec_tickers)
        mock_mc.side_effect = [_risk_result(rec_tickers), _risk_result(["AAPL"])]
        _ps.run_risk_pipeline(["AAPL"])
        assert mock_mc.call_args_list[0].kwargs["tickers"] == rec_tickers

    def test_second_mc_call_uses_user_portfolio(self, fresh_mocks):
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df()
        mock_mc.side_effect = [_risk_result(["NVDA", "JPM", "XOM"]),
                                _risk_result(["AAPL", "MSFT"])]
        _ps.run_risk_pipeline(["AAPL", "MSFT"])
        second_tickers = mock_mc.call_args_list[1].kwargs["tickers"]
        assert set(second_tickers) == {"AAPL", "MSFT"}

    def test_bq_client_forwarded_to_clustering(self, fresh_mocks):
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df()
        mock_mc.side_effect = [_risk_result(["NVDA", "JPM", "XOM"]),
                                _risk_result(["AAPL"])]
        fake_client = MagicMock()
        _ps.run_risk_pipeline(["AAPL"], bq_client=fake_client)
        assert mock_clust.call_args.kwargs["bq_client"] is fake_client

    def test_none_bq_client_is_default(self, fresh_mocks):
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df()
        mock_mc.side_effect = [_risk_result(["NVDA", "JPM", "XOM"]),
                                _risk_result(["AAPL"])]
        _ps.run_risk_pipeline(["AAPL"])
        assert mock_clust.call_args.kwargs["bq_client"] is None

    def test_risk_result_contains_all_expected_keys(self, fresh_mocks):
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df(["NVDA"])
        mock_mc.side_effect = [_risk_result(["NVDA"]), _risk_result(["AAPL"])]
        result = _ps.run_risk_pipeline(["AAPL"])
        for key in ("tickers", "missing", "weights", "horizon_days",
                    "n_simulations", "per_stock", "portfolio"):
            assert key in result["risk"], f"Missing key in risk: {key}"


# ── Parameter forwarding ──────────────────────────────────────────────────────

class TestParameterForwarding:

    def _run(self, fresh_mocks, **kwargs):
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df()
        mock_mc.side_effect = [_risk_result(["NVDA", "JPM", "XOM"]),
                                _risk_result(["AAPL"])]
        _ps.run_risk_pipeline(["AAPL"], **kwargs)
        return mock_mc

    def test_horizon_days_forwarded_to_both_mc_calls(self, fresh_mocks):
        mock_mc = self._run(fresh_mocks, horizon_days=126)
        for c in mock_mc.call_args_list:
            assert c.kwargs["horizon_days"] == 126

    def test_n_sims_forwarded_to_both_mc_calls(self, fresh_mocks):
        mock_mc = self._run(fresh_mocks, n_sims=5000)
        for c in mock_mc.call_args_list:
            assert c.kwargs["n_sims"] == 5000

    def test_target_return_forwarded(self, fresh_mocks):
        mock_mc = self._run(fresh_mocks, target_return=0.20)
        for c in mock_mc.call_args_list:
            assert c.kwargs["target_return"] == 0.20

    def test_seed_forwarded(self, fresh_mocks):
        mock_mc = self._run(fresh_mocks, seed=99)
        for c in mock_mc.call_args_list:
            assert c.kwargs["seed"] == 99

    def test_confidence_levels_forwarded(self, fresh_mocks):
        mock_mc = self._run(fresh_mocks, confidence_levels=[0.90, 0.95])
        for c in mock_mc.call_args_list:
            assert c.kwargs["confidence_levels"] == [0.90, 0.95]

    def test_default_confidence_levels_are_95_and_99(self, fresh_mocks):
        mock_mc = self._run(fresh_mocks)
        for c in mock_mc.call_args_list:
            assert c.kwargs["confidence_levels"] == [0.95, 0.99]

    def test_default_horizon_days_is_63(self, fresh_mocks):
        mock_mc = self._run(fresh_mocks)
        for c in mock_mc.call_args_list:
            assert c.kwargs["horizon_days"] == 63

    def test_default_n_sims_is_1000(self, fresh_mocks):
        mock_mc = self._run(fresh_mocks)
        for c in mock_mc.call_args_list:
            assert c.kwargs["n_sims"] == 1000

    def test_default_seed_is_42(self, fresh_mocks):
        mock_mc = self._run(fresh_mocks)
        for c in mock_mc.call_args_list:
            assert c.kwargs["seed"] == 42

    def test_default_target_return_is_0_10(self, fresh_mocks):
        mock_mc = self._run(fresh_mocks)
        for c in mock_mc.call_args_list:
            assert c.kwargs["target_return"] == 0.10


# ── Edge cases & error handling ───────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_clustering_result_raises_value_error(self, fresh_mocks):
        mock_clust, _ = fresh_mocks
        mock_clust.return_value = pd.DataFrame({"stock": []})
        with pytest.raises(ValueError, match="no recommendations"):
            _ps.run_risk_pipeline(["AAPL"])

    def test_clustering_exception_propagates(self, fresh_mocks):
        mock_clust, _ = fresh_mocks
        mock_clust.side_effect = RuntimeError("BigQuery timeout")
        with pytest.raises(RuntimeError, match="BigQuery timeout"):
            _ps.run_risk_pipeline(["AAPL"])

    def test_mc_exception_propagates(self, fresh_mocks):
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df()
        mock_mc.side_effect = RuntimeError("simulation failed")
        with pytest.raises(RuntimeError, match="simulation failed"):
            _ps.run_risk_pipeline(["AAPL"])

    def test_single_rec_ticker(self, fresh_mocks):
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df(["NVDA"])
        mock_mc.side_effect = [_risk_result(["NVDA"]), _risk_result(["AAPL"])]
        result = _ps.run_risk_pipeline(["AAPL"])
        assert len(result["recommendations"]) == 1

    def test_large_portfolio_all_tickers_reach_clustering(self, fresh_mocks):
        mock_clust, mock_mc = fresh_mocks
        portfolio = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
        mock_clust.return_value = _clustering_df(["JPM", "XOM"])
        mock_mc.side_effect = [_risk_result(["JPM", "XOM"]),
                                _risk_result(portfolio)]
        _ps.run_risk_pipeline(portfolio)
        assert mock_clust.call_args.kwargs["user_portfolio"] == portfolio

    def test_confidence_levels_none_defaults_to_95_99(self, fresh_mocks):
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df()
        mock_mc.side_effect = [_risk_result(["NVDA", "JPM", "XOM"]),
                                _risk_result(["AAPL"])]
        _ps.run_risk_pipeline(["AAPL"], confidence_levels=None)
        for c in mock_mc.call_args_list:
            assert c.kwargs["confidence_levels"] == [0.95, 0.99]

    def test_recommendations_length_matches_clustering_output(self, fresh_mocks):
        mock_clust, mock_mc = fresh_mocks
        mock_clust.return_value = _clustering_df(["NVDA", "JPM", "XOM", "WMT"])
        mock_mc.side_effect = [_risk_result(["NVDA", "JPM", "XOM", "WMT"]),
                                _risk_result(["AAPL"])]
        result = _ps.run_risk_pipeline(["AAPL"])
        assert len(result["recommendations"]) == 4