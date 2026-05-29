"""
Unit tests for app/routers/pairs.py

Run with:
    python -m pytest tests/pairs_router_tests.py -v

All BigQuery service calls are mocked at their source module.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.stock import StockDetail, PairDetail, NetworkResponse, NetworkNodeModel, NetworkEdgeModel

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_stock_detail(**kwargs) -> StockDetail:
    defaults = dict(
        symbol="AAPL", name="Apple Inc.", sector="Technology",
        industry="Consumer Electronics", market_cap=3e12,
        pe_ratio=28.5, high_52w=200.0, low_52w=130.0,
    )
    return StockDetail(**{**defaults, **kwargs})


def _make_pair_detail(**kwargs) -> PairDetail:
    defaults = dict(
        ticker_i="AAPL", ticker_j="MSFT", best_lag=2,
        mean_dcor=0.35, signal_strength=72.5, frequency=0.6,
        half_life=45.0, oos_sharpe_net=1.2, sector_i="Technology",
        sector_j="Technology", found=True,
    )
    return PairDetail(**{**defaults, **kwargs})


def _make_network_response(**kwargs) -> NetworkResponse:
    nodes = [
        NetworkNodeModel(id="AAPL", sector="Technology", centrality=0.8, out_degree=3),
        NetworkNodeModel(id="MSFT", sector="Technology", centrality=0.7, out_degree=2),
    ]
    edges = [
        NetworkEdgeModel(source="AAPL", target="MSFT",
                         signal_strength=72.5, best_lag=2, mean_dcor=0.35),
    ]
    defaults = dict(nodes=nodes, edges=edges,
                    analysis_mode="broad_market", min_signal=55.0)
    return NetworkResponse(**{**defaults, **kwargs})


# ── GET /api/stocks/{symbol}/detail ──────────────────────────────────────────

class TestStockDetail:
    URL = "/api/stocks/{}/detail"

    def test_returns_200_for_valid_symbol(self):
        with patch("app.routers.pairs.get_stock_detail",
                   return_value=_make_stock_detail()):
            response = client.get(self.URL.format("AAPL"))
        assert response.status_code == 200

    def test_symbol_uppercased(self):
        with patch("app.routers.pairs.get_stock_detail",
                   return_value=_make_stock_detail()) as mock:
            client.get(self.URL.format("aapl"))
        mock.assert_called_once_with("AAPL")

    def test_symbol_stripped_of_whitespace(self):
        with patch("app.routers.pairs.get_stock_detail",
                   return_value=_make_stock_detail()) as mock:
            client.get(self.URL.format(" AAPL "))
        mock.assert_called_once_with("AAPL")

    def test_response_contains_expected_fields(self):
        with patch("app.routers.pairs.get_stock_detail",
                   return_value=_make_stock_detail()):
            response = client.get(self.URL.format("AAPL"))
        body = response.json()
        for field in ("symbol", "name", "sector", "industry",
                      "pe_ratio", "high_52w", "low_52w"):
            assert field in body, f"Missing field: {field}"

    def test_response_symbol_matches_request(self):
        with patch("app.routers.pairs.get_stock_detail",
                   return_value=_make_stock_detail(symbol="NVDA")):
            response = client.get(self.URL.format("NVDA"))
        assert response.json()["symbol"] == "NVDA"

    def test_none_pe_ratio_allowed(self):
        with patch("app.routers.pairs.get_stock_detail",
                   return_value=_make_stock_detail(pe_ratio=None)):
            response = client.get(self.URL.format("AAPL"))
        assert response.status_code == 200
        assert response.json()["pe_ratio"] is None

    def test_none_52w_values_allowed(self):
        with patch("app.routers.pairs.get_stock_detail",
                   return_value=_make_stock_detail(high_52w=None, low_52w=None)):
            response = client.get(self.URL.format("AAPL"))
        assert response.status_code == 200

    def test_only_accepts_get(self):
        with patch("app.routers.pairs.get_stock_detail",
                   return_value=_make_stock_detail()):
            assert client.post(self.URL.format("AAPL")).status_code   == 405
            assert client.delete(self.URL.format("AAPL")).status_code == 405

    def test_content_type_is_json(self):
        with patch("app.routers.pairs.get_stock_detail",
                   return_value=_make_stock_detail()):
            response = client.get(self.URL.format("AAPL"))
        assert "application/json" in response.headers["content-type"]

    def test_fallback_detail_when_unknown_symbol(self):
        """get_stock_detail returns a minimal StockDetail for unknown symbols."""
        with patch("app.routers.pairs.get_stock_detail",
                   return_value=StockDetail(symbol="FAKE", name="FAKE")):
            response = client.get(self.URL.format("FAKE"))
        assert response.status_code == 200
        assert response.json()["symbol"] == "FAKE"


# ── GET /api/pairs/{ticker_i}/{ticker_j} ──────────────────────────────────────

class TestPairDetail:
    URL = "/api/pairs/{}/{}"

    def test_returns_200_for_valid_pair(self):
        with patch("app.routers.pairs.get_pair_data",
                   return_value=_make_pair_detail()):
            response = client.get(self.URL.format("AAPL", "MSFT"))
        assert response.status_code == 200

    def test_tickers_uppercased(self):
        with patch("app.routers.pairs.get_pair_data",
                   return_value=_make_pair_detail()) as mock:
            client.get(self.URL.format("aapl", "msft"))
        mock.assert_called_once_with("AAPL", "MSFT", "broad_market")

    def test_same_ticker_returns_400(self):
        response = client.get(self.URL.format("AAPL", "AAPL"))
        assert response.status_code == 400
        assert "different" in response.json()["detail"].lower()

    def test_response_contains_expected_fields(self):
        with patch("app.routers.pairs.get_pair_data",
                   return_value=_make_pair_detail()):
            response = client.get(self.URL.format("AAPL", "MSFT"))
        body = response.json()
        for field in ("ticker_i", "ticker_j", "best_lag", "mean_dcor",
                      "signal_strength", "found"):
            assert field in body, f"Missing field: {field}"

    def test_found_true_when_relationship_exists(self):
        with patch("app.routers.pairs.get_pair_data",
                   return_value=_make_pair_detail(found=True)):
            response = client.get(self.URL.format("AAPL", "MSFT"))
        assert response.json()["found"] is True

    def test_found_false_when_no_relationship(self):
        with patch("app.routers.pairs.get_pair_data",
                   return_value=_make_pair_detail(found=False, signal_strength=0.0)):
            response = client.get(self.URL.format("AAPL", "XOM"))
        assert response.status_code == 200
        assert response.json()["found"] is False

    def test_default_analysis_mode_is_broad_market(self):
        with patch("app.routers.pairs.get_pair_data",
                   return_value=_make_pair_detail()) as mock:
            client.get(self.URL.format("AAPL", "MSFT"))
        assert mock.call_args.args[2] == "broad_market"

    def test_in_sector_mode_forwarded(self):
        with patch("app.routers.pairs.get_pair_data",
                   return_value=_make_pair_detail()) as mock:
            client.get(self.URL.format("AAPL", "MSFT") + "?analysis_mode=in_sector")
        assert mock.call_args.args[2] == "in_sector"

    def test_invalid_analysis_mode_returns_422(self):
        response = client.get(self.URL.format("AAPL", "MSFT") + "?analysis_mode=invalid")
        assert response.status_code == 422

    def test_tickers_stripped_of_whitespace(self):
        with patch("app.routers.pairs.get_pair_data",
                   return_value=_make_pair_detail()) as mock:
            client.get(self.URL.format(" AAPL ", " MSFT "))
        mock.assert_called_once_with("AAPL", "MSFT", "broad_market")

    def test_only_accepts_get(self):
        with patch("app.routers.pairs.get_pair_data",
                   return_value=_make_pair_detail()):
            assert client.post(self.URL.format("AAPL", "MSFT")).status_code   == 405
            assert client.delete(self.URL.format("AAPL", "MSFT")).status_code == 405

    def test_signal_strength_in_response(self):
        with patch("app.routers.pairs.get_pair_data",
                   return_value=_make_pair_detail(signal_strength=72.5)):
            response = client.get(self.URL.format("AAPL", "MSFT"))
        assert response.json()["signal_strength"] == 72.5


# ── GET /api/network ──────────────────────────────────────────────────────────

class TestNetworkGraph:
    URL = "/api/network"

    def test_returns_200(self):
        with patch("app.routers.pairs.get_network_data",
                   return_value=_make_network_response()):
            response = client.get(self.URL)
        assert response.status_code == 200

    def test_response_contains_nodes_and_edges(self):
        with patch("app.routers.pairs.get_network_data",
                   return_value=_make_network_response()):
            response = client.get(self.URL)
        body = response.json()
        assert "nodes" in body
        assert "edges" in body

    def test_default_params_forwarded(self):
        with patch("app.routers.pairs.get_network_data",
                   return_value=_make_network_response()) as mock:
            client.get(self.URL)
        mock.assert_called_once_with("broad_market", 55.0, 50)

    def test_custom_analysis_mode_forwarded(self):
        with patch("app.routers.pairs.get_network_data",
                   return_value=_make_network_response()) as mock:
            client.get(self.URL + "?analysis_mode=in_sector")
        assert mock.call_args.args[0] == "in_sector"

    def test_custom_min_signal_forwarded(self):
        with patch("app.routers.pairs.get_network_data",
                   return_value=_make_network_response()) as mock:
            client.get(self.URL + "?min_signal=70.0")
        assert mock.call_args.args[1] == 70.0

    def test_custom_limit_forwarded(self):
        with patch("app.routers.pairs.get_network_data",
                   return_value=_make_network_response()) as mock:
            client.get(self.URL + "?limit=25")
        assert mock.call_args.args[2] == 25

    def test_min_signal_below_0_returns_422(self):
        with patch("app.routers.pairs.get_network_data",
                   return_value=_make_network_response()):
            response = client.get(self.URL + "?min_signal=-1")
        assert response.status_code == 422

    def test_min_signal_above_100_returns_422(self):
        with patch("app.routers.pairs.get_network_data",
                   return_value=_make_network_response()):
            response = client.get(self.URL + "?min_signal=101")
        assert response.status_code == 422

    def test_limit_below_10_returns_422(self):
        with patch("app.routers.pairs.get_network_data",
                   return_value=_make_network_response()):
            response = client.get(self.URL + "?limit=5")
        assert response.status_code == 422

    def test_limit_above_100_returns_422(self):
        with patch("app.routers.pairs.get_network_data",
                   return_value=_make_network_response()):
            response = client.get(self.URL + "?limit=101")
        assert response.status_code == 422

    def test_invalid_analysis_mode_returns_422(self):
        response = client.get(self.URL + "?analysis_mode=wrong")
        assert response.status_code == 422

    def test_nodes_have_expected_fields(self):
        with patch("app.routers.pairs.get_network_data",
                   return_value=_make_network_response()):
            response = client.get(self.URL)
        node = response.json()["nodes"][0]
        for field in ("id", "sector", "centrality", "out_degree"):
            assert field in node, f"Missing node field: {field}"

    def test_edges_have_expected_fields(self):
        with patch("app.routers.pairs.get_network_data",
                   return_value=_make_network_response()):
            response = client.get(self.URL)
        edge = response.json()["edges"][0]
        for field in ("source", "target", "signal_strength", "best_lag", "mean_dcor"):
            assert field in edge, f"Missing edge field: {field}"

    def test_empty_network_returns_200(self):
        empty = NetworkResponse(nodes=[], edges=[],
                                analysis_mode="broad_market", min_signal=55.0)
        with patch("app.routers.pairs.get_network_data", return_value=empty):
            response = client.get(self.URL)
        assert response.status_code == 200
        assert response.json()["nodes"] == []
        assert response.json()["edges"] == []

    def test_only_accepts_get(self):
        with patch("app.routers.pairs.get_network_data",
                   return_value=_make_network_response()):
            assert client.post(self.URL).status_code   == 405
            assert client.delete(self.URL).status_code == 405

    def test_content_type_is_json(self):
        with patch("app.routers.pairs.get_network_data",
                   return_value=_make_network_response()):
            response = client.get(self.URL)
        assert "application/json" in response.headers["content-type"]