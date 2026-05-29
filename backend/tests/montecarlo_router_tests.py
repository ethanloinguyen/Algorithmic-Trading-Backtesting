"""
Unit tests for app/routers/montecarlo.py

Run with:
    python -m pytest tests/montecarlo_router_tests.py -v

"""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.montecarlo import (
    _log_returns,
    _extract_params,
    _percentile_bands,
    _coverage,
    _build_result,
    _run_gbm,
    TRAIN_START, TRAIN_END, TEST_START, TEST_END, N_SIMS,
)

client = TestClient(app)


# Fixtures 

def _price_series(start_price=100.0, seed=42) -> pd.Series:
    """Spans 2015-01-01 to 2024-12-31 so train and test splits both have enough rows."""
    np.random.seed(seed)
    idx = pd.date_range("2015-01-01", "2024-12-31", freq="B")
    n = len(idx)
    prices = start_price * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n)))
    return pd.Series(prices, index=idx, name="Close")




def _mock_download(prices: pd.Series):
    """Patch _download to return a fixed price series (bypasses lru_cache too)."""
    return patch("app.routers.montecarlo._download", return_value=prices)


# _log_returns 

class TestLogReturns:
    def test_length_is_one_less_than_prices(self):
        p = pd.Series([100.0, 101.0, 99.0, 102.0])
        r = _log_returns(p)
        assert len(r) == len(p) - 1

    def test_flat_prices_give_zero_returns(self):
        p = pd.Series([100.0] * 10)
        r = _log_returns(p)
        assert np.allclose(r.values, 0.0)

    def test_doubling_price_gives_log2_return(self):
        p = pd.Series([100.0, 200.0])
        r = _log_returns(p)
        assert abs(r.iloc[0] - np.log(2)) < 1e-9

    def test_no_nulls_in_output(self):
        p = pd.Series([50.0, 55.0, 52.0, 60.0, 58.0])
        assert _log_returns(p).isnull().sum() == 0


# _extract_params 

class TestExtractParams:
    def _returns(self):
        np.random.seed(0)
        return pd.Series(np.random.normal(0.0004, 0.01, 252))

    def test_returns_dict_with_mu_and_sigma(self):
        params = _extract_params(self._returns())
        assert "mu" in params and "sigma" in params

    def test_mu_is_annualised(self):
        daily = pd.Series([0.001] * 252)
        params = _extract_params(daily)
        assert abs(params["mu"] - 0.001 * 252) < 1e-6

    def test_sigma_is_annualised(self):
        np.random.seed(1)
        daily = pd.Series(np.random.normal(0, 0.01, 252))
        params = _extract_params(daily)
        expected = daily.std() * np.sqrt(252)
        assert abs(params["sigma"] - expected) < 1e-9

    def test_sigma_positive(self):
        params = _extract_params(self._returns())
        assert params["sigma"] > 0

    def test_returns_floats(self):
        params = _extract_params(self._returns())
        assert isinstance(params["mu"],    float)
        assert isinstance(params["sigma"], float)


# _percentile_bands 

class TestPercentileBands:
    def _paths(self):
        np.random.seed(42)
        return np.random.lognormal(0, 0.1, (100, 50))

    def test_returns_all_five_bands(self):
        bands = _percentile_bands(self._paths())
        for key in ("p5", "p16", "p50", "p84", "p95"):
            assert key in bands

    def test_band_ordering_holds_at_every_step(self):
        bands = _percentile_bands(self._paths())
        assert np.all(bands["p5"]  <= bands["p16"])
        assert np.all(bands["p16"] <= bands["p50"])
        assert np.all(bands["p50"] <= bands["p84"])
        assert np.all(bands["p84"] <= bands["p95"])

    def test_band_length_equals_n_steps(self):
        paths = self._paths()
        bands = _percentile_bands(paths)
        assert len(bands["p50"]) == paths.shape[1]

    def test_single_sim_all_bands_equal(self):
        path = np.ones((1, 10)) * 5.0
        bands = _percentile_bands(path)
        assert np.allclose(bands["p5"], bands["p95"])


# _coverage 

class TestCoverage:
    def _bands(self, lo=90.0, hi=110.0, n=10):
        return {
            "p16": np.full(n, lo),
            "p84": np.full(n, hi),
        }

    def test_all_inside_returns_1(self):
        actual = np.full(10, 100.0)
        assert _coverage(actual, self._bands()) == pytest.approx(1.0)

    def test_all_outside_returns_0(self):
        actual = np.full(10, 200.0)
        assert _coverage(actual, self._bands()) == pytest.approx(0.0)

    def test_half_inside_returns_0_5(self):
        actual = np.array([100.0] * 5 + [200.0] * 5)
        assert _coverage(actual, self._bands()) == pytest.approx(0.5)

    def test_empty_actual_returns_0(self):
        assert _coverage(np.array([]), self._bands()) == pytest.approx(0.0)

    def test_actual_shorter_than_bands(self):
        actual = np.full(5, 100.0)
        result = _coverage(actual, self._bands(n=10))
        assert 0.0 <= result <= 1.0

    def test_boundary_values_are_inside(self):
        actual = np.array([90.0, 110.0])
        assert _coverage(actual, self._bands(n=2)) == pytest.approx(1.0)


# _build_result 

class TestBuildResult:
    def _bands(self, n=5):
        return {k: np.linspace(90, 110, n) for k in ("p5", "p16", "p50", "p84", "p95")}

    def _train(self):
        idx = pd.date_range("2015-01-01", periods=10, freq="B")
        return pd.Series(np.linspace(100, 110, 10), index=idx)

    def _test(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        return pd.Series(np.linspace(110, 120, 5), index=idx)

    def test_returns_expected_top_level_keys(self):
        result = _build_result("AAPL", self._train(), self._test(),
                               self._bands(), mu=0.1, sigma=0.2)
        for key in ("symbol", "train", "bands", "mu_annual",
                    "sigma_annual", "n_sims", "coverage_1s"):
            assert key in result

    def test_symbol_preserved(self):
        result = _build_result("MSFT", self._train(), self._test(),
                               self._bands(), mu=0.1, sigma=0.2)
        assert result["symbol"] == "MSFT"

    def test_train_list_length_matches_prices(self):
        result = _build_result("AAPL", self._train(), self._test(),
                               self._bands(), mu=0.1, sigma=0.2)
        assert len(result["train"]) == len(self._train())

    def test_train_days_are_negative(self):
        result = _build_result("AAPL", self._train(), self._test(),
                               self._bands(), mu=0.1, sigma=0.2)
        assert all(item["day"] < 0 for item in result["train"])

    def test_bands_list_length_matches_p50(self):
        bands = self._bands(n=5)
        result = _build_result("AAPL", self._train(), self._test(),
                               bands, mu=0.1, sigma=0.2)
        assert len(result["bands"]) == len(bands["p50"])

    def test_band_days_start_at_zero(self):
        result = _build_result("AAPL", self._train(), self._test(),
                               self._bands(), mu=0.1, sigma=0.2)
        assert result["bands"][0]["day"] == 0

    def test_mu_and_sigma_rounded(self):
        result = _build_result("AAPL", self._train(), self._test(),
                               self._bands(), mu=0.12345678, sigma=0.98765432)
        assert result["mu_annual"]    == round(0.12345678, 4)
        assert result["sigma_annual"] == round(0.98765432, 4)

    def test_coverage_between_0_and_1(self):
        result = _build_result("AAPL", self._train(), self._test(),
                               self._bands(), mu=0.1, sigma=0.2)
        assert 0.0 <= result["coverage_1s"] <= 1.0

    def test_actual_present_for_test_days(self):
        result = _build_result("AAPL", self._train(), self._test(),
                               self._bands(), mu=0.1, sigma=0.2)
        # All band rows that overlap with test prices should have a real actual
        n_test = len(self._test())
        for item in result["bands"][:n_test]:
            assert item["actual"] is not None


# GET /api/montecarlo/single 

class TestMontecarloSingle:
    def _make_prices(self):
        return _price_series(seed=42)

    def test_returns_200_for_valid_ticker(self):
        with _mock_download(self._make_prices()):
            response = client.get("/api/montecarlo/single?symbol=AAPL")
        assert response.status_code == 200

    def test_response_contains_expected_keys(self):
        with _mock_download(self._make_prices()):
            response = client.get("/api/montecarlo/single?symbol=AAPL")
        body = response.json()
        for key in ("symbol", "train", "bands", "mu_annual",
                    "sigma_annual", "n_sims", "coverage_1s"):
            assert key in body

    def test_symbol_uppercased_in_response(self):
        with _mock_download(self._make_prices()):
            response = client.get("/api/montecarlo/single?symbol=aapl")
        assert response.json()["symbol"] == "AAPL"

    def test_n_sims_matches_constant(self):
        with _mock_download(self._make_prices()):
            response = client.get("/api/montecarlo/single?symbol=AAPL")
        assert response.json()["n_sims"] == N_SIMS

    def test_missing_symbol_returns_422(self):
        response = client.get("/api/montecarlo/single")
        assert response.status_code == 422

    def test_insufficient_data_returns_400(self):
        short = pd.Series([100.0]*50, index=pd.date_range('2015-01-01', periods=50, freq='B'))
        with _mock_download(short):
            response = client.get("/api/montecarlo/single?symbol=FAKE")
        assert response.status_code == 400

    def test_download_failure_returns_400(self):
        with patch("app.routers.montecarlo._download",
                   side_effect=Exception("network error")):
            response = client.get("/api/montecarlo/single?symbol=AAPL")
        assert response.status_code == 400

    def test_train_days_are_negative_integers(self):
        with _mock_download(self._make_prices()):
            response = client.get("/api/montecarlo/single?symbol=AAPL")
        train = response.json()["train"]
        assert all(item["day"] < 0 for item in train)

    def test_bands_days_start_at_zero(self):
        with _mock_download(self._make_prices()):
            response = client.get("/api/montecarlo/single?symbol=AAPL")
        assert response.json()["bands"][0]["day"] == 0

    def test_coverage_between_0_and_1(self):
        with _mock_download(self._make_prices()):
            response = client.get("/api/montecarlo/single?symbol=AAPL")
        cov = response.json()["coverage_1s"]
        assert 0.0 <= cov <= 1.0

    def test_mu_and_sigma_are_floats(self):
        with _mock_download(self._make_prices()):
            response = client.get("/api/montecarlo/single?symbol=AAPL")
        body = response.json()
        assert isinstance(body["mu_annual"],    float)
        assert isinstance(body["sigma_annual"], float)

    def test_symbol_stripped_of_whitespace(self):
        with _mock_download(self._make_prices()):
            response = client.get("/api/montecarlo/single?symbol= AAPL ")
        assert response.json()["symbol"] == "AAPL"


# GET /api/montecarlo/pair 

class TestMontecarloPair:
    def _prices(self, seed=0):
        return _price_series(seed=seed)

    def _mock_both(self, l_prices=None, f_prices=None):
        l_prices = l_prices or self._prices(seed=0)
        f_prices = f_prices or self._prices(seed=1)
        # _download is called with leader first, then follower
        return patch("app.routers.montecarlo._download",
                     side_effect=[l_prices, f_prices])

    def test_returns_200_for_valid_pair(self):
        with self._mock_both():
            response = client.get("/api/montecarlo/pair?leader=WM&follower=WMB")
        assert response.status_code == 200

    def test_response_contains_expected_keys(self):
        with self._mock_both():
            response = client.get("/api/montecarlo/pair?leader=WM&follower=WMB")
        body = response.json()
        for key in ("leader", "follower", "lag", "beta", "pearson",
                    "without", "with_ll"):
            assert key in body

    def test_leader_and_follower_uppercased(self):
        with self._mock_both():
            response = client.get("/api/montecarlo/pair?leader=wm&follower=wmb")
        body = response.json()
        assert body["leader"]   == "WM"
        assert body["follower"] == "WMB"

    def test_same_ticker_returns_400(self):
        response = client.get("/api/montecarlo/pair?leader=AAPL&follower=AAPL")
        assert response.status_code == 400

    def test_missing_leader_returns_422(self):
        response = client.get("/api/montecarlo/pair?follower=WMB")
        assert response.status_code == 422

    def test_missing_follower_returns_422(self):
        response = client.get("/api/montecarlo/pair?leader=WM")
        assert response.status_code == 422

    def test_lag_is_positive_integer(self):
        with self._mock_both():
            response = client.get("/api/montecarlo/pair?leader=WM&follower=WMB")
        assert isinstance(response.json()["lag"], int)
        assert response.json()["lag"] >= 1

    def test_pearson_between_minus1_and_1(self):
        with self._mock_both():
            response = client.get("/api/montecarlo/pair?leader=WM&follower=WMB")
        rho = response.json()["pearson"]
        assert -1.0 <= rho <= 1.0

    def test_without_and_with_ll_are_dicts(self):
        with self._mock_both():
            response = client.get("/api/montecarlo/pair?leader=WM&follower=WMB")
        body = response.json()
        assert isinstance(body["without"], dict)
        assert isinstance(body["with_ll"], dict)

    def test_both_results_have_bands_key(self):
        with self._mock_both():
            response = client.get("/api/montecarlo/pair?leader=WM&follower=WMB")
        body = response.json()
        assert "bands" in body["without"]
        assert "bands" in body["with_ll"]

    def test_download_failure_returns_400(self):
        with patch("app.routers.montecarlo._download",
                   side_effect=Exception("network error")):
            response = client.get("/api/montecarlo/pair?leader=WM&follower=WMB")
        assert response.status_code == 400

    def test_insufficient_shared_history_returns_400(self):
        short = pd.Series([100.0]*50, index=pd.date_range('2015-01-01', periods=50, freq='B'))
        with patch("app.routers.montecarlo._download",
                   side_effect=[short, short]):
            response = client.get("/api/montecarlo/pair?leader=WM&follower=WMB")
        assert response.status_code == 400

    def test_follower_symbol_in_both_results(self):
        with self._mock_both():
            response = client.get("/api/montecarlo/pair?leader=WM&follower=WMB")
        body = response.json()
        assert body["without"]["symbol"] == "WMB"
        assert body["with_ll"]["symbol"] == "WMB"