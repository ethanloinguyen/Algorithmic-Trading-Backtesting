"""
Unit tests for app/services/portfolio_engine.py

Run with:
    python -m pytest tests/portfolio_engine_tests.py -v

No mocking needed — pure pandas/numpy logic throughout.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from app.services.portfolio_engine import (
    _normalize_tickers,
    _compute_durability_score,
    _interpretation,
    _signal_reasoning,
    _independent_reasoning,
    get_ticker_metadata,
    analyze_portfolio_overlap,
    get_signal_recommendations,
    get_independent_recommendations,
    get_holdings_sectors,
    _HALF_LIFE_CAP_DAYS,
    OverlapResult,
    Recommendation,
    IndependentRecommendation,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_network_df(rows: list[dict] | None = None) -> pd.DataFrame:
    """
    Build a minimal network DataFrame with all columns portfolio_engine expects.
    Each row represents a directed lead-lag pair.
    """
    default_rows = [
        dict(ticker_i="AAPL", ticker_j="MSFT", signal_strength=72.0, best_lag=2,
             mean_dcor=0.35, oos_sharpe_net=0.8, predicted_sharpe=0.9,
             sector_i="Technology", sector_j="Technology",
             centrality_i=0.8, centrality_j=0.6,
             frequency=0.6, half_life=45.0, sharpness=0.4),
        dict(ticker_i="JPM",  ticker_j="BAC",  signal_strength=65.0, best_lag=1,
             mean_dcor=0.28, oos_sharpe_net=0.5, predicted_sharpe=0.6,
             sector_i="Finance", sector_j="Finance",
             centrality_i=0.5, centrality_j=0.4,
             frequency=0.4, half_life=30.0, sharpness=0.3),
        dict(ticker_i="NVDA", ticker_j="AAPL", signal_strength=80.0, best_lag=3,
             mean_dcor=0.45, oos_sharpe_net=1.1, predicted_sharpe=1.2,
             sector_i="Technology", sector_j="Technology",
             centrality_i=0.9, centrality_j=0.8,
             frequency=0.7, half_life=60.0, sharpness=0.5),
    ]
    return pd.DataFrame(rows if rows is not None else default_rows)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


# ── _normalize_tickers ────────────────────────────────────────────────────────

class TestNormalizeTickers:
    def test_uppercases_tickers(self):
        assert _normalize_tickers(["aapl", "msft"]) == ["AAPL", "MSFT"]

    def test_strips_whitespace(self):
        assert _normalize_tickers([" AAPL ", "MSFT "]) == ["AAPL", "MSFT"]

    def test_filters_empty_strings(self):
        assert _normalize_tickers(["AAPL", "", "  "]) == ["AAPL"]

    def test_empty_list(self):
        assert _normalize_tickers([]) == []

    def test_mixed_case_and_whitespace(self):
        result = _normalize_tickers([" goog ", "nvda", "TSLA"])
        assert result == ["GOOG", "NVDA", "TSLA"]


# ── _compute_durability_score ─────────────────────────────────────────────────

class TestComputeDurabilityScore:
    def test_zero_half_life_returns_zero(self):
        assert _compute_durability_score(0.0) == 0.0

    def test_negative_half_life_returns_zero(self):
        assert _compute_durability_score(-1.0) == 0.0

    def test_none_returns_zero(self):
        assert _compute_durability_score(None) == 0.0

    def test_at_cap_returns_100(self):
        assert _compute_durability_score(_HALF_LIFE_CAP_DAYS) == 100.0

    def test_above_cap_clamped_to_100(self):
        assert _compute_durability_score(_HALF_LIFE_CAP_DAYS * 2) == 100.0

    def test_half_of_cap_returns_50(self):
        result = _compute_durability_score(_HALF_LIFE_CAP_DAYS / 2)
        assert abs(result - 50.0) < 0.01

    def test_score_between_0_and_100(self):
        for hl in [1, 10, 50, 100, 200, 300]:
            score = _compute_durability_score(float(hl))
            assert 0.0 <= score <= 100.0


# ── _interpretation ───────────────────────────────────────────────────────────

class TestInterpretation:
    def _make_row(self, **kwargs):
        defaults = dict(
            ticker_i="AAPL", ticker_j="MSFT", best_lag=2,
            signal_strength=72.5, mean_dcor=0.35,
            half_life=45.0, frequency=0.6,
        )
        return pd.Series({**defaults, **kwargs})

    def test_contains_both_tickers(self):
        result = _interpretation(self._make_row())
        assert "AAPL" in result
        assert "MSFT" in result

    def test_contains_lag(self):
        result = _interpretation(self._make_row(best_lag=3))
        assert "3" in result

    def test_singular_day_for_lag_1(self):
        result = _interpretation(self._make_row(best_lag=1))
        assert "day " in result     # "day " not "days "

    def test_plural_days_for_lag_gt_1(self):
        result = _interpretation(self._make_row(best_lag=2))
        assert "days" in result

    def test_contains_signal_strength(self):
        result = _interpretation(self._make_row(signal_strength=72.5))
        assert "72.5" in result

    def test_contains_frequency_percent(self):
        result = _interpretation(self._make_row(frequency=0.6))
        assert "60%" in result

    def test_contains_half_life(self):
        result = _interpretation(self._make_row(half_life=45.0))
        assert "45" in result


# ── get_ticker_metadata ───────────────────────────────────────────────────────

class TestGetTickerMetadata:
    def test_empty_df_returns_empty_metadata(self):
        meta = get_ticker_metadata(_empty_df())
        assert meta.empty
        assert "sector" in meta.columns

    def test_index_is_ticker(self):
        meta = get_ticker_metadata(_make_network_df())
        assert meta.index.name == "ticker"

    def test_all_tickers_present(self):
        meta = get_ticker_metadata(_make_network_df())
        for t in ["AAPL", "MSFT", "JPM", "BAC", "NVDA"]:
            assert t in meta.index

    def test_sector_preserved(self):
        meta = get_ticker_metadata(_make_network_df())
        assert meta.loc["AAPL", "sector"] == "Technology"
        assert meta.loc["JPM",  "sector"] == "Finance"

    def test_centrality_preserved(self):
        meta = get_ticker_metadata(_make_network_df())
        assert meta.loc["NVDA", "centrality"] == pytest.approx(0.9)

    def test_no_duplicate_tickers(self):
        meta = get_ticker_metadata(_make_network_df())
        assert meta.index.is_unique


# ── analyze_portfolio_overlap ─────────────────────────────────────────────────

class TestAnalyzePortfolioOverlap:
    def test_returns_empty_for_single_ticker(self):
        result = analyze_portfolio_overlap(["AAPL"], _make_network_df())
        assert result == []

    def test_returns_empty_for_empty_network(self):
        result = analyze_portfolio_overlap(["AAPL", "MSFT"], _empty_df())
        assert result == []

    def test_detects_overlap_between_holdings(self):
        result = analyze_portfolio_overlap(["AAPL", "MSFT"], _make_network_df())
        assert len(result) == 1
        assert result[0].ticker_leader   == "AAPL"
        assert result[0].ticker_follower == "MSFT"

    def test_results_sorted_by_signal_strength_desc(self):
        df = _make_network_df([
            dict(ticker_i="AAPL", ticker_j="MSFT", signal_strength=60.0, best_lag=1,
                 mean_dcor=0.3, oos_sharpe_net=0.5, predicted_sharpe=0.6,
                 sector_i="Tech", sector_j="Tech", centrality_i=0.5, centrality_j=0.4,
                 frequency=0.5, half_life=30.0, sharpness=0.3),
            dict(ticker_i="MSFT", ticker_j="AAPL", signal_strength=80.0, best_lag=1,
                 mean_dcor=0.4, oos_sharpe_net=0.8, predicted_sharpe=0.9,
                 sector_i="Tech", sector_j="Tech", centrality_i=0.6, centrality_j=0.5,
                 frequency=0.6, half_life=40.0, sharpness=0.4),
        ])
        result = analyze_portfolio_overlap(["AAPL", "MSFT"], df)
        assert result[0].signal_strength >= result[1].signal_strength

    def test_min_signal_strength_filter(self):
        result = analyze_portfolio_overlap(
            ["AAPL", "MSFT"], _make_network_df(), min_signal_strength=99.0
        )
        assert result == []

    def test_returns_overlap_result_instances(self):
        result = analyze_portfolio_overlap(["AAPL", "MSFT"], _make_network_df())
        assert all(isinstance(r, OverlapResult) for r in result)

    def test_signal_strength_rounded_to_1dp(self):
        df = _make_network_df([
            dict(ticker_i="AAPL", ticker_j="MSFT", signal_strength=72.555,
                 best_lag=1, mean_dcor=0.3, oos_sharpe_net=0.5, predicted_sharpe=0.6,
                 sector_i="Tech", sector_j="Tech", centrality_i=0.5, centrality_j=0.4,
                 frequency=0.5, half_life=30.0, sharpness=0.3),
        ])
        result = analyze_portfolio_overlap(["AAPL", "MSFT"], df)
        assert result[0].signal_strength == round(72.555, 1)

    def test_interpretation_is_non_empty_string(self):
        result = analyze_portfolio_overlap(["AAPL", "MSFT"], _make_network_df())
        assert isinstance(result[0].interpretation, str)
        assert len(result[0].interpretation) > 0

    def test_tickers_normalized(self):
        result = analyze_portfolio_overlap(["aapl", " msft "], _make_network_df())
        assert len(result) == 1

    def test_only_pairs_within_holdings_returned(self):
        # NVDA→AAPL is in the df but NVDA is not in user_tickers; should not appear
        result = analyze_portfolio_overlap(["AAPL", "MSFT"], _make_network_df())
        tickers_in_results = {(r.ticker_leader, r.ticker_follower) for r in result}
        assert ("NVDA", "AAPL") not in tickers_in_results


# ── get_signal_recommendations ────────────────────────────────────────────────

class TestGetSignalRecommendations:
    def _network(self):
        """Network where AAPL is held, NVDA has a connection to AAPL (candidate)."""
        return _make_network_df([
            dict(ticker_i="NVDA", ticker_j="AAPL", signal_strength=80.0, best_lag=2,
                 mean_dcor=0.45, oos_sharpe_net=1.1, predicted_sharpe=1.2,
                 sector_i="Technology", sector_j="Technology",
                 centrality_i=0.9, centrality_j=0.8,
                 frequency=0.7, half_life=60.0, sharpness=0.5),
            dict(ticker_i="JPM",  ticker_j="AAPL", signal_strength=65.0, best_lag=1,
                 mean_dcor=0.28, oos_sharpe_net=0.5, predicted_sharpe=0.6,
                 sector_i="Finance", sector_j="Technology",
                 centrality_i=0.5, centrality_j=0.8,
                 frequency=0.4, half_life=30.0, sharpness=0.3),
        ])

    def test_empty_tickers_returns_empty(self):
        assert get_signal_recommendations([], _make_network_df()) == []

    def test_empty_network_returns_empty(self):
        assert get_signal_recommendations(["AAPL"], _empty_df()) == []

    def test_returns_recommendation_instances(self):
        result = get_signal_recommendations(["AAPL"], self._network())
        assert all(isinstance(r, Recommendation) for r in result)

    def test_holdings_not_in_recommendations(self):
        result = get_signal_recommendations(["AAPL"], self._network())
        rec_tickers = {r.ticker for r in result}
        assert "AAPL" not in rec_tickers

    def test_min_signal_filter_respected(self):
        result = get_signal_recommendations(
            ["AAPL"], self._network(), min_signal_strength=99.0
        )
        assert result == []

    def test_top_n_limits_results(self):
        result = get_signal_recommendations(["AAPL"], self._network(), top_n=1)
        assert len(result) <= 1

    def test_sorted_by_composite_score_desc(self):
        result = get_signal_recommendations(["AAPL"], self._network())
        scores = [r.composite_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_composite_score_positive(self):
        result = get_signal_recommendations(["AAPL"], self._network())
        assert all(r.composite_score > 0 for r in result)

    def test_related_holdings_contains_user_ticker(self):
        result = get_signal_recommendations(["AAPL"], self._network())
        for r in result:
            assert "AAPL" in r.related_holdings

    def test_signal_score_within_range(self):
        result = get_signal_recommendations(["AAPL"], self._network())
        for r in result:
            assert 0 <= r.signal_score <= 100

    def test_centrality_score_within_range(self):
        result = get_signal_recommendations(["AAPL"], self._network())
        for r in result:
            assert 0 <= r.centrality_score <= 100

    def test_coverage_score_within_range(self):
        result = get_signal_recommendations(["AAPL"], self._network())
        for r in result:
            assert 0 <= r.coverage_score <= 100

    def test_sector_diversity_score_within_range(self):
        result = get_signal_recommendations(["AAPL"], self._network())
        for r in result:
            assert 0 <= r.sector_diversity_score <= 100

    def test_durability_score_within_range(self):
        result = get_signal_recommendations(["AAPL"], self._network())
        for r in result:
            assert 0 <= r.durability_score <= 100

    def test_reasoning_non_empty(self):
        result = get_signal_recommendations(["AAPL"], self._network())
        for r in result:
            assert isinstance(r.reasoning, str) and len(r.reasoning) > 0

    def test_zero_centrality_redistributes_weight(self):
        """When all centrality values are 0, no crash and results still returned."""
        df = _make_network_df([
            dict(ticker_i="NVDA", ticker_j="AAPL", signal_strength=70.0, best_lag=1,
                 mean_dcor=0.3, oos_sharpe_net=0.7, predicted_sharpe=0.8,
                 sector_i="Technology", sector_j="Technology",
                 centrality_i=0.0, centrality_j=0.0,
                 frequency=0.5, half_life=40.0, sharpness=0.4),
        ])
        result = get_signal_recommendations(["AAPL"], df)
        assert len(result) >= 1
        assert result[0].centrality_score == 0.0

    def test_both_direction_when_bidirectional(self):
        """If a candidate both leads and follows a holding, direction should be 'both'."""
        df = _make_network_df([
            dict(ticker_i="NVDA", ticker_j="AAPL", signal_strength=70.0, best_lag=1,
                 mean_dcor=0.3, oos_sharpe_net=0.7, predicted_sharpe=0.8,
                 sector_i="Tech", sector_j="Tech",
                 centrality_i=0.8, centrality_j=0.7,
                 frequency=0.5, half_life=40.0, sharpness=0.4),
            dict(ticker_i="AAPL", ticker_j="NVDA", signal_strength=65.0, best_lag=1,
                 mean_dcor=0.25, oos_sharpe_net=0.5, predicted_sharpe=0.6,
                 sector_i="Tech", sector_j="Tech",
                 centrality_i=0.7, centrality_j=0.8,
                 frequency=0.4, half_life=35.0, sharpness=0.35),
        ])
        result = get_signal_recommendations(["AAPL"], df)
        nvda = next((r for r in result if r.ticker == "NVDA"), None)
        assert nvda is not None
        assert nvda.direction == "both"

    def test_tickers_normalized(self):
        result_upper = get_signal_recommendations(["AAPL"], self._network())
        result_lower = get_signal_recommendations(["aapl"], self._network())
        assert len(result_upper) == len(result_lower)


# ── get_independent_recommendations ──────────────────────────────────────────

class TestGetIndependentRecommendations:
    def _network(self):
        """AAPL and MSFT are connected. XOM has no connection to AAPL."""
        return _make_network_df([
            dict(ticker_i="NVDA", ticker_j="AAPL", signal_strength=80.0, best_lag=2,
                 mean_dcor=0.45, oos_sharpe_net=1.1, predicted_sharpe=1.2,
                 sector_i="Technology", sector_j="Technology",
                 centrality_i=0.9, centrality_j=0.8,
                 frequency=0.7, half_life=60.0, sharpness=0.5),
        ])

    def _universe(self):
        return pd.DataFrame([
            {"ticker": "XOM",  "sector": "Energy",     "centrality": 0.5},
            {"ticker": "WMT",  "sector": "Consumer",   "centrality": 0.4},
            {"ticker": "NVDA", "sector": "Technology", "centrality": 0.9},
            {"ticker": "AAPL", "sector": "Technology", "centrality": 0.8},
        ])

    def test_empty_tickers_returns_empty(self):
        assert get_independent_recommendations([], self._network()) == []

    def test_returns_independent_recommendation_instances(self):
        result = get_independent_recommendations(
            ["AAPL"], self._network(), universe_meta_df=self._universe()
        )
        assert all(isinstance(r, IndependentRecommendation) for r in result)

    def test_user_holdings_excluded(self):
        result = get_independent_recommendations(
            ["AAPL"], self._network(), universe_meta_df=self._universe()
        )
        rec_tickers = {r.ticker for r in result}
        assert "AAPL" not in rec_tickers

    def test_connected_tickers_excluded(self):
        """NVDA is connected to AAPL (in network_df) — should not appear as independent."""
        result = get_independent_recommendations(
            ["AAPL"], self._network(), universe_meta_df=self._universe()
        )
        rec_tickers = {r.ticker for r in result}
        assert "NVDA" not in rec_tickers

    def test_truly_independent_tickers_included(self):
        """XOM and WMT have no connection to AAPL — should appear."""
        result = get_independent_recommendations(
            ["AAPL"], self._network(), universe_meta_df=self._universe()
        )
        rec_tickers = {r.ticker for r in result}
        assert "XOM" in rec_tickers or "WMT" in rec_tickers

    def test_sorted_by_composite_score_desc(self):
        result = get_independent_recommendations(
            ["AAPL"], self._network(), universe_meta_df=self._universe()
        )
        scores = [r.composite_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limits_results(self):
        result = get_independent_recommendations(
            ["AAPL"], self._network(), universe_meta_df=self._universe(), top_n=1
        )
        assert len(result) <= 1

    def test_composite_score_positive(self):
        result = get_independent_recommendations(
            ["AAPL"], self._network(), universe_meta_df=self._universe()
        )
        assert all(r.composite_score >= 0 for r in result)

    def test_sector_gap_score_between_0_and_100(self):
        result = get_independent_recommendations(
            ["AAPL"], self._network(), universe_meta_df=self._universe()
        )
        for r in result:
            assert 0 <= r.sector_gap_score <= 100

    def test_reasoning_non_empty(self):
        result = get_independent_recommendations(
            ["AAPL"], self._network(), universe_meta_df=self._universe()
        )
        for r in result:
            assert isinstance(r.reasoning, str) and len(r.reasoning) > 0

    def test_zero_centrality_universe_sets_sector_gap_weight_to_1(self):
        """When all centrality = 0, no crash and sector_gap_score drives everything."""
        universe = pd.DataFrame([
            {"ticker": "XOM", "sector": "Energy",   "centrality": 0.0},
            {"ticker": "WMT", "sector": "Consumer", "centrality": 0.0},
        ])
        result = get_independent_recommendations(
            ["AAPL"], self._network(), universe_meta_df=universe
        )
        for r in result:
            assert r.centrality_score == 0.0

    def test_falls_back_to_network_df_when_no_universe(self):
        """Without universe_meta_df, derive candidates from network_df itself."""
        result = get_independent_recommendations(["AAPL"], self._network())
        # May be empty since all non-AAPL tickers in the default network are connected
        assert isinstance(result, list)

    def test_tickers_normalized(self):
        result_upper = get_independent_recommendations(
            ["AAPL"], self._network(), universe_meta_df=self._universe()
        )
        result_lower = get_independent_recommendations(
            ["aapl"], self._network(), universe_meta_df=self._universe()
        )
        assert len(result_upper) == len(result_lower)


# ── get_holdings_sectors ──────────────────────────────────────────────────────

class TestGetHoldingsSectors:
    def test_returns_sector_for_known_ticker(self):
        result = get_holdings_sectors(["AAPL"], _make_network_df())
        assert result["AAPL"] == "Technology"

    def test_returns_empty_for_unknown_ticker(self):
        result = get_holdings_sectors(["FAKE"], _make_network_df())
        assert "FAKE" not in result

    def test_handles_empty_tickers(self):
        result = get_holdings_sectors([], _make_network_df())
        assert result == {}

    def test_handles_empty_network(self):
        result = get_holdings_sectors(["AAPL"], _empty_df())
        assert result == {}

    def test_multiple_tickers(self):
        result = get_holdings_sectors(["AAPL", "JPM"], _make_network_df())
        assert result["AAPL"] == "Technology"
        assert result["JPM"]  == "Finance"

    def test_tickers_normalized(self):
        result = get_holdings_sectors(["aapl"], _make_network_df())
        assert "AAPL" in result

    def test_returns_strings(self):
        result = get_holdings_sectors(["AAPL", "JPM"], _make_network_df())
        assert all(isinstance(v, str) for v in result.values())