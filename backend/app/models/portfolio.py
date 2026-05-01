# backend/app/models/portfolio.py
from typing import Literal
from pydantic import BaseModel


class PortfolioRequest(BaseModel):
    tickers: list[str]
    analysis_mode: Literal["broad_market", "in_sector"] = "broad_market"


class OverlapResult(BaseModel):
    ticker_leader:    str
    ticker_follower:  str
    best_lag:         int
    signal_strength:  float
    mean_dcor:        float
    oos_sharpe_net:   float
    predicted_sharpe: float
    sector_leader:    str
    sector_follower:  str
    frequency:        float
    half_life:        float
    sharpness:        float
    interpretation:   str


class Recommendation(BaseModel):
    """Group A — stock with a detected lead-lag connection to user's holdings."""
    ticker:                     str
    sector:                     str
    centrality:                 float
    composite_score:            float
    n_portfolio_relationships:  int
    best_relationship_strength: float
    mean_relationship_strength: float
    related_holdings:           list[str]
    direction:                  str
    signal_score:               float
    centrality_score:           float
    sector_diversity_score:     float
    coverage_score:             float
    durability_score:           float   # normalized half-life 0-100
    reasoning:                  str


class IndependentRecommendation(BaseModel):
    """Group B — stock with zero detected relationship to any of user's holdings."""
    ticker:            str
    sector:            str
    centrality:        float
    composite_score:   float
    sector_gap_score:  float
    centrality_score:  float
    reasoning:         str


class DcorCandidate(BaseModel):
    """
    Component 1 output — a candidate stock that passed the dCor threshold filter.

    mean_dcor_to_portfolio is the average pairwise dCor between this stock and
    all user portfolio holdings for which a network pair exists.  Stocks with no
    network pairs at all are assigned 0.0 (no detected dependency).

    paired_holdings maps each portfolio holding that has a detected pair with
    this candidate to its individual dCor value — used by the UI to annotate
    a per-holding price comparison chart.

    The list is sorted ascending by mean_dcor_to_portfolio so the most
    independent stocks appear first.  Feed this list to Component 2 (clustering)
    and Component 3 (Monte Carlo risk assessment).
    """
    ticker:                  str
    sector:                  str
    centrality:              float
    mean_dcor_to_portfolio:  float
    n_portfolio_pairs:       int
    paired_holdings:         dict[str, float] = {}
    reasoning:               str


class PortfolioAnalysisResponse(BaseModel):
    tickers_analyzed:            list[str]
    unknown_tickers:             list[str]
    overlaps:                    list[OverlapResult]
    signal_recommendations:      list[Recommendation]
    independent_recommendations: list[IndependentRecommendation]
    holdings_sectors:            dict[str, str]   # {ticker: sector} for all known holdings
    # Component 1 output — dCor-filtered diversification candidate pool
    # sorted ascending by mean_dcor_to_portfolio (most independent first)
    dcor_filtered_candidates:    list[DcorCandidate] = []