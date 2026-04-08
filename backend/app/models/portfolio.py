# backend/app/models/portfolio.py
from pydantic import BaseModel


class PortfolioRequest(BaseModel):
    tickers: list[str]


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
    ticker:                     str
    sector:                     str
    centrality:                 float
    composite_score:            float
    n_portfolio_relationships:  int
    best_relationship_strength: float
    mean_relationship_strength: float
    related_holdings:           list[str]
    direction:                  str   # "leads_your_holdings" | "follows_your_holdings" | "both"
    signal_score:               float
    centrality_score:           float
    sector_diversity_score:     float
    coverage_score:             float
    reasoning:                  str


class PortfolioAnalysisResponse(BaseModel):
    tickers_analyzed: list[str]
    unknown_tickers:  list[str]
    overlaps:         list[OverlapResult]
    recommendations:  list[Recommendation]