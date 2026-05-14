# backend/app/models/portfolio.py
from typing import Any, Literal
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


class QualityRecommendation(BaseModel):
    """
    Quality Picks — stock scored on five portfolio-aware quality dimensions.
    Three scores (momentum, fundamental_quality, centrality) are precomputed
    nightly; two (sector_diversity, volatility_compatibility) are computed
    on-the-fly per user request.

    Weights: momentum 35% · fundamental_quality 25% · sector_diversity 20%
             · volatility_compatibility 10% · centrality 10%
    """
    ticker:                          str
    sector:                          str
    centrality:                      float   # raw eigenvector centrality
    composite_score:                 float
    momentum_score:                  float
    fundamental_quality_score:       float
    sector_diversity_score:          float
    volatility_compatibility_score:  float
    centrality_score:                float
    reasoning:                       str


class PortfolioAnalysisResponse(BaseModel):
    tickers_analyzed:            list[str]
    unknown_tickers:             list[str]
    overlaps:                    list[OverlapResult]
    signal_recommendations:      list[Recommendation]
    independent_recommendations: list[IndependentRecommendation]
    holdings_sectors:            dict[str, str]   # {ticker: sector} for all known holdings


# ---------------------------------------------------------------------------
# Pipeline: hierarchical clustering → Monte Carlo risk assessment
# ---------------------------------------------------------------------------

class PipelineRequest(BaseModel):
    tickers:           list[str]
    horizon_days:      Literal[21, 63, 126, 252] = 63
    n_sims:            int = 1000
    target_return:     float = 0.10
    confidence_levels: list[float] = [0.95, 0.99]
    seed:              int = 42


class ClusteringRecommendation(BaseModel):
    sector:                str
    stock:                 str
    cluster:               int
    is_medoid:             bool
    avg_dcor_to_portfolio: float
    mean_intra_dist:       float
    n_sector_candidates:   int
    cluster_size:          int


class PipelineResponse(BaseModel):
    user_portfolio:  list[str]
    recommendations: list[ClusteringRecommendation]
    risk:            dict[str, Any]  # full mc_engine output — see mc_engine.py docstring
    quality_picks:               list[QualityRecommendation]
    holdings_sectors:            dict[str, str]   # {ticker: sector} for all known holdings
