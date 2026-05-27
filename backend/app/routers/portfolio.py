# backend/app/routers/portfolio.py
from fastapi import APIRouter, HTTPException
from app.models.portfolio import (
    PortfolioAnalysisResponse, PortfolioRequest,
    PipelineRequest, PipelineResponse,
    PortfolioRiskResponse, ClusteringPipelineResponse,
)
from app.services.portfolio_service import run_portfolio_analysis
from app.services.pipeline_service import (
    run_risk_pipeline,
    run_portfolio_risk_assessment,
    run_clustering_pipeline,
)
from app.core.bigquery import get_bq_client

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


@router.post("/analyze", response_model=PortfolioAnalysisResponse)
def analyze_portfolio(body: PortfolioRequest):
    """
    Analyze a portfolio for hidden lead-lag concentration risk and return
    two groups of diversification recommendations:
      - signal_recommendations:      stocks with detected connections to your holdings
      - independent_recommendations: stocks with zero detected relationships (pure diversification)
    """
    if not body.tickers:
        raise HTTPException(status_code=400, detail="At least one ticker is required.")
    if len(body.tickers) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 tickers per request.")
    try:
        result = run_portfolio_analysis(tickers=body.tickers, analysis_mode=body.analysis_mode)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return PortfolioAnalysisResponse(**result)


@router.post("/risk-pipeline", response_model=PipelineResponse)
def risk_pipeline(body: PipelineRequest):
    """
    Full pipeline: hierarchical K-Medoids clustering → Monte Carlo risk assessment.

    1. Queries BigQuery for stocks that are decorrelated from the provided holdings.
    2. Clusters candidates via K-Medoids and selects one diversifying stock per sector.
    3. Runs Monte Carlo simulation on those picks and returns per-stock and
       portfolio-level risk metrics (VaR, CVaR, max drawdown, etc.).

    Note: this endpoint calls BigQuery twice and runs a full clustering sweep,
    so response times are typically 15–60 seconds depending on pool size.
    """
    if not body.tickers:
        raise HTTPException(status_code=400, detail="At least one ticker is required.")
    if len(body.tickers) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 tickers per request.")
    try:
        result = run_risk_pipeline(
            user_portfolio=body.tickers,
            horizon_days=body.horizon_days,
            n_sims=body.n_sims,
            target_return=body.target_return,
            confidence_levels=body.confidence_levels,
            seed=body.seed,
            bq_client=get_bq_client(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return PipelineResponse(**result)


@router.post("/portfolio-risk", response_model=PortfolioRiskResponse)
def portfolio_risk(body: PipelineRequest):
    """
    Fast endpoint: Monte Carlo risk assessment for the user's own holdings only.
    No BigQuery clustering — typically resolves in a few seconds.

    Returns portfolio-level metrics (VaR, CVaR, drawdown, diversification benefit).
    """
    if not body.tickers:
        raise HTTPException(status_code=400, detail="At least one ticker is required.")
    if len(body.tickers) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 tickers per request.")
    try:
        result = run_portfolio_risk_assessment(
            user_portfolio=body.tickers,
            horizon_days=body.horizon_days,
            n_sims=body.n_sims,
            target_return=body.target_return,
            confidence_levels=body.confidence_levels,
            seed=body.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return PortfolioRiskResponse(**result)


@router.post("/clustering-pipeline", response_model=ClusteringPipelineResponse)
def clustering_pipeline(body: PipelineRequest):
    """
    Slow endpoint: K-Medoids clustering → per-stock Monte Carlo risk for recommendations.
    Queries BigQuery and runs a full clustering sweep — typically 15–60 seconds.

    Returns recommended tickers and per-stock risk metrics (VaR, CVaR, drawdown).
    """
    if not body.tickers:
        raise HTTPException(status_code=400, detail="At least one ticker is required.")
    if len(body.tickers) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 tickers per request.")
    try:
        result = run_clustering_pipeline(
            user_portfolio=body.tickers,
            horizon_days=body.horizon_days,
            n_sims=body.n_sims,
            target_return=body.target_return,
            confidence_levels=body.confidence_levels,
            seed=body.seed,
            bq_client=get_bq_client(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return ClusteringPipelineResponse(**result)