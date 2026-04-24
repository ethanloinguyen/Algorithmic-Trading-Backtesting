# backend/app/routers/portfolio.py
from fastapi import APIRouter, HTTPException
from app.models.portfolio import PortfolioAnalysisResponse, PortfolioRequest
from app.services.portfolio_service import run_portfolio_analysis

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