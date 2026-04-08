# backend/app/routers/portfolio.py
from fastapi import APIRouter, HTTPException
from app.models.portfolio import PortfolioAnalysisResponse, PortfolioRequest
from app.services.portfolio_service import run_portfolio_analysis

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


@router.post("/analyze", response_model=PortfolioAnalysisResponse)
def analyze_portfolio(body: PortfolioRequest):
    """
    Analyze a portfolio for hidden lead-lag concentration risk
    and return ranked diversification recommendations.

    Body: { "tickers": ["AAPL", "MSFT", "NVDA"] }
    """
    if not body.tickers:
        raise HTTPException(status_code=400, detail="At least one ticker is required.")
    if len(body.tickers) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 tickers per request.")

    result = run_portfolio_analysis(tickers=body.tickers)
    return PortfolioAnalysisResponse(**result)