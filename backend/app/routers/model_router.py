# backend/app/routers/model.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.model_inference import ModelInfoResponse, ModelResponse
from app.services.model_service import get_model_info, run_analysis

router = APIRouter(prefix="/api/model", tags=["model"])


class AnalyzeRequest(BaseModel):
    symbol: str


@router.get("/info", response_model=ModelInfoResponse)
def model_info():
    """
    Returns metadata about the loaded DeltaLag model:
    1,690 available tickers (including AAPL, MSFT, GOOGL etc.),
    sectors, lookback window (L=40), and max lag (l_max=10).
    """
    try:
        return ModelInfoResponse(**get_model_info())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/analyze", response_model=ModelResponse)
def analyze(req: AnalyzeRequest):
    """
    Run DeltaLag inference for a target ticker.

    Returns top 10 leader stocks and sector breakdown for:
      - 5d  (lags 1–5)
      - 10d (lags 1–10)
      - 30d (lags 1–10, re-ranked by signal strength)

    First call fetches data for all tickers from BigQuery (~30–60s).
    Subsequent calls are faster due to BigQuery result caching.
    """
    symbol = req.symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")

    try:
        results = run_analysis(symbol)
        return ModelResponse(target=symbol, results=results)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"[model] analyze error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    
    