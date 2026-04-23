# backend/app/routers/pairs.py
from typing import Literal
from fastapi import APIRouter, HTTPException

from app.models.stock import StockDetail, PairDetail
from app.services.bigquery_services import get_stock_detail, get_pair_data

router = APIRouter(prefix="/api", tags=["analysis"])


@router.get("/stocks/{symbol}/detail", response_model=StockDetail)
def stock_detail(symbol: str):
    """
    Extended stock info for the Analysis page fundamentals panel.
    Returns market_cap, pe_ratio, sector, industry, 52W high/low
    computed from ticker_metadata + market_data.
    """
    sym = symbol.strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail="Symbol is required.")
    return get_stock_detail(sym)


@router.get("/pairs/{ticker_i}/{ticker_j}", response_model=PairDetail)
def pair_detail(
    ticker_i: str,
    ticker_j: str,
    analysis_mode: Literal["broad_market", "in_sector"] = "broad_market",
):
    """
    Lead-lag pair details for the Lag Alignment Lab.
    Queries final_network (broad_market) or sector_final_network (in_sector).
    Returns found=False with zero values if no relationship is detected.
    """
    ti = ticker_i.strip().upper()
    tj = ticker_j.strip().upper()
    if not ti or not tj:
        raise HTTPException(status_code=400, detail="Both tickers are required.")
    if ti == tj:
        raise HTTPException(status_code=400, detail="Tickers must be different.")
    return get_pair_data(ti, tj, analysis_mode)
