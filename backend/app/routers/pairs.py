# backend/app/routers/pairs.py
from typing import Literal
from fastapi import APIRouter, HTTPException, Query

from app.models.stock import StockDetail, PairDetail, NetworkResponse
from app.services.bigquery_services import get_stock_detail, get_pair_data, get_network_data

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


@router.get("/network", response_model=NetworkResponse)
def network_graph(
    analysis_mode: Literal["broad_market", "in_sector"] = "broad_market",
    min_signal:    float = Query(default=55.0, ge=0.0,  le=100.0),
    limit:         int   = Query(default=50,   ge=10,   le=100),
):
    """
    Top `limit` stocks by eigenvector centrality and all directed edges
    between them with signal_strength >= min_signal, from the latest
    network snapshot. Used to power the Analysis page network graph.
    """
    return get_network_data(analysis_mode, min_signal, limit)
