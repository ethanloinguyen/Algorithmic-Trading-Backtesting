# backend/app/routers/stocks.py
import re
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from app.models.stock import (
    OHLCVResponse,
    StockListResponse,
    TimeRange,
)
from app.services.bigquery_service import (
    get_all_stock_summaries,
    get_ohlcv,
    get_stock_summaries,
)

router = APIRouter(prefix="/api/stocks", tags=["stocks"])

_SYMBOL_RE = re.compile(r"^[A-Z^.]{1,10}$")


def _validate_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if not _SYMBOL_RE.match(s):
        raise HTTPException(status_code=400, detail=f"Invalid symbol: '{symbol}'")
    return s


# ── GET /api/stocks ───────────────────────────────────────────────────────────
@router.get("", response_model=StockListResponse)
def list_stocks():
    """
    Returns the latest summary (price, % change, volume) for every stock
    in the database. Used to auto-populate the Featured Stocks table.
    """
    data = get_all_stock_summaries()
    return StockListResponse(data=data)


# ── GET /api/stocks/summaries?symbols=AAPL,MSFT ──────────────────────────────
@router.get("/summaries", response_model=StockListResponse)
def stock_summaries(
    symbols: Annotated[str, Query(description="Comma-separated ticker list")]
):
    """
    Returns latest summaries for a specific list of symbols.
    Used by the profile page to refresh starred-stock prices.
    """
    symbol_list = [_validate_symbol(s) for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="At least one symbol is required")
    data = get_stock_summaries(symbol_list)
    return StockListResponse(data=data)


# ── GET /api/stocks/{symbol}/ohlcv?range=1M ──────────────────────────────────
@router.get("/{symbol}/ohlcv", response_model=OHLCVResponse)
def stock_ohlcv(
    symbol: str,
    range: TimeRange = TimeRange.ONE_MONTH,
):
    """
    Returns OHLCV candles for a single symbol over the requested time range.
    Used by StockModal when a user clicks on any stock or index card.

    - **symbol**: ticker (e.g. `AAPL`)
    - **range**: `1D` | `1W` | `1M` | `3M` | `1Y` | `5Y`  (default `1M`)
    """
    sym     = _validate_symbol(symbol)
    candles = get_ohlcv(sym, range)

    if not candles:
        raise HTTPException(
            status_code=404,
            detail=f"No OHLCV data found for '{sym}' in range '{range.value}'"
        )

    return OHLCVResponse(symbol=sym, range=range, candles=candles)