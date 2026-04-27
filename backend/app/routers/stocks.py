# backend/app/routers/stocks.py
import re
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from app.models.stock import OHLCVResponse, StockListResponse, TimeRange
from app.services.bigquery_services import get_all_stock_summaries, get_ohlcv, get_stock_summaries
from app.services.cache_service import (
    get_cached_summaries,
    set_cached_summaries,
    get_cached_ohlcv,
    set_cached_ohlcv,
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
    Returns the latest summary for every stock in FEATURED_TICKERS.
    Checks Firestore cache first (TTL 5 min) before querying BigQuery.
    """
    # 1. Try Firestore cache
    cached = get_cached_summaries("stock_summaries")
    if cached is not None:
        return StockListResponse(data=cached)

    # 2. Cache miss — query BigQuery
    data = get_all_stock_summaries()

    # 3. Write result back to Firestore cache
    set_cached_summaries(data, "stock_summaries")

    return StockListResponse(data=data)


# ── GET /api/stocks/summaries?symbols=AAPL,MSFT ──────────────────────────────
@router.get("/summaries", response_model=StockListResponse)
def stock_summaries(
    symbols: Annotated[str, Query(description="Comma-separated ticker list")]
):
    """
    Returns latest summaries for a specific list of symbols.
    Used by the profile page to refresh saved-stock prices.
    No caching here — these are user-specific requests.
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
    Returns OHLCV candles for a single symbol.
    Checks Firestore cache first before querying BigQuery.

    - range: 1D | 1W | 1M | 3M | 1Y | 5Y  (default 1M)
    - TTL:   1 min for 1D, 10 min for all others
    """
    sym = _validate_symbol(symbol)

    # 1. Try Firestore cache
    cached = get_cached_ohlcv(sym, range)
    if cached is not None:
        return OHLCVResponse(symbol=sym, range=range, candles=cached)

    # 2. Cache miss — query BigQuery
    candles = get_ohlcv(sym, range)
    if not candles:
        raise HTTPException(
            status_code=404,
            detail=f"No OHLCV data found for '{sym}' in range '{range.value}'"
        )

    # 3. Write to Firestore cache
    set_cached_ohlcv(sym, range, candles)

    return OHLCVResponse(symbol=sym, range=range, candles=candles)