# backend/app/routers/stocks.py
import re
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from app.models.stock import OHLCVResponse, StockListResponse, TimeRange
from app.services.bigquery_services import (
    get_all_stock_summaries,
    get_ohlcv,
    get_stock_summaries,
)
from app.services.ticker_cache import search_tickers
from app.services.cache_service import (
    get_cached_summaries,
    set_cached_summaries,
    get_cached_ohlcv,
    set_cached_ohlcv,
)
from app.services.dashboard_cache import (
    get_cached_stocks,
    dashboard_cache_ready,
    trigger_refresh_if_stale,
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

    Priority:
      1. In-memory dashboard cache (loaded at startup, refreshed nightly at
         8:30 PM ET) — zero BigQuery / Firestore cost on the hot path.
      2. Firestore cache — covers the brief startup window before the in-memory
         cache has completed its first BigQuery load (~3-5 s).
      3. BigQuery direct query — last resort, only on very first cold start.
    """
    # 1. In-memory cache (always warm after startup)
    if dashboard_cache_ready():
        trigger_refresh_if_stale()   # no-op if fresh; fires background refresh if >23 h old
        return StockListResponse(data=get_cached_stocks())

    # 2. Firestore cache (startup fallback)
    cached = get_cached_summaries("stock_summaries")
    if cached is not None:
        return StockListResponse(data=cached)

    # 3. BigQuery fallback — write result to Firestore so the next request
    #    during startup is served from there rather than hitting BQ again.
    data = get_all_stock_summaries()
    set_cached_summaries(data, "stock_summaries")
    return StockListResponse(data=data)


# ── GET /api/stocks/summaries?symbols=AAPL,MSFT ──────────────────────────────
@router.get("/summaries", response_model=StockListResponse)
def stock_summaries(
    symbols: Annotated[str, Query(description="Comma-separated ticker list")]
):
    """
    Returns latest summaries for a specific list of symbols.

    Single-symbol requests (e.g. from the search-bar modal) are served from a
    5-minute Firestore cache so repeated views of the same stock are instant.
    Multi-symbol requests (e.g. profile page watchlist refresh) bypass the cache
    because the symbol set is user-specific and not worth caching individually.
    """
    symbol_list = [_validate_symbol(s) for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="At least one symbol is required")

    # Cache single-symbol lookups — these are the autocomplete-driven requests
    if len(symbol_list) == 1:
        doc_id = f"stock_summary_{symbol_list[0]}"
        cached = get_cached_summaries(doc_id)
        if cached is not None:
            return StockListResponse(data=cached)
        data = get_stock_summaries(symbol_list)
        if data:
            set_cached_summaries(data, doc_id)
        return StockListResponse(data=data)

    data = get_stock_summaries(symbol_list)
    return StockListResponse(data=data)


# ── GET /api/stocks/search?q=AAPL&limit=20 ───────────────────────────────────
@router.get("/search")
def search_stocks_endpoint(
    q:     Annotated[str, Query(description="Ticker prefix or company name substring")] = "",
    limit: Annotated[int, Query(ge=1, le=50, description="Max results")] = 20,
):
    """
    Autocomplete search — returns up to `limit` stocks whose ticker starts with
    `q` or whose company name contains `q` (case-insensitive).

    Must be declared before /{symbol}/ohlcv so FastAPI doesn't treat
    "search" as a symbol parameter.
    """
    query = q.strip()
    if len(query) < 1:
        return {"data": []}
    data = search_tickers(query, limit)
    return {"data": data}


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