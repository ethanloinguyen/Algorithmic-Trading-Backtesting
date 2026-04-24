# backend/app/models/stock.py
from pydantic import BaseModel
from enum import Enum


class TimeRange(str, Enum):
    ONE_DAY   = "1D"
    ONE_WEEK  = "1W"
    ONE_MONTH = "1M"
    THREE_MONTHS = "3M"
    ONE_YEAR  = "1Y"
    FIVE_YEARS = "5Y"


class OHLCVCandle(BaseModel):
    """A single OHLCV candle, formatted for display in the frontend."""
    date: str    # e.g. "Mar 01" or "09:30 AM" for intraday
    open: str    # e.g. "189.12"
    high: str
    low: str
    close: str
    volume: str  # e.g. "52.3M"


class StockSummary(BaseModel):
    """Summary row used in the Featured Stocks / watchlist table."""
    symbol: str     # "AAPL"
    name: str       # "Apple Inc."
    price: str      # "$189.84"
    change: str     # "+1.25%"
    volume: str     # "52.3M"
    positive: bool  # True if change >= 0


class IndexSummary(BaseModel):
    """Index card data for SPX, IXIC, DJI."""
    symbol: str
    name: str
    value: str      # "5,248.49"
    change: str     # "+32.64"
    pct: str        # "+0.63%"
    price: str      # same as value with $ prefix, used by StockModal
    positive: bool


class OHLCVResponse(BaseModel):
    """Response body for GET /api/stocks/{symbol}/ohlcv"""
    symbol: str
    range: TimeRange
    candles: list[OHLCVCandle]


class StockDetail(BaseModel):
    """Extended stock info for the Analysis page fundamentals panel."""
    symbol:     str
    name:       str
    sector:     str | None = None
    industry:   str | None = None
    market_cap: int | None = None   # raw integer from ticker_metadata
    pe_ratio:   float | None = None
    high_52w:   float | None = None  # computed from market_data
    low_52w:    float | None = None


class PairDetail(BaseModel):
    """Lead-lag pair details from final_network / sector_final_network."""
    ticker_i:       str
    ticker_j:       str
    best_lag:       int
    mean_dcor:      float
    signal_strength: float
    frequency:      float
    half_life:      float
    oos_sharpe_net: float
    sector_i:       str
    sector_j:       str
    found:          bool = True   # False when no relationship detected


class StockListResponse(BaseModel):
    data: list[StockSummary]


class IndexListResponse(BaseModel):
    data: list[IndexSummary]


# ── Network graph models ───────────────────────────────────────────────────────

class NetworkNodeModel(BaseModel):
    """A node in the lead-lag network graph (one stock)."""
    id:          str
    sector:      str
    centrality:  float   # raw eigenvector centrality from final_network
    out_degree:  int     # number of outgoing (leader) edges in filtered graph


class NetworkEdgeModel(BaseModel):
    """A directed edge: source leads target."""
    source:          str
    target:          str
    signal_strength: float
    best_lag:        int
    mean_dcor:       float


class NetworkResponse(BaseModel):
    nodes:         list[NetworkNodeModel]
    edges:         list[NetworkEdgeModel]
    analysis_mode: str
    min_signal:    float