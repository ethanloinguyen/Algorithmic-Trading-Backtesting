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


class StockListResponse(BaseModel):
    data: list[StockSummary]


class IndexListResponse(BaseModel):
    data: list[IndexSummary]