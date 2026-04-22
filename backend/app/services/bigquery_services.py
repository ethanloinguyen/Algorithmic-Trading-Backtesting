# backend/app/services/bigquery_services.py
from __future__ import annotations

from datetime import datetime
from google.cloud import bigquery

from app.core.bigquery import get_bq_client
from app.core.config import get_settings
from app.models.stock import OHLCVCandle, StockSummary, IndexSummary, TimeRange

# ── Constants ─────────────────────────────────────────────────────────────────

# Calendar days to look back per time range
RANGE_DAYS: dict[TimeRange, int] = {
    TimeRange.ONE_DAY:      1,
    TimeRange.ONE_WEEK:     7,
    TimeRange.ONE_MONTH:    30,
    TimeRange.THREE_MONTHS: 90,
    TimeRange.ONE_YEAR:     365,
    TimeRange.FIVE_YEARS:   1825,
}

INDEX_META: dict[str, str] = {
    "SPX":  "S&P 500 Index",
    "IXIC": "NASDAQ Composite",
    "DJI":  "Dow Jones Industrial Average",
}

FEATURED_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "JPM", "V", "UNH",
    "XOM", "JNJ", "WMT", "AVGO", "AMD",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _abbreviate_volume(vol: int | float) -> str:
    vol = float(vol)
    if vol >= 1_000_000_000:
        return f"{vol / 1_000_000_000:.1f}B"
    if vol >= 1_000_000:
        return f"{vol / 1_000_000:.1f}M"
    if vol >= 1_000:
        return f"{vol / 1_000:.1f}K"
    return str(int(vol))


def _format_price(n: float) -> str:
    return f"${n:,.2f}"


def _format_date_label(iso_date: str, range_: TimeRange) -> str:
    """
    Convert BigQuery DATETIME string → display label.

    BigQuery CAST(datetime_col AS STRING) produces ISO-8601 strings like:
        "2024-01-15T09:30:00" or "2024-01-15 09:30:00"
    Both are handled by datetime.fromisoformat() in Python 3.7+.
    """
    try:
        d = datetime.fromisoformat(iso_date.replace(" ", "T"))
    except ValueError:
        return iso_date  # return raw string if unparseable

    if range_ == TimeRange.ONE_DAY:
        return d.strftime("%-I:%M %p")   # "9:30 AM"
    if range_ == TimeRange.FIVE_YEARS:
        return d.strftime("%b '%y")       # "Mar '24"
    return d.strftime("%b %d")            # "Mar 01"


# ── OHLCV query ───────────────────────────────────────────────────────────────

def get_ohlcv(symbol: str, range_: TimeRange) -> list[OHLCVCandle]:
    """
    Fetch OHLCV candles for a single symbol over the requested time range.

    Confirmed BigQuery schema for market_data (capstone-487001):
        date        DATETIME   ← DATETIME not DATE — must use DATETIME_SUB
        adj_close   FLOAT
        close       FLOAT
        high        FLOAT
        low         FLOAT
        open        FLOAT
        volume      INTEGER
        ticker      STRING
        log_return  FLOAT
    """
    client   = get_bq_client()
    settings = get_settings()
    days     = RANGE_DAYS[range_]

    query = f"""
        SELECT
            CAST(date AS STRING) AS date,
            open,
            high,
            low,
            close,
            volume
        FROM {settings.fq_market_data}
        WHERE
            ticker = @symbol
            AND date >= DATETIME_SUB(CURRENT_DATETIME(), INTERVAL @days DAY)
        ORDER BY date ASC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("symbol", "STRING", symbol.upper()),
            bigquery.ScalarQueryParameter("days",   "INT64",  days),
        ]
    )

    rows = client.query(query, job_config=job_config).result()

    candles: list[OHLCVCandle] = []
    for row in rows:
        candles.append(OHLCVCandle(
            date   = _format_date_label(str(row.date), range_),
            open   = f"{row.open:.2f}",
            high   = f"{row.high:.2f}",
            low    = f"{row.low:.2f}",
            close  = f"{row.close:.2f}",
            volume = _abbreviate_volume(row.volume),
        ))

    return candles


# ── Summary queries ───────────────────────────────────────────────────────────

def _summaries_from_query(rows) -> list[StockSummary]:
    results: list[StockSummary] = []
    for row in rows:
        prev  = row.prev_close
        today = row.today_close
        pct   = ((today - prev) / prev * 100) if prev else 0.0
        positive   = pct >= 0
        change_str = f"{'+'if positive else ''}{pct:.2f}%"
        results.append(StockSummary(
            symbol   = row.ticker,
            name     = row.company_name,
            price    = _format_price(today),
            change   = change_str,
            volume   = _abbreviate_volume(row.today_volume),
            positive = positive,
        ))
    return results


def get_all_stock_summaries() -> list[StockSummary]:
    """Fetch summaries for the featured tickers list."""
    return get_stock_summaries(FEATURED_TICKERS)


def get_stock_summaries(symbols: list[str]) -> list[StockSummary]:
    """
    Fetch latest price, change %, and volume for a list of symbols.

    Uses a window function to get the two most recent DATETIME rows per
    ticker so we can compute day-over-day % change.

    The LEFT JOIN against ticker_metadata is optional — if that table does
    not exist or has no matching row, company_name falls back to the ticker
    symbol itself via COALESCE.
    """
    if not symbols:
        return []

    client   = get_bq_client()
    settings = get_settings()

    query = f"""
        WITH ranked AS (
            SELECT
                o.ticker,
                COALESCE(s.company_name, o.ticker) AS company_name,
                o.date,
                o.close,
                o.volume,
                ROW_NUMBER() OVER (PARTITION BY o.ticker ORDER BY o.date DESC) AS rn
            FROM {settings.fq_market_data} o
            LEFT JOIN {settings.fq_ticker_metadata} s ON o.ticker = s.ticker
            WHERE o.ticker IN UNNEST(@symbols)
        ),
        today AS (
            SELECT ticker, company_name,
                   close  AS today_close,
                   volume AS today_volume
            FROM ranked WHERE rn = 1
        ),
        yesterday AS (
            SELECT ticker, close AS prev_close
            FROM ranked WHERE rn = 2
        )
        SELECT
            t.ticker,
            t.company_name,
            t.today_close,
            t.today_volume,
            COALESCE(y.prev_close, t.today_close) AS prev_close
        FROM today t
        LEFT JOIN yesterday y USING (ticker)
        ORDER BY t.ticker
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter(
                "symbols", "STRING", [s.upper() for s in symbols]
            )
        ]
    )

    rows = client.query(query, job_config=job_config).result()
    return _summaries_from_query(rows)


def get_index_summaries() -> list[IndexSummary]:
    """Fetch latest data for SPX, IXIC, DJI."""
    summaries = get_stock_summaries(list(INDEX_META.keys()))

    results: list[IndexSummary] = []
    for s in summaries:
        raw_value = s.price.lstrip("$").replace(",", "")
        results.append(IndexSummary(
            symbol   = s.symbol,
            name     = INDEX_META.get(s.symbol, s.name),
            value    = raw_value,
            change   = s.change,
            pct      = s.change,
            price    = s.price,
            positive = s.positive,
        ))
    return results