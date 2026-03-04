# backend/app/services/bigquery_service.py
from __future__ import annotations

from datetime import datetime, timezone
from google.cloud import bigquery

from app.core.bigquery import get_bq_client
from app.core.config import get_settings
from app.models.stock import OHLCVCandle, StockSummary, IndexSummary, TimeRange

# ── Constants ─────────────────────────────────────────────────────────────────

# Number of calendar days to look back per range
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
    """Convert raw volume to abbreviated string: 52300000 → '52.3M'"""
    vol = float(vol)
    if vol >= 1_000_000_000:
        return f"{vol / 1_000_000_000:.1f}B"
    if vol >= 1_000_000:
        return f"{vol / 1_000_000:.1f}M"
    if vol >= 1_000:
        return f"{vol / 1_000:.1f}K"
    return str(int(vol))


def _format_price(n: float) -> str:
    """Format price as '$189.84'"""
    return f"${n:,.2f}"


def _format_date_label(iso_date: str, range_: TimeRange) -> str:
    """
    Convert a raw ISO date/datetime string into the display label the
    frontend expects per time range.
    """
    try:
        d = datetime.fromisoformat(iso_date)
    except ValueError:
        return iso_date

    if range_ == TimeRange.ONE_DAY:
        return d.strftime("%-I:%M %p")   # "9:30 AM"
    if range_ == TimeRange.FIVE_YEARS:
        return d.strftime("%b '%y")       # "Mar '24"
    return d.strftime("%b %d")            # "Mar 01"


# ── Queries ───────────────────────────────────────────────────────────────────

def get_ohlcv(symbol: str, range_: TimeRange) -> list[OHLCVCandle]:
    """
    Fetch OHLCV candles for a single symbol over the requested time range.

    Expected BigQuery table schema (ohlcv_daily):
        symbol        STRING
        trade_date    DATE (or DATETIME for intraday)
        open_price    FLOAT64
        high_price    FLOAT64
        low_price     FLOAT64
        close_price   FLOAT64
        volume        INT64
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


def _summaries_from_query(rows) -> list[StockSummary]:
    """Shared row-to-model conversion for summary queries."""
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
    # """
    # Fetch the latest price, change, and volume for every stock in the database.
    # Uses a window function to grab the two most recent trading days per symbol
    # so we can compute day-over-day percentage change.
    # Used to populate the Featured Stocks table.
    # """
    # client   = get_bq_client()
    # settings = get_settings()

    # query = f"""
    #     WITH ranked AS (
    #         SELECT
    #             o.ticker,
    #             COALESCE(s.company_name, o.ticker) AS company_name,
    #             o.date,
    #             o.close,
    #             o.volume,
    #             ROW_NUMBER() OVER (PARTITION BY o.ticker ORDER BY o.date DESC) AS rn
    #         FROM {settings.fq_market_data} o
    #         LEFT JOIN {settings.fq_ticker_metadata} s ON o.ticker = s.ticker
    #     ),
    #     today AS (
    #         SELECT ticker, company_name,
    #                close  AS today_close,
    #                volume AS today_volume
    #         FROM ranked WHERE rn = 1
    #     ),
    #     yesterday AS (
    #         SELECT ticker, close AS prev_close
    #         FROM ranked WHERE rn = 2
    #     )
    #     SELECT
    #         t.ticker,
    #         t.company_name,
    #         t.today_close,
    #         t.today_volume,
    #         COALESCE(y.prev_close, t.today_close) AS prev_close
    #     FROM today t
    #     LEFT JOIN yesterday y USING (ticker)
    #     ORDER BY t.ticker
    #     LIMIT 15
    # """

    # rows = client.query(query).result()
    # return _summaries_from_query(rows)

    """
    Fetch the latest price, change, and volume for a curated list
    of popular stocks. Used to populate the Featured Stocks table.
    """
    return get_stock_summaries(FEATURED_TICKERS)


def get_stock_summaries(symbols: list[str]) -> list[StockSummary]:
    """
    Fetch summaries for a specific list of symbols.
    Used by the profile page to refresh starred-stock prices.
    """
    if not symbols:
        return []

    client   = get_bq_client()
    settings = get_settings()

    # BigQuery parameterized ARRAY queries
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
            bigquery.ArrayQueryParameter("symbols", "STRING", [s.upper() for s in symbols])
        ]
    )

    rows = client.query(query, job_config=job_config).result()
    return _summaries_from_query(rows)


def get_index_summaries() -> list[IndexSummary]:
    """
    Fetch the latest data for SPX, IXIC, and DJI.
    These are expected to exist as rows in the stocks + ohlcv tables.
    """
    summaries = get_stock_summaries(list(INDEX_META.keys()))

    results: list[IndexSummary] = []
    for s in summaries:
        # Strip the "$" and commas to get the raw numeric display value
        raw_value = s.price.lstrip("$").replace(",", "")
        # Split "+1.25%" into change amount and pct — for indices the
        # change field already contains the pct; raw point change comes
        # from the price delta which we approximate here
        results.append(IndexSummary(
            symbol   = s.symbol,
            name     = INDEX_META.get(s.symbol, s.name),
            value    = raw_value,
            change   = s.change.split("(")[0].strip() if "(" in s.change else s.change,
            pct      = s.change,
            price    = s.price,
            positive = s.positive,
        ))

    return results