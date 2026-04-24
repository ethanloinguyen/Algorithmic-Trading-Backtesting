# backend/app/services/bigquery_services.py
from __future__ import annotations

from datetime import datetime, timezone
from google.cloud import bigquery

from app.core.bigquery import get_bq_client
from app.core.config import get_settings
from app.models.stock import (
    OHLCVCandle, StockSummary, StockDetail, PairDetail, IndexSummary, TimeRange,
    NetworkNodeModel, NetworkEdgeModel, NetworkResponse,
)

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

    Expected BigQuery table schema for market_data:
        ticker      STRING
        date        DATE        ← DATE type, not DATETIME
        open        FLOAT64
        high        FLOAT64
        low         FLOAT64
        close       FLOAT64
        volume      INT64

    NOTE: Uses DATE_SUB + CURRENT_DATE() (not DATETIME_SUB) because the
    `date` column is type DATE. If your table uses DATETIME, switch to
    DATETIME_SUB(CURRENT_DATETIME(), INTERVAL @days DAY).
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
            AND date >= DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY)
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
    """
    Fetch the latest price, change, and volume for a curated list
    of popular stocks. Used to populate the Watchlist table.
    """
    return get_stock_summaries(FEATURED_TICKERS)


def get_stock_summaries(symbols: list[str]) -> list[StockSummary]:
    """
    Fetch summaries for a specific list of symbols.
    Used by the profile page to refresh starred-stock prices.

    Expected BigQuery table schemas:
      market_data:     ticker STRING, date DATE, open FLOAT64, high FLOAT64,
                       low FLOAT64, close FLOAT64, volume INT64
      ticker_metadata: ticker STRING, company_name STRING
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
            bigquery.ArrayQueryParameter("symbols", "STRING", [s.upper() for s in symbols])
        ]
    )

    rows = client.query(query, job_config=job_config).result()
    return _summaries_from_query(rows)


def get_index_summaries() -> list[IndexSummary]:
    """
    Fetch the latest data for SPX, IXIC, and DJI.
    These are expected to exist as tickers in the market_data table.
    """
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


def get_stock_detail(symbol: str) -> StockDetail:
    """
    Fetch extended stock info for the Analysis page fundamentals panel.
    Pulls market_cap, pe_ratio, sector, industry from ticker_metadata,
    and computes 52W high/low from market_data (no external API needed).
    """
    client   = get_bq_client()
    settings = get_settings()
    sym      = symbol.upper()

    query = f"""
        SELECT
            tm.company_name,
            tm.sector,
            tm.industry,
            tm.market_cap,
            tm.pe_ratio,
            w.high_52w,
            w.low_52w
        FROM (
            SELECT company_name, sector, industry, market_cap, pe_ratio
            FROM {settings.fq_ticker_metadata}
            WHERE ticker = @symbol
        ) tm
        CROSS JOIN (
            SELECT
                MAX(close) AS high_52w,
                MIN(close) AS low_52w
            FROM {settings.fq_market_data}
            WHERE ticker = @symbol
              AND date >= DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY)
        ) w
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("symbol", "STRING", sym),
        ]
    )

    rows = list(client.query(query, job_config=job_config).result())
    if not rows:
        return StockDetail(symbol=sym, name=sym)

    row = rows[0]
    return StockDetail(
        symbol     = sym,
        name       = row.company_name or sym,
        sector     = row.sector,
        industry   = row.industry,
        market_cap = row.market_cap,
        pe_ratio   = float(row.pe_ratio) if row.pe_ratio is not None else None,
        high_52w   = float(row.high_52w) if row.high_52w is not None else None,
        low_52w    = float(row.low_52w) if row.low_52w is not None else None,
    )


def get_pair_data(
    ticker_i: str,
    ticker_j: str,
    analysis_mode: str = "broad_market",
) -> PairDetail:
    """
    Fetch lead-lag pair details for (ticker_i, ticker_j) from the appropriate
    network table. Returns a PairDetail with found=False if no relationship exists.

    Checks both orderings (i→j and j→i) and returns the row with the highest
    signal_strength.
    """
    from app.services.portfolio_service import _TABLE_NAMES

    client     = get_bq_client()
    settings   = get_settings()
    table_name = _TABLE_NAMES.get(analysis_mode, "final_network")
    table      = f"`{settings.gcp_project_id}.{settings.bq_dataset}.{table_name}`"
    ti         = ticker_i.upper()
    tj         = ticker_j.upper()

    query = f"""
        SELECT
            ticker_i, ticker_j, best_lag, mean_dcor, signal_strength,
            frequency, half_life, COALESCE(oos_sharpe_net, 0.0) AS oos_sharpe_net,
            sector_i, sector_j
        FROM {table}
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM {table})
          AND (
              (ticker_i = @ti AND ticker_j = @tj)
           OR (ticker_i = @tj AND ticker_j = @ti)
          )
        ORDER BY signal_strength DESC
        LIMIT 1
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("ti", "STRING", ti),
            bigquery.ScalarQueryParameter("tj", "STRING", tj),
        ]
    )

    rows = list(client.query(query, job_config=job_config).result())
    if not rows:
        return PairDetail(
            ticker_i=ti, ticker_j=tj, best_lag=0, mean_dcor=0.0,
            signal_strength=0.0, frequency=0.0, half_life=0.0,
            oos_sharpe_net=0.0, sector_i="", sector_j="", found=False,
        )

    r = rows[0]
    return PairDetail(
        ticker_i        = r.ticker_i,
        ticker_j        = r.ticker_j,
        best_lag        = int(r.best_lag),
        mean_dcor       = round(float(r.mean_dcor), 4),
        signal_strength = round(float(r.signal_strength), 1),
        frequency       = round(float(r.frequency), 3),
        half_life       = round(float(r.half_life), 1),
        oos_sharpe_net  = round(float(r.oos_sharpe_net), 3),
        sector_i        = r.sector_i or "",
        sector_j        = r.sector_j or "",
        found           = True,
    )


def get_network_data(
    analysis_mode: str = "broad_market",
    min_signal:    float = 55.0,
    limit:         int   = 50,
) -> NetworkResponse:
    """
    Return the top `limit` nodes from the latest network snapshot, plus all
    directed edges between them above min_signal.

    Primary query ranks nodes by eigenvector centrality (centrality_i/j columns).
    Falls back to ranking by connection frequency if those columns don't exist yet,
    assigning centrality=0.0 so the frontend renders all nodes at a fixed size.
    """
    from app.services.portfolio_service import _TABLE_NAMES

    client     = get_bq_client()
    settings   = get_settings()
    table_name = _TABLE_NAMES.get(analysis_mode, "final_network")
    table      = f"`{settings.gcp_project_id}.{settings.bq_dataset}.{table_name}`"

    # NOTE: LIMIT in a CTE cannot be parameterised in BigQuery, so limit is
    # injected via f-string.  It is clamped to [10, 100] by the router before
    # reaching this function, so there is no injection risk.
    query_with_centrality = f"""
        WITH latest AS (
            SELECT MAX(as_of_date) AS max_date FROM {table}
        ),
        filtered AS (
            SELECT
                ticker_i, ticker_j,
                signal_strength, best_lag, mean_dcor,
                COALESCE(sector_i, 'Unknown') AS sector_i,
                COALESCE(sector_j, 'Unknown') AS sector_j,
                COALESCE(centrality_i, 0.0)   AS cent_i,
                COALESCE(centrality_j, 0.0)   AS cent_j
            FROM {table}, latest
            WHERE as_of_date = latest.max_date
              AND signal_strength >= @min_signal
        ),
        all_node_cents AS (
            SELECT ticker_i AS ticker, sector_i AS sector, MAX(cent_i) AS centrality
            FROM filtered GROUP BY ticker_i, sector_i
            UNION ALL
            SELECT ticker_j AS ticker, sector_j AS sector, MAX(cent_j) AS centrality
            FROM filtered GROUP BY ticker_j, sector_j
        ),
        top_nodes AS (
            SELECT ticker, ANY_VALUE(sector) AS sector, MAX(centrality) AS centrality
            FROM all_node_cents
            GROUP BY ticker
            ORDER BY MAX(centrality) DESC
            LIMIT {limit}
        )
        SELECT
            f.ticker_i, f.ticker_j,
            f.signal_strength, f.best_lag, f.mean_dcor,
            f.sector_i, f.sector_j,
            f.cent_i,   f.cent_j
        FROM filtered f
        WHERE f.ticker_i IN (SELECT ticker FROM top_nodes)
          AND f.ticker_j IN (SELECT ticker FROM top_nodes)
        ORDER BY f.signal_strength DESC
        LIMIT 2000
    """

    # Fallback: no centrality columns — rank by connection frequency, fixed size (0.0)
    query_no_centrality = f"""
        WITH latest AS (
            SELECT MAX(as_of_date) AS max_date FROM {table}
        ),
        filtered AS (
            SELECT
                ticker_i, ticker_j,
                signal_strength, best_lag, mean_dcor,
                COALESCE(sector_i, 'Unknown') AS sector_i,
                COALESCE(sector_j, 'Unknown') AS sector_j,
                0.0 AS cent_i,
                0.0 AS cent_j
            FROM {table}, latest
            WHERE as_of_date = latest.max_date
              AND signal_strength >= @min_signal
        ),
        all_tickers AS (
            SELECT ticker_i AS ticker, sector_i AS sector FROM filtered
            UNION ALL
            SELECT ticker_j AS ticker, sector_j AS sector FROM filtered
        ),
        top_nodes AS (
            SELECT ticker, ANY_VALUE(sector) AS sector
            FROM all_tickers
            GROUP BY ticker
            ORDER BY COUNT(*) DESC
            LIMIT {limit}
        )
        SELECT
            f.ticker_i, f.ticker_j,
            f.signal_strength, f.best_lag, f.mean_dcor,
            f.sector_i, f.sector_j,
            f.cent_i,   f.cent_j
        FROM filtered f
        WHERE f.ticker_i IN (SELECT ticker FROM top_nodes)
          AND f.ticker_j IN (SELECT ticker FROM top_nodes)
        ORDER BY f.signal_strength DESC
        LIMIT 2000
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("min_signal", "FLOAT64", min_signal),
        ]
    )

    try:
        rows = list(client.query(query_with_centrality, job_config=job_config).result())
    except Exception:
        # centrality_i / centrality_j columns not yet present — use fixed-size fallback
        rows = list(client.query(query_no_centrality, job_config=job_config).result())

    # Build node map: ticker → {sector, max centrality}
    node_map:   dict[str, dict]  = {}
    out_degree: dict[str, int]   = {}
    edges:      list[NetworkEdgeModel] = []

    for r in rows:
        for ticker, sector, cent in [
            (r.ticker_i, r.sector_i, float(r.cent_i)),
            (r.ticker_j, r.sector_j, float(r.cent_j)),
        ]:
            if ticker not in node_map:
                node_map[ticker] = {"sector": sector, "centrality": cent}
            else:
                node_map[ticker]["centrality"] = max(node_map[ticker]["centrality"], cent)

        out_degree[r.ticker_i] = out_degree.get(r.ticker_i, 0) + 1

        edges.append(NetworkEdgeModel(
            source          = r.ticker_i,
            target          = r.ticker_j,
            signal_strength = round(float(r.signal_strength), 1),
            best_lag        = int(r.best_lag),
            mean_dcor       = round(float(r.mean_dcor), 4),
        ))

    nodes = [
        NetworkNodeModel(
            id          = ticker,
            sector      = info["sector"],
            centrality  = round(info["centrality"], 4),
            out_degree  = out_degree.get(ticker, 0),
        )
        for ticker, info in node_map.items()
    ]

    return NetworkResponse(
        nodes         = nodes,
        edges         = edges,
        analysis_mode = analysis_mode,
        min_signal    = min_signal,
    )