# backend/app/services/bigquery_services.py
from __future__ import annotations

import logging
from datetime import datetime
from google.cloud import bigquery

logger = logging.getLogger(__name__)

from app.core.bigquery import get_bq_client
from app.core.config import get_settings
from app.models.stock import (
    OHLCVCandle, StockSummary, StockDetail, PairDetail, IndexSummary, TimeRange,
    NetworkNodeModel, NetworkEdgeModel, NetworkResponse,
)

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

    Checks market_data (lead-lag universe) first, then falls back to
    general_market_data (extra stocks like GOOG, PLTR, LMND).

    Confirmed BigQuery schema for both tables (capstone-487001):
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
        FROM (
            SELECT date, open, high, low, close, volume
            FROM {settings.fq_market_data}
            WHERE ticker = @symbol
              AND date >= DATETIME_SUB(CURRENT_DATETIME(), INTERVAL @days DAY)
            UNION ALL
            SELECT date, open, high, low, close, volume
            FROM {settings.fq_general_market_data}
            WHERE ticker = @symbol
              AND date >= DATETIME_SUB(CURRENT_DATETIME(), INTERVAL @days DAY)
        )
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
        WITH combined AS (
            -- Universe stocks (lead-lag analysis set)
            SELECT ticker, date, close, volume
            FROM {settings.fq_market_data}
            WHERE ticker IN UNNEST(@symbols)
            UNION ALL
            -- General extra stocks (GOOG, PLTR, LMND, etc.)
            SELECT ticker, date, close, volume
            FROM {settings.fq_general_market_data}
            WHERE ticker IN UNNEST(@symbols)
        ),
        ranked AS (
            SELECT
                o.ticker,
                COALESCE(s.company_name, o.ticker) AS company_name,
                o.date,
                o.close,
                o.volume,
                ROW_NUMBER() OVER (PARTITION BY o.ticker ORDER BY o.date DESC) AS rn
            FROM combined o
            LEFT JOIN {settings.fq_ticker_metadata} s ON o.ticker = s.ticker
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
            FROM (
                SELECT close FROM {settings.fq_market_data}
                WHERE ticker = @symbol
                  AND date >= DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY)
                UNION ALL
                SELECT close FROM {settings.fq_general_market_data}
                WHERE ticker = @symbol
                  AND date >= DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY)
            )
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
    except Exception as e:
        logger.warning("Network centrality query failed (%s), falling back to degree-ranked query", e)
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


# ── Quality Picks ─────────────────────────────────────────────────────────────

def get_quality_picks(holdings: list[str], top_n: int = 10) -> list[dict]:
    """
    Return top-N Quality Picks for a given portfolio of holdings.

    Three scores (momentum, fundamental_quality, centrality) are fetched from
    the precomputed quality_picks_scores table.  Two portfolio-specific scores
    (sector_diversity, volatility_compatibility) are computed on-the-fly.

    Weights: momentum 35% · fundamental_quality 25% · sector_diversity 20%
             · volatility_compatibility 10% · centrality 10%
    """
    settings  = get_settings()
    client    = get_bq_client()
    holdings_upper = [t.upper() for t in holdings]

    # ── 1. Fetch all precomputed scores ───────────────────────────────────────
    query = f"""
        SELECT
            ticker,
            sector,
            centrality_raw,
            momentum_score,
            fundamental_quality_score,
            centrality_score,
            annualized_vol
        FROM {settings.fq_quality_picks}
        WHERE DATE(updated_at) = (
            SELECT MAX(DATE(updated_at)) FROM {settings.fq_quality_picks}
        )
    """
    rows = list(client.query(query).result())
    if not rows:
        logger.warning("quality_picks_scores table is empty — returning empty list")
        return []

    import math

    # Build a records list, exclude user holdings
    records: list[dict] = []
    for r in rows:
        ticker = r["ticker"]
        if ticker in holdings_upper:
            continue
        records.append({
            "ticker":                    ticker,
            "sector":                    r["sector"] or "Unknown",
            "centrality_raw":            float(r["centrality_raw"] or 0.0),
            "momentum_score":            float(r["momentum_score"] or 50.0),
            "fundamental_quality_score": float(r["fundamental_quality_score"] or 50.0),
            "centrality_score":          float(r["centrality_score"] or 50.0),
            "annualized_vol":            r["annualized_vol"],  # may be None
        })

    if not records:
        return []

    # ── 2. Sector diversity score ─────────────────────────────────────────────
    # Holdings sector distribution
    holdings_sector_query = f"""
        SELECT ticker, sector
        FROM {settings.fq_ticker_metadata}
        WHERE ticker IN UNNEST(@tickers)
    """
    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("tickers", "STRING", holdings_upper)]
    )
    holdings_meta = list(client.query(holdings_sector_query, job_config=job_cfg).result())
    holdings_sectors: dict[str, int] = {}
    for row in holdings_meta:
        sec = row["sector"] or "Unknown"
        holdings_sectors[sec] = holdings_sectors.get(sec, 0) + 1

    total_holdings = max(len(holdings_upper), 1)
    holdings_sector_weights = {s: c / total_holdings for s, c in holdings_sectors.items()}

    def _sector_diversity(candidate_sector: str) -> float:
        """Higher score → candidate sector is underrepresented in portfolio."""
        weight = holdings_sector_weights.get(candidate_sector, 0.0)
        # Invert: 0 overlap → max diversity (100); full overlap → 0
        return round((1.0 - weight) * 100.0, 2)

    # ── 3. Volatility compatibility score ─────────────────────────────────────
    # Portfolio average annualized vol from precomputed table (holdings that exist)
    holdings_vol_query = f"""
        SELECT AVG(annualized_vol) AS avg_vol
        FROM {settings.fq_quality_picks}
        WHERE ticker IN UNNEST(@tickers)
          AND annualized_vol IS NOT NULL
          AND DATE(updated_at) = (
              SELECT MAX(DATE(updated_at)) FROM {settings.fq_quality_picks}
          )
    """
    vol_job_cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("tickers", "STRING", holdings_upper)]
    )
    vol_rows = list(client.query(holdings_vol_query, job_config=vol_job_cfg).result())
    portfolio_avg_vol: float = float(vol_rows[0]["avg_vol"] or 0.20) if vol_rows else 0.20

    # Collect raw vol differences for candidates that have vol data
    vol_diffs: list[tuple[int, float]] = []
    for idx, rec in enumerate(records):
        v = rec["annualized_vol"]
        if v is not None:
            vol_diffs.append((idx, abs(float(v) - portfolio_avg_vol)))

    # Rank-normalize: smaller diff → higher compatibility (closer to 100)
    if vol_diffs:
        sorted_diffs = sorted(vol_diffs, key=lambda x: x[1])
        n_valid = len(sorted_diffs)
        rank_map: dict[int, float] = {}
        for rank, (idx, _) in enumerate(sorted_diffs):
            # rank 0 = smallest diff → score 100; rank n-1 → score ~0
            rank_map[idx] = round((1.0 - rank / max(n_valid - 1, 1)) * 100.0, 2)
    else:
        rank_map = {}

    # ── 4. Composite score + reasoning ────────────────────────────────────────
    W_MOM  = 0.35
    W_FUND = 0.25
    W_DIV  = 0.20
    W_VOL  = 0.10
    W_CENT = 0.10

    results: list[dict] = []
    for idx, rec in enumerate(records):
        sdiv  = _sector_diversity(rec["sector"])
        vcomp = rank_map.get(idx, 50.0)   # fallback 50 if vol missing

        composite = (
            W_MOM  * rec["momentum_score"] +
            W_FUND * rec["fundamental_quality_score"] +
            W_DIV  * sdiv +
            W_VOL  * vcomp +
            W_CENT * rec["centrality_score"]
        )

        # Build concise reasoning
        top_factors = sorted(
            [
                ("Momentum",            rec["momentum_score"]),
                ("Fundamental Quality", rec["fundamental_quality_score"]),
                ("Sector Diversity",    sdiv),
                ("Volatility Fit",      vcomp),
                ("Market Centrality",   rec["centrality_score"]),
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        top2 = [f[0] for f in top_factors[:2]]
        reasoning = (
            f"Strong {top2[0]} ({top_factors[0][1]:.0f}/100) and "
            f"{top2[1]} ({top_factors[1][1]:.0f}/100) scores drive this pick. "
            f"Composite: {composite:.1f}/100."
        )

        results.append({
            "ticker":                         rec["ticker"],
            "sector":                         rec["sector"],
            "centrality":                     rec["centrality_raw"],
            "composite_score":                round(composite, 2),
            "momentum_score":                 rec["momentum_score"],
            "fundamental_quality_score":      rec["fundamental_quality_score"],
            "sector_diversity_score":         sdiv,
            "volatility_compatibility_score": vcomp,
            "centrality_score":               rec["centrality_score"],
            "reasoning":                      reasoning,
        })

    # ── 5. Return top-N by composite score ────────────────────────────────────
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results[:top_n]