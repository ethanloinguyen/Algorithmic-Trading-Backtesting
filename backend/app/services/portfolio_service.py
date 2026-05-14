# backend/app/services/portfolio_service.py
"""
Portfolio analysis service — reads from BigQuery final_network or sector_final_network.

Performance optimizations vs. naive SELECT *:
  1. as_of_date is cached in memory for 1 hour (per table) so the MAX subquery
     only runs once per hour instead of on every request.
  2. The main network query is split into two targeted fetches:
       a. Rows where ticker_i OR ticker_j is a user holding
          → used for overlap detection + signal recommendations
       b. Sector/centrality metadata for all tickers seen in (a)
          → used to build the ticker metadata index
     This means BigQuery returns ~hundreds of rows per request
     instead of all 242,898 rows.
  3. Independent recommendations require knowing which tickers are
     NOT connected to the user — we derive this from the metadata
     universe built in (b) rather than fetching the full table.
"""
from __future__ import annotations

import logging
import time
import pandas as pd

logger = logging.getLogger(__name__)

# ── Table name mapping ────────────────────────────────────────────────────────
_TABLE_NAMES = {
    "broad_market": "final_network",
    "in_sector":    "sector_final_network",
}

# ── In-memory cache for as_of_date, keyed per table (refreshes every hour) ───
_AS_OF_CACHE: dict[str, dict] = {}
_AS_OF_TTL_SECONDS = 3600  # 1 hour — tables update monthly


def _get_latest_as_of_date(table_name: str) -> str:
    """
    Return the most recent as_of_date for the given table as a string.
    Cached in memory for 1 hour per table to avoid a full-table scan on every request.
    """
    now = time.monotonic()
    cache = _AS_OF_CACHE.setdefault(table_name, {"date": None, "fetched_at": 0.0})
    if cache["date"] is not None and (now - cache["fetched_at"]) < _AS_OF_TTL_SECONDS:
        return cache["date"]

    from app.core.bigquery import get_bq_client
    from app.core.config import get_settings
    settings = get_settings()
    client   = get_bq_client()
    table    = f"`{settings.gcp_project_id}.{settings.bq_dataset}.{table_name}`"

    result = client.query(
        f"SELECT MAX(as_of_date) AS latest FROM {table}"
    ).to_dataframe()

    raw      = result["latest"].iloc[0]
    if pd.isna(raw):
        raise ValueError(
            f"Table '{table_name}' returned no valid as_of_date — "
            "the table may be empty or not yet populated."
        )
    date_val = str(raw)
    cache["date"] = date_val
    cache["fetched_at"] = now
    return date_val


def _get_network_for_tickers(tickers: list[str], table_name: str) -> pd.DataFrame:
    """
    Fetch only the relevant rows for the given tickers from the specified table.

    Returns two kinds of rows:
      - Rows where ticker_i OR ticker_j is in the user's holdings
        (needed for overlap analysis + signal recommendations)
      - Rows where ticker_i OR ticker_j appears alongside a user holding
        (needed to build the full candidate universe and metadata)

    BigQuery does the filtering — we never load all rows.
    """
    from google.cloud import bigquery
    from app.core.bigquery import get_bq_client
    from app.core.config import get_settings

    settings   = get_settings()
    client     = get_bq_client()
    table      = f"`{settings.gcp_project_id}.{settings.bq_dataset}.{table_name}`"
    as_of_date = _get_latest_as_of_date(table_name)

    # Build a safe ticker list for the IN clause using parameterized query
    ticker_params = [
        bigquery.ArrayQueryParameter("tickers", "STRING", [t.upper() for t in tickers])
    ]
    job_config = bigquery.QueryJobConfig(query_parameters=ticker_params)

    # Step 1: Get all pairs directly involving the user's tickers.
    # This covers:
    #   - Overlap pairs (both ticker_i and ticker_j are user holdings)
    #   - Signal recommendation candidates (one side is a user holding)
    query = f"""
        SELECT
            as_of_date,
            ticker_i,
            ticker_j,
            best_lag,
            mean_dcor,
            variance_dcor,
            frequency,
            half_life,
            sharpness,
            predicted_sharpe,
            signal_strength,
            oos_sharpe_net,
            oos_dcor,
            sector_i,
            sector_j,
            rank,
            COALESCE(centrality_i, 0.0) AS centrality_i,
            COALESCE(centrality_j, 0.0) AS centrality_j
        FROM {table}
        WHERE as_of_date = '{as_of_date}'
          AND (ticker_i IN UNNEST(@tickers) OR ticker_j IN UNNEST(@tickers))
    """

    df = client.query(query, job_config=job_config).to_dataframe()

    # Cast numeric columns
    numeric_cols = [
        "centrality_i", "centrality_j", "frequency", "half_life",
        "signal_strength", "mean_dcor", "oos_sharpe_net", "predicted_sharpe",
        "variance_dcor", "sharpness",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def _get_universe_metadata(as_of_date: str, table_name: str, limit: int = 5000) -> pd.DataFrame:
    """
    Fetch a compact metadata sample from the specified table for the independent
    recommendations universe — one row per unique ticker with their sector
    and centrality. We only need distinct tickers, not all pair rows.

    Uses a GROUP BY to collapse the full table into one row per ticker,
    which BigQuery executes efficiently as an aggregation rather than a scan.
    """
    from app.core.bigquery import get_bq_client
    from app.core.config import get_settings

    settings = get_settings()
    client   = get_bq_client()
    table    = f"`{settings.gcp_project_id}.{settings.bq_dataset}.{table_name}`"

    query = f"""
        SELECT ticker, sector, MAX(centrality) AS centrality
        FROM (
            SELECT ticker_i AS ticker, sector_i AS sector,
                   COALESCE(centrality_i, 0.0) AS centrality
            FROM {table} WHERE as_of_date = '{as_of_date}'
            UNION ALL
            SELECT ticker_j AS ticker, sector_j AS sector,
                   COALESCE(centrality_j, 0.0) AS centrality
            FROM {table} WHERE as_of_date = '{as_of_date}'
        )
        GROUP BY ticker, sector
        LIMIT {limit}
    """

    df = client.query(query).to_dataframe()
    df["centrality"] = pd.to_numeric(df["centrality"], errors="coerce").fillna(0.0)
    return df


def get_final_network(
    tickers: list[str] | None = None,
    analysis_mode: str = "broad_market",
) -> pd.DataFrame:
    """
    Return the relevant slice of the network table for the given tickers.
    analysis_mode selects the source table: "broad_market" → final_network,
    "in_sector" → sector_final_network.
    If tickers is None or empty, returns an empty DataFrame (never fetches all rows).
    """
    if not tickers:
        return pd.DataFrame()
    table_name = _TABLE_NAMES.get(analysis_mode, "final_network")
    return _get_network_for_tickers(tickers, table_name)


def run_portfolio_analysis(
    tickers: list[str],
    analysis_mode: str = "broad_market",
    top_n: int = 10,
    min_signal: float = 55.0,
) -> dict:
    from app.services.portfolio_engine import (
        _normalize_tickers,
        get_ticker_metadata,
        analyze_portfolio_overlap,
        get_signal_recommendations,
        get_independent_recommendations,
        get_holdings_sectors,
    )
    from dataclasses import asdict

    table_name = _TABLE_NAMES.get(analysis_mode, "final_network")

    from app.services.bigquery_services import get_quality_picks

    normalized = _normalize_tickers(tickers)
    if not normalized:
        return {
            "tickers_analyzed":            [],
            "unknown_tickers":             [],
            "overlaps":                    [],
            "signal_recommendations":      [],
            "independent_recommendations": [],
            "quality_picks":               [],
            "holdings_sectors":            {},
        }

    # Fetch only the rows that involve the user's tickers — fast targeted query
    df = get_final_network(normalized, analysis_mode=analysis_mode)
    meta = get_ticker_metadata(df)

    known   = [t for t in normalized if t in meta.index]
    unknown = [t for t in normalized if t not in meta.index]

    # For independent recommendations we need a broader universe of tickers
    # that have NO relationship with the user's holdings.
    # Fetch a compact metadata table (ticker, sector, centrality) for this.
    as_of_date = _get_latest_as_of_date(table_name)
    universe_meta_df = _get_universe_metadata(as_of_date, table_name)

    # in_sector data has signal_strength=50.0 (pipeline fallback) — threshold
    # would filter everything out at 55.0, so bypass it until backfill is complete.
    effective_min_signal = 0.0 if analysis_mode == "in_sector" else min_signal

    try:
        quality_picks = get_quality_picks(known if known else normalized, top_n=top_n)
    except Exception as exc:
        logger.warning("get_quality_picks failed (%s) — returning empty list", exc)
        quality_picks = []

    return {
        "tickers_analyzed":            known,
        "unknown_tickers":             unknown,
        "overlaps":                    [asdict(o) for o in analyze_portfolio_overlap(normalized, df)],
        "signal_recommendations":      [asdict(r) for r in get_signal_recommendations(normalized, df, top_n=top_n, min_signal_strength=effective_min_signal)],
        "independent_recommendations": [asdict(r) for r in get_independent_recommendations(normalized, df, top_n=top_n, universe_meta_df=universe_meta_df)],
        "quality_picks":               quality_picks,
        "holdings_sectors":            get_holdings_sectors(normalized, df),
    }