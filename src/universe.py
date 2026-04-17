"""
universe.py
-----------
Quality filter for the equity universe.

Applies configurable thresholds to produce a list of tickers
with sufficient data quality for reliable dCor computation.

Filters:
    1. Minimum continuous years of data
    2. Minimum trading day coverage (fraction of expected days present)
    3. Minimum median daily dollar volume ($5M default)
    4. Minimum market cap ($300M default)

Results are written to the filtered_universe BigQuery table
and cached in memory for the lifetime of the process.
"""

import logging
from datetime import date
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.bq_io import get_client, full_table, write_dataframe
from src.config_loader import get_config

logger = logging.getLogger(__name__)

# Module-level cache — populated by get_valid_tickers()
_valid_tickers_cache: Optional[List[str]] = None


def run_universe_filter(
    start_date: date = None,
    end_date: date = None,
) -> pd.DataFrame:
    """
    Apply quality filters to the full ticker universe.
    Reads from market_data and ticker_metadata in BigQuery.

    Parameters
    ----------
    start_date : filter start date (defaults to config universe.start_date)
    end_date   : filter end date (defaults to config universe.end_date)

    Returns
    -------
    DataFrame with columns: [ticker, is_valid, reason, n_days,
                              coverage, median_dollar_volume, market_cap]
    Written to filtered_universe table.
    """
    cfg = get_config()["universe"]
    client = get_client()

    if start_date is None:
        start_date = date.fromisoformat(cfg["start_date"])
    if end_date is None:
        end_date = date.fromisoformat(cfg["end_date"])

    min_years = cfg["min_continuous_years"]
    min_coverage = cfg["min_trading_day_coverage"]
    min_dollar_vol = cfg["min_median_dollar_volume"]
    min_mkt_cap = cfg["min_market_cap"]

    logger.info(
        f"Running universe filter: {start_date} → {end_date}, "
        f"min_years={min_years}, min_coverage={min_coverage:.0%}, "
        f"min_dollar_vol=${min_dollar_vol:,.0f}, min_mkt_cap=${min_mkt_cap:,.0f}"
    )

    # ── Pull trading day stats per ticker ─────────────────────────────────
    # Expected trading days: approximately 252 per year
    total_calendar_days = (end_date - start_date).days
    expected_trading_days = int(total_calendar_days * 252 / 365)
    min_trading_days = int(expected_trading_days * min_coverage)
    min_obs_absolute = int(min_years * 252)


    market_data_table = full_table("market_data")
    ticker_meta_table = full_table("ticker_metadata")

    query = f"""
    WITH daily_stats AS (
        SELECT
            ticker,
            COUNT(*) AS n_days,
            MIN(DATE(date)) AS first_date,
            MAX(DATE(date)) AS last_date,
            APPROX_QUANTILES(close * volume, 2)[OFFSET(1)] AS median_dollar_volume
        FROM `{market_data_table}`
        WHERE DATE(date) >= '{start_date}'
          AND DATE(date) <= '{end_date}'
          AND close > 0
          AND volume > 0
        GROUP BY ticker
    ),
        meta AS (
            SELECT ticker, market_cap
            FROM `{ticker_meta_table}`
        )
        SELECT
            d.ticker,
            d.n_days,
            d.first_date,
            d.last_date,
            d.median_dollar_volume,
            COALESCE(m.market_cap, 0) AS market_cap,
            SAFE_DIVIDE(d.n_days, {expected_trading_days}) AS coverage
        FROM daily_stats d
        LEFT JOIN meta m ON d.ticker = m.ticker
        ORDER BY d.ticker
    """

    logger.info("Querying market data stats from BigQuery...")
    df = client.query(query).to_dataframe()

    if df.empty:
        logger.error("No market data found. Check market_data table and date range.")
        return pd.DataFrame()

    logger.info(f"Found {len(df):,} tickers with market data")

    # ── Apply filters ─────────────────────────────────────────────────────
    df["is_valid"] = True
    df["reason"] = "OK"

    # Filter 1: minimum trading days / coverage
    coverage_fail = df["n_days"] < min_obs_absolute
    df.loc[coverage_fail & df["is_valid"], "reason"] = (
        f"Insufficient trading days (< {min_obs_absolute})"
    )
    df.loc[coverage_fail, "is_valid"] = False

    # Filter 2: coverage fraction
    cov_fail = df["coverage"] < min_coverage
    df.loc[cov_fail & df["is_valid"], "reason"] = (
        f"Coverage below {min_coverage:.0%}"
    )
    df.loc[cov_fail, "is_valid"] = False

    # Filter 3: median dollar volume
    vol_fail = df["median_dollar_volume"] < min_dollar_vol
    df.loc[vol_fail & df["is_valid"], "reason"] = (
        f"Median dollar volume below ${min_dollar_vol:,.0f}"
    )
    df.loc[vol_fail, "is_valid"] = False

    # Filter 4: market cap
    cap_fail = df["market_cap"] < min_mkt_cap
    df.loc[cap_fail & df["is_valid"], "reason"] = (
        f"Market cap below ${min_mkt_cap:,.0f}"
    )
    df.loc[cap_fail, "is_valid"] = False

    n_valid = df["is_valid"].sum()
    n_total = len(df)
    logger.info(
        f"Universe filter complete: {n_valid:,} / {n_total:,} tickers pass "
        f"({n_valid/n_total:.1%})"
    )

    # Log rejection breakdown
    if n_valid < n_total:
        rejected = df[~df["is_valid"]]
        reason_counts = rejected["reason"].value_counts()
        for reason, count in reason_counts.items():
            logger.info(f"  Rejected ({count:,}): {reason}")

    # Write results
    df["filter_date"] = date.today()
    write_dataframe(df, "filtered_universe", write_disposition="WRITE_TRUNCATE")

    # Invalidate cache
    global _valid_tickers_cache
    _valid_tickers_cache = None

    return df


def get_valid_tickers(force_refresh: bool = False) -> List[str]:
    """
    Return list of valid ticker symbols from the filtered universe.
    Uses in-memory cache after first call.

    Parameters
    ----------
    force_refresh : if True, re-query BigQuery even if cache exists

    Returns
    -------
    List of ticker strings, sorted alphabetically.
    """
    global _valid_tickers_cache

    # 1. Check Memory Cache
    if _valid_tickers_cache is not None and not force_refresh:
        return _valid_tickers_cache

    client = get_client()
    table_id = full_table('filtered_universe')

    # 2. Try to load from BigQuery
    try:
        query = f"SELECT ticker FROM `{table_id}` WHERE is_valid = TRUE ORDER BY ticker"
        df = client.query(query).to_dataframe()
    except Exception:
        # If the table doesn't even exist yet, treat it as empty
        df = pd.DataFrame()

    # 3. THE "FAIL-SAFE": If no tickers found, run the filter automatically
    if df.empty:
        logger.warning("No valid tickers found in filtered_universe. Running filter now...")
        # This populates the table so the next run has data
        run_universe_filter() 
        
        # Re-query the table we just populated
        df = client.query(query).to_dataframe()

    if df.empty:
        logger.error("Universe filter returned 0 valid tickers. Check your data and config thresholds!")
        return []

    _valid_tickers_cache = df["ticker"].tolist()
    logger.info(f"Loaded {len(_valid_tickers_cache):,} valid tickers")
    return _valid_tickers_cache


def get_all_pairs(tickers: List[str]) -> List[Tuple[str, str]]:
    """
    Generate all unique (ticker_i, ticker_j) pairs from a list of tickers.
    ticker_i < ticker_j alphabetically to ensure canonical ordering.

    Parameters
    ----------
    tickers : list of ticker strings

    Returns
    -------
    List of (ticker_i, ticker_j) tuples — all C(n, 2) combinations.
    """
    tickers_sorted = sorted(tickers)
    pairs = list(combinations(tickers_sorted, 2))
    logger.info(f"Generated {len(pairs):,} pairs from {len(tickers):,} tickers")
    return pairs


def get_sector_pairs(tickers: List[str]) -> List[Tuple[str, str]]:
    """
    Generate all unique (ticker_i, ticker_j) pairs where both tickers
    share the same GICS sector, based on ticker_metadata.

    Tickers with no sector metadata are logged and excluded — they cannot
    be assigned to any sector group.

    Parameters
    ----------
    tickers : list of valid ticker strings (from get_valid_tickers())

    Returns
    -------
    List of (ticker_i, ticker_j) tuples — all within-sector C(n_sector, 2)
    combinations across all sectors, ticker_i < ticker_j alphabetically.
    """
    from collections import defaultdict

    client = get_client()
    tickers_sorted = sorted(tickers)
    ticker_list = ", ".join(f"'{t}'" for t in tickers_sorted)

    df = client.query(f"""
        SELECT ticker, sector
        FROM `{full_table('ticker_metadata')}`
        WHERE ticker IN ({ticker_list})
          AND sector IS NOT NULL
          AND TRIM(sector) != ''
    """).to_dataframe()

    sector_map = dict(zip(df["ticker"], df["sector"]))

    sector_groups: dict = defaultdict(list)
    no_sector = []
    for ticker in tickers_sorted:
        sector = sector_map.get(ticker)
        if sector:
            sector_groups[sector].append(ticker)
        else:
            no_sector.append(ticker)

    if no_sector:
        logger.warning(
            f"  {len(no_sector)} tickers have no sector metadata and will be "
            f"excluded from sector pairs: {no_sector[:10]}"
            + (" ..." if len(no_sector) > 10 else "")
        )

    pairs = []
    for sector, members in sorted(sector_groups.items()):
        sector_pairs = list(combinations(members, 2))
        logger.info(
            f"  Sector '{sector}': {len(members)} tickers "
            f"→ {len(sector_pairs):,} pairs"
        )
        pairs.extend(sector_pairs)

    logger.info(
        f"Generated {len(pairs):,} in-sector pairs from "
        f"{len(tickers):,} tickers across {len(sector_groups)} sectors"
    )
    return pairs


def partition_pairs(
    pairs: List[Tuple[str, str]],
    partition_id: int,
    num_partitions: int,
) -> List[Tuple[str, str]]:
    """
    Return the subset of pairs assigned to this partition worker.

    Uses modulo assignment: pair i → partition (i % num_partitions).
    This distributes pairs evenly without requiring coordination between workers.

    Parameters
    ----------
    pairs : full list of all pairs
    partition_id : 0-indexed partition number for this worker
    num_partitions : total number of parallel workers

    Returns
    -------
    List of (ticker_i, ticker_j) tuples for this partition.
    """
    if partition_id >= num_partitions:
        raise ValueError(
            f"partition_id ({partition_id}) must be < num_partitions ({num_partitions})"
        )

    my_pairs = [
        pair for i, pair in enumerate(pairs)
        if i % num_partitions == partition_id
    ]

    logger.info(
        f"Partition {partition_id}/{num_partitions}: "
        f"{len(my_pairs):,} / {len(pairs):,} pairs assigned"
    )
    return my_pairs


def estimate_pair_count(n_tickers: int) -> int:
    """Return the number of unique pairs for n tickers: C(n, 2) = n*(n-1)/2."""
    return n_tickers * (n_tickers - 1) // 2