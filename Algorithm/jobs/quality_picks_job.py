"""
quality_picks_job.py
--------------------
Daily Precomputation Job: Quality Picks Factor Scores

Runs nightly after the OHLCV / metadata ingest completes. Computes three
portfolio-independent factor scores for every stock in the filtered universe
and writes them to the quality_picks_scores BigQuery table (WRITE_TRUNCATE).

Scores computed here (0–100, rank-normalised cross-sectionally):
    momentum_score            — 6-month price return vs. filtered universe peers
    fundamental_quality_score — P/E vs. within-sector median + log market cap
    centrality_score          — eigenvector centrality from latest final_network

The two portfolio-specific factors (sector_diversity, volatility_compatibility)
are computed on-the-fly by the backend when a user submits their holdings.
This table serves as the fast cache they merge against.

Usage
-----
    # Write scores to BigQuery:
    python -m Algorithm.jobs.quality_picks_job

    # Validate locally without writing:
    python -m Algorithm.jobs.quality_picks_job --dry-run
"""

import argparse
import logging
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ── Path bootstrap — mirrors aggregation_job.py ───────────────────────────────
_job_dir  = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_job_dir)
for _p in (_proj_root, "/app"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Algorithm.src.bq_io import get_client, full_table, write_dataframe
from Algorithm.src.config_loader import load_config, get_config
from Algorithm.src.universe import get_valid_tickers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Tunable constants ─────────────────────────────────────────────────────────
MOMENTUM_LOOKBACK_DAYS = 180   # calendar days (~6 months of trading days)
PE_CLIP_MAX            = 100.0 # P/E above this → treated as overpriced ceiling
PE_CLIP_MIN            = 0.0   # P/E ≤ 0 (loss-making) → treated as null
PE_WEIGHT              = 0.50  # weight of P/E component within fundamental score
MC_WEIGHT              = 0.50  # weight of market-cap component within fundamental score


# ── Data Fetchers ─────────────────────────────────────────────────────────────

def _fetch_metadata(tickers: list) -> pd.DataFrame:
    """
    Single bulk query: fetch sector, market_cap, pe_ratio for all tickers
    in the filtered universe from ticker_metadata.

    Returns
    -------
    DataFrame [ticker, sector, market_cap, pe_ratio]
    pe_ratio may be null for loss-making or pre-earnings stocks.
    """
    client    = get_client()
    table_id  = full_table("ticker_metadata")
    tlist     = ", ".join(f"'{t}'" for t in tickers)

    query = f"""
        SELECT
            ticker,
            sector,
            market_cap,
            pe_ratio
        FROM `{table_id}`
        WHERE ticker IN ({tlist})
    """
    df = client.query(query).to_dataframe()
    logger.info(f"Metadata: {len(df):,} rows fetched "
                f"({df['pe_ratio'].notna().sum():,} with pe_ratio, "
                f"{df['market_cap'].notna().sum():,} with market_cap)")
    return df


def _fetch_momentum_prices(tickers: list, today: date) -> pd.DataFrame:
    """
    For each ticker, retrieve:
        latest_close  — most recent available closing price
        start_close   — closest trading-day close to MOMENTUM_LOOKBACK_DAYS ago

    Uses a ±7 calendar-day window around the target start date to handle
    weekends and market holidays. Tickers with no close within that window
    are excluded (not enough history) and will appear as null in the
    output with a warning logged.

    Returns
    -------
    DataFrame [ticker, latest_close, start_close]
    """
    client       = get_client()
    table_id     = full_table("market_data")
    tlist        = ", ".join(f"'{t}'" for t in tickers)
    start_target = today - timedelta(days=MOMENTUM_LOOKBACK_DAYS)
    start_from   = start_target - timedelta(days=7)
    start_to     = start_target + timedelta(days=7)

    query = f"""
        WITH latest AS (
            SELECT
                ticker,
                close          AS latest_close,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM `{table_id}`
            WHERE ticker IN ({tlist})
              AND close > 0
        ),
        start_window AS (
            SELECT
                ticker,
                close          AS start_close,
                DATE(date)     AS start_date,
                ABS(DATE_DIFF(DATE(date), DATE('{start_target}'), DAY)) AS days_away,
                ROW_NUMBER() OVER (
                    PARTITION BY ticker
                    ORDER BY ABS(DATE_DIFF(DATE(date), DATE('{start_target}'), DAY))
                ) AS rn
            FROM `{table_id}`
            WHERE ticker IN ({tlist})
              AND DATE(date) BETWEEN '{start_from}' AND '{start_to}'
              AND close > 0
        )
        SELECT
            l.ticker,
            l.latest_close,
            s.start_close
        FROM latest       l
        JOIN start_window s ON l.ticker = s.ticker AND s.rn = 1
        WHERE l.rn = 1
          AND s.days_away <= 7
    """
    df = client.query(query).to_dataframe()
    logger.info(f"Momentum prices: {len(df):,} ticker pairs fetched "
                f"(out of {len(tickers):,} requested)")
    return df


def _fetch_centrality(tickers: list) -> pd.DataFrame:
    """
    Extract per-ticker eigenvector centrality from the most recent
    final_network snapshot. Both (ticker_i, centrality_i) and
    (ticker_j, centrality_j) edges are unioned; MAX is taken since
    centrality is stored symmetrically per edge.

    Tickers absent from final_network (no significant pairs) are NOT
    returned here — they will receive centrality_raw = 0.0 in the
    scoring step via a left-join fill.

    Returns
    -------
    DataFrame [ticker, centrality_raw]
    """
    client   = get_client()
    table_id = full_table("final_network")
    tlist    = ", ".join(f"'{t}'" for t in tickers)

    query = f"""
        WITH latest_date AS (
            SELECT MAX(as_of_date) AS max_date FROM `{table_id}`
        ),
        unioned AS (
            SELECT ticker_i AS ticker, centrality_i AS centrality
            FROM `{table_id}`, latest_date
            WHERE as_of_date = max_date
              AND ticker_i IN ({tlist})
              AND centrality_i IS NOT NULL

            UNION ALL

            SELECT ticker_j AS ticker, centrality_j AS centrality
            FROM `{table_id}`, latest_date
            WHERE as_of_date = max_date
              AND ticker_j IN ({tlist})
              AND centrality_j IS NOT NULL
        )
        SELECT ticker, MAX(centrality) AS centrality_raw
        FROM unioned
        GROUP BY ticker
    """
    df = client.query(query).to_dataframe()
    logger.info(f"Centrality: {len(df):,} tickers have non-null centrality "
                f"in final_network")
    return df


# ── Score Computers ───────────────────────────────────────────────────────────

def _compute_momentum_score(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank-normalise 6-month returns to a 0–100 score across the universe.

        6m_return     = (latest_close - start_close) / start_close
        momentum_score = percentile rank × 100  (higher return → higher score)

    Returns
    -------
    DataFrame [ticker, momentum_6m_return, momentum_score]
    """
    df = prices_df.copy()
    df["momentum_6m_return"] = (
        (df["latest_close"] - df["start_close"]) / df["start_close"]
    ).round(6)

    df["momentum_score"] = (
        df["momentum_6m_return"]
          .rank(method="average", ascending=True, pct=True)
          * 100
    ).round(2)

    return df[["ticker", "momentum_6m_return", "momentum_score"]]


def _compute_fundamental_quality_score(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite fundamental quality score (0–100):

        PE_WEIGHT × P/E score (within-sector rank, lower P/E → higher score)
      + MC_WEIGHT × market-cap score (log-scale rank, larger cap → higher score)

    P/E handling:
      - P/E ≤ 0 or null (loss-making / pre-earnings): treated as null
      - P/E > PE_CLIP_MAX: clipped to PE_CLIP_MAX before ranking
      - Null P/E stocks receive their sector's median P/E score (neutral)
      - Stocks with no sector tag: receive universe-median P/E score

    Returns
    -------
    DataFrame [ticker, fundamental_quality_score]
    """
    df = meta_df.copy()

    # ── Market cap score ──────────────────────────────────────────────────────
    df["log_mc"] = np.log1p(df["market_cap"].clip(lower=0).fillna(0))
    df["mc_score"] = (
        df["log_mc"]
          .rank(method="average", ascending=True, pct=True)
          * 100
    ).round(2)

    # ── P/E score ─────────────────────────────────────────────────────────────
    # Clip and null-out invalid values
    df["pe_clipped"] = df["pe_ratio"].copy()
    df.loc[df["pe_clipped"] <= PE_CLIP_MIN, "pe_clipped"] = np.nan
    df.loc[df["pe_clipped"] > PE_CLIP_MAX,  "pe_clipped"] = PE_CLIP_MAX

    df["pe_score"] = np.nan

    for sector, grp in df.groupby("sector", dropna=True):
        valid_mask = grp["pe_clipped"].notna()

        if valid_mask.sum() == 0:
            # No usable P/E in this sector — assign neutral score to everyone
            df.loc[grp.index, "pe_score"] = 50.0
            continue

        # Rank within sector: lower P/E → higher rank (ascending=False)
        ranked = (
            grp.loc[valid_mask, "pe_clipped"]
               .rank(method="average", ascending=False, pct=True)
               * 100
        ).round(2)
        df.loc[ranked.index, "pe_score"] = ranked

        # Null-P/E stocks in this sector: give them the sector median score
        null_idx = grp.index[~valid_mask]
        if len(null_idx):
            df.loc[null_idx, "pe_score"] = ranked.median()

    # Stocks with no sector tag fall back to universe median
    universe_pe_median = df["pe_score"].median()
    df["pe_score"] = df["pe_score"].fillna(universe_pe_median).round(2)

    # ── Composite ─────────────────────────────────────────────────────────────
    df["fundamental_quality_score"] = (
        PE_WEIGHT * df["pe_score"] + MC_WEIGHT * df["mc_score"]
    ).round(2)

    return df[["ticker", "fundamental_quality_score"]]


def _compute_centrality_score(
    tickers: list,
    centrality_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalise raw eigenvector centrality to 0–100 via rank percentile.
    Tickers absent from final_network receive centrality_raw = 0.0,
    placing them at the bottom of the distribution.

    Returns
    -------
    DataFrame [ticker, centrality_raw, centrality_score]
    """
    base   = pd.DataFrame({"ticker": tickers})
    merged = base.merge(centrality_df, on="ticker", how="left")
    merged["centrality_raw"] = merged["centrality_raw"].fillna(0.0)

    merged["centrality_score"] = (
        merged["centrality_raw"]
              .rank(method="average", ascending=True, pct=True)
              * 100
    ).round(2)

    return merged[["ticker", "centrality_raw", "centrality_score"]]


# ── Main Job ──────────────────────────────────────────────────────────────────

def run_quality_picks_job(dry_run: bool = False) -> pd.DataFrame:
    """
    Orchestrate the nightly Quality Picks precomputation.

    Parameters
    ----------
    dry_run : if True, compute and log results without writing to BigQuery.

    Returns
    -------
    Final scored DataFrame (returned regardless of dry_run, useful for tests).
    """
    today = date.today()
    logger.info(
        f"=== QUALITY PICKS JOB START === date={today}  dry_run={dry_run}"
    )

    # ── Step 1: Filtered universe ─────────────────────────────────────────────
    tickers = get_valid_tickers()
    if not tickers:
        logger.error("No valid tickers found in filtered_universe. Aborting.")
        return pd.DataFrame()
    logger.info(f"Filtered universe: {len(tickers):,} tickers")

    # ── Step 2: Raw data fetches ──────────────────────────────────────────────
    logger.info("Step 2a: Fetching ticker metadata...")
    meta_df = _fetch_metadata(tickers)

    logger.info("Step 2b: Fetching 6-month price pairs for momentum...")
    prices_df = _fetch_momentum_prices(tickers, today)

    logger.info("Step 2c: Fetching centrality from final_network...")
    centrality_raw_df = _fetch_centrality(tickers)

    # ── Step 3: Score computation ─────────────────────────────────────────────
    logger.info("Step 3a: Computing momentum scores...")
    momentum_df = _compute_momentum_score(prices_df)
    logger.info(
        f"  momentum_score  — n={len(momentum_df):,}  "
        f"mean={momentum_df['momentum_score'].mean():.1f}  "
        f"p10={momentum_df['momentum_score'].quantile(0.1):.1f}  "
        f"p90={momentum_df['momentum_score'].quantile(0.9):.1f}"
    )

    logger.info("Step 3b: Computing fundamental quality scores...")
    fundamental_df = _compute_fundamental_quality_score(meta_df)
    logger.info(
        f"  fundamental_quality_score — n={len(fundamental_df):,}  "
        f"mean={fundamental_df['fundamental_quality_score'].mean():.1f}"
    )

    logger.info("Step 3c: Computing centrality scores...")
    centrality_scored_df = _compute_centrality_score(tickers, centrality_raw_df)
    n_nonzero = (centrality_scored_df["centrality_raw"] > 0).sum()
    logger.info(
        f"  centrality_score — {n_nonzero:,} tickers with non-zero raw centrality "
        f"({n_nonzero / len(tickers) * 100:.1f}% of universe)"
    )

    # ── Step 4: Assemble final table ──────────────────────────────────────────
    logger.info("Step 4: Assembling final scores table...")
    base = meta_df[["ticker", "sector", "market_cap", "pe_ratio"]].copy()

    result = (
        base
        .merge(momentum_df,           on="ticker", how="left")
        .merge(fundamental_df,        on="ticker", how="left")
        .merge(centrality_scored_df,  on="ticker", how="left")
    )
    result["updated_at"] = today

    # Diagnostics
    n_missing_mom = result["momentum_score"].isna().sum()
    if n_missing_mom:
        logger.warning(
            f"  {n_missing_mom:,} tickers lack momentum_score "
            f"(no price data within 6-month window)"
        )
    logger.info(
        f"  Final: {len(result):,} rows — "
        f"{result['momentum_score'].notna().sum():,} with momentum, "
        f"{result['fundamental_quality_score'].notna().sum():,} with fundamental, "
        f"{(result['centrality_raw'] > 0).sum():,} with centrality"
    )

    # ── Step 5: Write or dry-run ──────────────────────────────────────────────
    if dry_run:
        logger.info("DRY RUN — skipping write. Top 20 by composite (simple avg):")
        preview = result.copy()
        score_cols = ["momentum_score", "fundamental_quality_score", "centrality_score"]
        preview["avg_score"] = preview[score_cols].mean(axis=1)
        top20 = (
            preview.dropna(subset=["momentum_score"])
                   .sort_values("avg_score", ascending=False)
                   .head(20)
        )
        display_cols = [
            "ticker", "sector",
            "momentum_score", "fundamental_quality_score", "centrality_score",
            "momentum_6m_return", "pe_ratio", "market_cap",
        ]
        with pd.option_context("display.max_columns", None, "display.width", 140,
                               "display.float_format", "{:.2f}".format):
            print("\n" + top20[display_cols].to_string(index=False))

        # Also print score distribution summary
        print("\n── Score distributions ──")
        for col in score_cols:
            vals = result[col].dropna()
            print(
                f"  {col:<32}  "
                f"n={len(vals):>4}  "
                f"mean={vals.mean():5.1f}  "
                f"std={vals.std():5.1f}  "
                f"p10={vals.quantile(0.1):5.1f}  "
                f"p90={vals.quantile(0.9):5.1f}"
            )
    else:
        logger.info("Step 5: Writing to BigQuery (WRITE_TRUNCATE)...")
        write_dataframe(
            result,
            "quality_picks_scores",
            write_disposition="WRITE_TRUNCATE",
        )
        logger.info(f"  ✓ Wrote {len(result):,} rows to quality_picks_scores")

    logger.info("=== QUALITY PICKS JOB COMPLETE ===")
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quality Picks nightly precomputation — computes momentum, "
                    "fundamental quality, and centrality scores for the filtered universe."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print scores without writing to BigQuery.",
    )
    args = parser.parse_args()

    load_config()
    run_quality_picks_job(dry_run=args.dry_run)
