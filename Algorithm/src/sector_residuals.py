"""
sector_residuals.py
-------------------
Sector-augmented FF residualization for within-sector pair analysis.

Extends the standard Fama-French OLS regression with an equal-weighted
sector return factor, so that sector-level common movement is stripped out
before the dCor test runs.  Without this, within-sector pairs show elevated
dCor driven by the shared sector component (e.g. both Energy stocks moving
with crude oil) rather than a genuine pairwise lead-lag relationship.

Regression model (3-factor + sector):
    r_t − rf_t = α + β_mkt·MKT-RF + β_smb·SMB + β_hml·HML
               + β_sec·SECTOR_t + ε_t

SECTOR_t is the equal-weighted mean excess return of all OTHER tickers
in the same sector on day t (leave-one-out: excludes the target ticker to
avoid self-contamination).  For tickers with no sector metadata or whose
sector has only one member, the regression falls back to the standard
3-factor model.

The 5-factor variant adds RMW and CMA before the sector term:
    r_t − rf_t = α + β_mkt·MKT-RF + β_smb·SMB + β_hml·HML
               + β_rmw·RMW + β_cma·CMA + β_sec·SECTOR_t + ε_t

Output format is identical to rolling_residuals so all downstream
pipeline stages (pair_job, OOS, finalize) are unaffected.
"""

import logging
from datetime import date
from typing import List

import numpy as np
import pandas as pd

from Algorithm.src.bq_io import get_client, full_table, read_daily_prices, read_ff_factors, write_residuals
from Algorithm.src.config_loader import get_config

logger = logging.getLogger(__name__)


def _fetch_sector_map(tickers: List[str]) -> dict:
    """Return {ticker: sector} for the given tickers from ticker_metadata."""
    client = get_client()
    ticker_list = ", ".join(f"'{t}'" for t in tickers)
    df = client.query(f"""
        SELECT ticker, sector
        FROM `{full_table('ticker_metadata')}`
        WHERE ticker IN ({ticker_list})
          AND sector IS NOT NULL
          AND TRIM(sector) != ''
    """).to_dataframe()
    return dict(zip(df["ticker"], df["sector"]))


def compute_sector_residuals_for_window(
    window_start: date,
    window_end: date,
    tickers: List[str],
    factor_model: str = "3factor",
    run_id: str = "manual",
) -> pd.DataFrame:
    """
    Compute FF residuals augmented with a sector factor for each ticker.

    For each ticker the sector factor is the equal-weighted mean excess return
    of all OTHER tickers in the same sector (leave-one-out).  Tickers with no
    sector metadata or a singleton sector use standard FF factors only.

    Parameters
    ----------
    window_start, window_end : rolling window boundaries
    tickers : valid ticker list (from get_valid_tickers())
    factor_model : "3factor" | "5factor"
    run_id : pipeline run identifier

    Returns
    -------
    DataFrame with columns:
        run_id, window_start, window_end, ticker, date, residual, factor_model
    """
    logger.info(
        f"Computing sector-augmented residuals for window "
        f"{window_start} → {window_end} (factor_model={factor_model})"
    )

    prices  = read_daily_prices(tickers, window_start, window_end)
    factors = read_ff_factors(window_start, window_end)

    if prices.empty or factors.empty:
        logger.warning(f"No data for window {window_start}→{window_end}")
        return pd.DataFrame()

    prices["date"]  = pd.to_datetime(prices["date"]).dt.date
    factors["date"] = pd.to_datetime(factors["date"]).dt.date

    prices = prices.merge(factors, on="date", how="inner")
    prices = prices.dropna(subset=["log_return"])
    prices["excess_return"] = prices["log_return"] - prices["rf"]

    factor_cols = (
        ["mkt_rf", "smb", "hml", "rmw", "cma"]
        if factor_model == "5factor"
        else ["mkt_rf", "smb", "hml"]
    )

    # ── Build sector map and per-day sector sums ──────────────────────────────
    sector_map = _fetch_sector_map(tickers)
    prices["sector"] = prices["ticker"].map(sector_map)

    # Daily sector totals: sum of excess returns + count per (date, sector)
    # Used to compute leave-one-out mean efficiently.
    sector_daily = (
        prices.dropna(subset=["sector"])
        .groupby(["date", "sector"])["excess_return"]
        .agg(sector_sum="sum", sector_n="count")
        .reset_index()
    )
    prices = prices.merge(sector_daily, on=["date", "sector"], how="left")

    residuals_list = []

    for ticker, group in prices.groupby("ticker"):
        group = group.sort_values("date").dropna(subset=factor_cols + ["excess_return"])

        if len(group) < 60:
            logger.debug(f"Skipping {ticker}: insufficient observations ({len(group)})")
            continue

        X_base = group[factor_cols].values
        y      = group["excess_return"].values
        dates  = group["date"].values

        # ── Build sector factor column (leave-one-out) ────────────────────
        sector = sector_map.get(ticker)
        use_sector_factor = False

        if sector is not None and "sector_sum" in group.columns:
            n   = group["sector_n"].values          # tickers in sector on each day
            s   = group["sector_sum"].values         # sum of excess returns in sector
            loo = np.where(
                n > 1,
                (s - y) / (n - 1),                  # leave-one-out mean
                np.nan,
            )
            valid_loo = ~np.isnan(loo)
            if valid_loo.sum() >= 0.9 * len(y):     # ≥90% of days have sector peers
                use_sector_factor = True

        if use_sector_factor:
            # Fill any remaining NaN days with 0 (no sector signal)
            loo = np.where(np.isnan(loo), 0.0, loo)
            X = np.column_stack([np.ones(len(X_base)), X_base, loo])
            used_model = f"{factor_model}+sector"
        else:
            X = np.column_stack([np.ones(len(X_base)), X_base])
            used_model = factor_model
            if sector is None:
                logger.debug(f"{ticker}: no sector metadata — using {factor_model} only")

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ coeffs
        except np.linalg.LinAlgError:
            logger.warning(f"OLS failed for {ticker}, skipping.")
            continue

        residuals_list.append(pd.DataFrame({
            "run_id":       run_id,
            "window_start": window_start,
            "window_end":   window_end,
            "ticker":       ticker,
            "date":         dates,
            "residual":     resid,
            "factor_model": used_model,
        }))

    if not residuals_list:
        return pd.DataFrame()

    result = pd.concat(residuals_list, ignore_index=True)
    logger.info(
        f"Sector residuals computed: {len(result):,} rows across "
        f"{result['ticker'].nunique()} tickers"
    )
    return result