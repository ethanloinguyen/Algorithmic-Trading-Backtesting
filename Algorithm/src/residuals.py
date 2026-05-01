"""
residuals.py
------------
1. Downloads Fama-French factors from Kenneth French data library.
2. Ingests into BigQuery ff_factors table.
3. Computes rolling OLS residuals for each ticker in the universe.
4. Stores residuals in rolling_residuals table.
"""

import io
import logging
import os   # Added as of 3/1
import zipfile
from datetime import date, datetime, timedelta # Add datetime 3/1
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
from scipy import stats

from Algorithm.src.bq_io import (get_client, full_table, read_ff_factors,
                       read_daily_prices, write_dataframe, write_residuals)
from Algorithm.src.config_loader import get_config
from Algorithm.src.universe import get_valid_tickers
from Algorithm.src.windows import generate_rolling_windows

logger = logging.getLogger(__name__)

# Kenneth French data library URLs
FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"


# ── Factor Download ─────────────────────────────────────────────────────────

def download_ff_factors(factor_model: str = "3factor") -> pd.DataFrame:
    """
    Download Fama-French factors from Kenneth French's website.
    Returns daily factor DataFrame.
    """
    url = FF5_URL if factor_model == "5factor" else FF3_URL
    logger.info(f"Downloading {factor_model} factors from French library...")

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(csv_name) as f:
            raw = f.read().decode("utf-8")

    # Parse: French CSV has a header section, then data, then annual section
    lines = raw.split("\n")
    # Find where numeric data starts (after header rows)
    # Find where numeric data starts — must be 8-digit date like 20100103
    data_start = 0
    for i, line in enumerate(lines):
        parts = line.strip().split(",")
        first = parts[0].strip()
        if (len(parts) >= 4 
                and first.isdigit() 
                and len(first) == 8
                and first.startswith(("19", "20"))):
            data_start = i
            break

    # Find where annual section starts (detect year-only dates)
    data_end = len(lines)
    for i in range(data_start + 1, len(lines)):
        parts = lines[i].strip().split(",")
        if len(parts) >= 4 and parts[0].strip().isdigit() and len(parts[0].strip()) == 4:
            data_end = i
            break

    data_lines = lines[data_start:data_end]
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None)

    if factor_model == "5factor":
        df.columns = ["date_str", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    else:
        df.columns = ["date_str", "mkt_rf", "smb", "hml", "rf"]
        df["rmw"] = np.nan
        df["cma"] = np.nan

    df["date"] = pd.to_datetime(df["date_str"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]]

    # French data is in percentage points — convert to decimal
    for col in ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]:
        df[col] = df[col] / 100.0

    df["date"] = df["date"].dt.date
    logger.info(f"Downloaded {len(df)} daily factor observations.")
    return df


def ingest_ff_factors(factor_model: str = "3factor") -> None:
    """Download and store FF factors to BigQuery (full replace)."""
    df = download_ff_factors(factor_model)
    write_dataframe(df, "ff_factors", write_disposition="WRITE_TRUNCATE")
    logger.info("FF factors written to BigQuery.")


# ── Residualization ─────────────────────────────────────────────────────────

def compute_residuals_for_window(
    window_start: date,
    window_end: date,
    tickers: List[str],
    factor_model: str = "3factor",
    run_id: str = "manual" # Added as of 3/1
) -> pd.DataFrame:
    """
    For each ticker in `tickers`, run rolling OLS regression of
    log returns on FF factors, extract residuals.

    Regression model:
        r_t - rf_t = alpha + beta_mkt * MKT-RF + beta_smb * SMB
                   + beta_hml * HML [+ beta_rmw * RMW + beta_cma * CMA] + eps_t

    Parameters
    ----------
    window_start, window_end : date
        Rolling window boundaries.
    tickers : list of str
    factor_model : "3factor" or "5factor"

    Returns
    -------
    DataFrame with columns: [window_start, window_end, ticker, date, residual, factor_model]
    """
    logger.info(f"Computing residuals for window {window_start} → {window_end}")

    # Pull prices and factors
    prices = read_daily_prices(tickers, window_start, window_end)
    factors = read_ff_factors(window_start, window_end)

    if prices.empty or factors.empty:
        logger.warning(f"No data for window {window_start}→{window_end}")
        return pd.DataFrame()

    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    factors["date"] = pd.to_datetime(factors["date"]).dt.date

    # Merge factors into prices
    prices = prices.merge(factors, on="date", how="inner")
    logger.info(f"DEBUG: Merged data has {len(prices)} rows after joining prices and factors.") #debug added 3/1
    prices = prices.dropna(subset=["log_return"])
    prices["excess_return"] = prices["log_return"] - prices["rf"]

    # Factor columns for regression
    if factor_model == "5factor":
        factor_cols = ["mkt_rf", "smb", "hml", "rmw", "cma"]
    else:
        factor_cols = ["mkt_rf", "smb", "hml"]

    residuals_list = []

    for ticker, group in prices.groupby("ticker"):
        group = group.sort_values("date").dropna(subset=factor_cols + ["excess_return"])

        if len(group) < 60:  # Need minimum observations for meaningful regression
            logger.debug(f"Skipping {ticker}: insufficient observations ({len(group)})")
            continue

        X = group[factor_cols].values
        y = group["excess_return"].values
        dates = group["date"].values

        try:
            # OLS via numpy (faster than statsmodels for bulk)
            X_with_const = np.column_stack([np.ones(len(X)), X])
            coeffs, residuals, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
            resid = y - X_with_const @ coeffs
        except np.linalg.LinAlgError:
            logger.warning(f"OLS failed for {ticker}, skipping.")
            continue

        ticker_df = pd.DataFrame({
            "run_id": run_id, # Added as of 3/1
            "window_start": window_start,
            "window_end": window_end,
            "ticker": ticker,
            "date": dates,
            "residual": resid,
            "factor_model": factor_model,
        })
        residuals_list.append(ticker_df)

    if not residuals_list:
        return pd.DataFrame()

    result = pd.concat(residuals_list, ignore_index=True)
    logger.info(f"Residuals computed: {len(result):,} rows across {result['ticker'].nunique()} tickers")
    return result


def run_residuals_pipeline(
    windows: Optional[List] = None,
    only_latest: bool = False
) -> None:
    """
    Compute and store residuals for all (or just the latest) rolling window.

    Parameters
    ----------
    windows : list of (start, end) tuples, or None to generate from config
    only_latest : if True, only compute the most recent window (monthly update mode)
    """
    cfg = get_config()
    factor_model = cfg["residuals"]["factor_model"]
    tickers = get_valid_tickers()

    # Get the run_id from environment (set by Cloud Run/Workflows) 
    # or default to a timestamp if running locally
    run_id = os.environ.get("RUN_ID", f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if windows is None:
        from Algorithm.src.windows import generate_rolling_windows
        windows = generate_rolling_windows()

    if only_latest:
        windows = [windows[-1]]
        logger.info(f"Monthly mode: computing residuals for latest window only: {windows[0]}")

    for window_start, window_end in windows:
        logger.info(f"Processing window: {window_start} → {window_end}")
        df = compute_residuals_for_window(window_start, window_end, tickers, factor_model, run_id)
        if not df.empty:
            write_residuals(df, window_start)
        else:
            logger.warning(f"No residuals produced for window {window_start}")
            
if __name__ == "__main__":
    import sys
    from Algorithm.src.config_loader import load_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    load_config()

    only_latest = os.environ.get("ONLY_LATEST", "false").lower() == "true"

    logger.info(
        f"residuals.py starting — "
        f"WINDOW_START={os.environ.get('WINDOW_START')}, "
        f"WINDOW_END={os.environ.get('WINDOW_END')}, "
        f"ONLY_LATEST={only_latest}, "
        f"RUN_ID={os.environ.get('RUN_ID')}"
    )

    run_residuals_pipeline(only_latest=only_latest)
