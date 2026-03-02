"""
oos_model.py
------------
Evaluates out-of-sample trading strategy for each significant pair.

Strategy:
    - Signal: rolling z-score of ticker_i residuals (60-day lookback)
    - If z > threshold → Long ticker_j at t+lag
    - If z < -threshold → Short ticker_j at t+lag
    - Hold for 1 day
    - Apply 10bps round-trip transaction cost

OOS Metric (Primary Y_ij):
    Global net Sharpe = mean(all OOS daily returns) / std(all OOS daily returns) * sqrt(252)
    Concatenated across all OOS windows.

Secondary: OOS dCor (statistical persistence check)
"""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.bq_io import (get_client, full_table, write_dataframe,
                       read_oos_strategy_returns)
from src.config_loader import get_config
from src.dcor_engine import dcor_at_lag
from src.windows import get_oos_windows

logger = logging.getLogger(__name__)


# ── Signal Generation ─────────────────────────────────────────────────────────

def compute_rolling_zscore(
    series: np.ndarray,
    lookback: int,
    threshold: float
) -> np.ndarray:
    """
    Compute rolling z-score of a series.
    Returns array of z-scores aligned to same index as series.
    First `lookback` values are NaN.
    """
    n = len(series)
    zscores = np.full(n, np.nan)
    for t in range(lookback, n):
        window = series[t - lookback:t]
        mu = np.mean(window)
        sigma = np.std(window)
        if sigma > 1e-10:
            zscores[t] = (series[t] - mu) / sigma
        # else: zscore stays NaN (flat period)
    return zscores


def compute_strategy_returns(
    x_residuals: pd.Series,  # leader (ticker_i)
    y_residuals: pd.Series,  # follower (ticker_j)
    lag: int,
    oos_start: date,
    oos_end: date,
) -> pd.DataFrame:
    """
    Compute daily strategy returns for pair (i → j) at a given lag.

    Parameters
    ----------
    x_residuals : Series indexed by date, leader residuals (pre-computed for full window)
    y_residuals : Series indexed by date, follower residuals
    lag : int, lead lag for direction
    oos_start, oos_end : OOS evaluation window

    Returns
    -------
    DataFrame with: [date, signal, position, strategy_return_gross, strategy_return_net]
    """
    cfg = get_config()["strategy"]
    lookback = cfg["zscore_lookback_days"]
    threshold = cfg["zscore_threshold"]
    tc_bps = cfg["transaction_cost_bps"]
    tc_decimal = tc_bps / 10000.0

    # Align series on common dates
    common_idx = x_residuals.index.intersection(y_residuals.index)
    x = x_residuals.loc[common_idx].sort_index()
    y = y_residuals.loc[common_idx].sort_index()

    if len(x) < lookback + lag + 10:
        return pd.DataFrame()

    dates = np.array(x.index)
    x_arr = x.values
    y_arr = y.values
    n = len(dates)

    # Compute rolling z-score on x (the leader)
    zscores = compute_rolling_zscore(x_arr, lookback, threshold)

    # Generate positions using lag:
    # Position at t = f(z_{t-lag}) applied to y_t
    records = []
    for t in range(lag, n):
        sig_t = zscores[t - lag]
        if np.isnan(sig_t):
            continue
        current_date = dates[t]
        if not (oos_start <= current_date <= oos_end):
            continue

        # Position
        if sig_t > threshold:
            position = 1   # Long y
        elif sig_t < -threshold:
            position = -1  # Short y
        else:
            position = 0   # No trade

        # Strategy return = position * y_t return
        y_return = y_arr[t]
        gross_return = position * y_return

        # Transaction cost: paid when position changes
        # Simplified: always pay if position != 0 (1-day hold means every active day costs)
        cost = tc_decimal * abs(position)
        net_return = gross_return - cost

        records.append({
            "oos_date": current_date,
            "signal": float(sig_t),
            "position": int(position),
            "strategy_return_gross": float(gross_return),
            "strategy_return_net": float(net_return),
        })

    return pd.DataFrame(records)


# ── Sharpe Computation ────────────────────────────────────────────────────────

def compute_sharpe(returns: np.ndarray, annualize: bool = True) -> Optional[float]:
    """
    Compute Sharpe ratio from daily return series.
    Assumes zero risk-free rate.
    """
    cfg = get_config()["oos"]
    ann_factor = cfg["annualization_factor"]

    if len(returns) < 10:
        return None
    mu = np.mean(returns)
    sigma = np.std(returns)
    if sigma < 1e-10:
        return None
    sharpe = mu / sigma
    if annualize:
        sharpe *= np.sqrt(ann_factor)
    return float(sharpe)


# ── OOS dCor (Secondary Metric) ───────────────────────────────────────────────

def compute_oos_dcor(x_residuals: np.ndarray, y_residuals: np.ndarray, lag: int) -> Optional[float]:
    """Compute OOS dCor at best lag as secondary diagnostic."""
    return dcor_at_lag(x_residuals, y_residuals, lag)


# ── Full OOS Evaluation Pipeline ──────────────────────────────────────────────

def run_oos_evaluation_for_window(
    window_start: date,
    window_end: date,
    oos_start: date,
    oos_end: date,
    significant_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """
    For all significant pairs in a training window, compute OOS strategy returns
    in the subsequent OOS window.

    Parameters
    ----------
    window_start, window_end : training window boundaries
    oos_start, oos_end : OOS evaluation window
    significant_pairs : DataFrame with ticker_i, ticker_j, lag for sig pairs

    Returns
    -------
    DataFrame of daily OOS returns per pair for storage in oos_strategy_returns.
    """
    if significant_pairs.empty:
        return pd.DataFrame()

    client = get_client()
    all_tickers = list(set(
        significant_pairs["ticker_i"].tolist() +
        significant_pairs["ticker_j"].tolist()
    ))

    # Pull residuals for training window (for z-score lookback) + OOS window
    # We need enough pre-OOS data for the z-score lookback
    cfg = get_config()["strategy"]
    lookback = cfg["zscore_lookback_days"]
    extended_start = oos_start - timedelta(days=lookback * 2)

    query = f"""
        SELECT date, ticker, residual
        FROM `{full_table('rolling_residuals')}`
        WHERE window_start = '{window_start}'
          AND date >= '{extended_start}'
          AND date <= '{oos_end}'
          AND ticker IN ({', '.join(f"'{t}'" for t in all_tickers)})
        ORDER BY ticker, date
    """
    resid_df = client.query(query).to_dataframe()
    resid_df["date"] = pd.to_datetime(resid_df["date"]).dt.date

    if resid_df.empty:
        logger.warning(f"No residuals found for OOS window {oos_start}→{oos_end}")
        return pd.DataFrame()

    # Pivot to ticker→Series
    resid_pivot = resid_df.pivot(index="date", columns="ticker", values="residual")

    all_records = []
    for _, row in significant_pairs.iterrows():
        ti, tj, lag = row["ticker_i"], row["ticker_j"], int(row["lag"])

        if ti not in resid_pivot.columns or tj not in resid_pivot.columns:
            continue

        x_series = resid_pivot[ti].dropna()
        y_series = resid_pivot[tj].dropna()

        oos_returns = compute_strategy_returns(
            x_series, y_series, lag, oos_start, oos_end
        )

        if oos_returns.empty:
            continue

        oos_returns["ticker_i"] = ti
        oos_returns["ticker_j"] = tj
        oos_returns["lag"] = lag
        oos_returns["window_start"] = window_start
        all_records.append(oos_returns)

    if not all_records:
        return pd.DataFrame()

    result = pd.concat(all_records, ignore_index=True)
    logger.info(
        f"OOS evaluation {oos_start}→{oos_end}: "
        f"{len(result):,} daily return records across "
        f"{result[['ticker_i','ticker_j']].drop_duplicates().shape[0]:,} pairs"
    )
    return result


def compute_global_oos_sharpe() -> pd.DataFrame:
    """
    For each pair, concatenate all OOS daily returns across windows,
    then compute the global net Sharpe. This is Y_ij for the regression.

    Returns
    -------
    DataFrame with: ticker_i, ticker_j, oos_sharpe_net, oos_sharpe_gross,
                    n_oos_days, best_lag
    """
    logger.info("Computing global OOS Sharpe per pair...")
    all_returns = read_oos_strategy_returns()

    if all_returns.empty:
        logger.warning("No OOS strategy returns found.")
        return pd.DataFrame()

    results = []
    for (ti, tj), group in all_returns.groupby(["ticker_i", "ticker_j"]):
        group = group.sort_values("oos_date")

        # Concatenate all OOS days across windows
        net_returns = group["strategy_return_net"].values
        gross_returns = group["strategy_return_gross"].values
        n_days = len(net_returns)

        # Only compute if enough data
        if n_days < 30:
            continue

        sharpe_net = compute_sharpe(net_returns)
        sharpe_gross = compute_sharpe(gross_returns)
        best_lag = int(group["lag"].mode().iloc[0])

        results.append({
            "ticker_i": ti,
            "ticker_j": tj,
            "oos_sharpe_net": sharpe_net,
            "oos_sharpe_gross": sharpe_gross,
            "n_oos_days": n_days,
            "best_lag": best_lag,
        })

    df = pd.DataFrame(results)
    logger.info(f"Global OOS Sharpe computed for {len(df):,} pairs")
    return df
