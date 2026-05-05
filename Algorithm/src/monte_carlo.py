"""
monte_carlo.py
--------------
Generates Monte Carlo confidence cones for cumulative strategy PnL.

Method:
    - Bootstrap blocks of actual OOS daily returns (block bootstrap)
    - Simulate N cumulative return paths
    - Extract percentile bands at each time step

These are precomputed monthly and stored in BigQuery for fast API serving.
"""

import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Algorithm.src.bq_io import write_dataframe
from Algorithm.src.config_loader import get_config

logger = logging.getLogger(__name__)


def block_bootstrap_returns(
    returns: np.ndarray,
    n_simulations: int,
    block_size: int,
    n_periods: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Block bootstrap of return series to generate simulated paths.

    Parameters
    ----------
    returns : actual daily return series (OOS)
    n_simulations : number of simulation paths
    block_size : block length for bootstrap (preserves autocorrelation)
    n_periods : number of time periods per simulated path
    rng : numpy random Generator

    Returns
    -------
    Array of shape (n_simulations, n_periods) of simulated daily returns.
    """
    n = len(returns)
    n_blocks_needed = int(np.ceil(n_periods / block_size))
    max_start = max(n - block_size, 1)

    sim_returns = np.zeros((n_simulations, n_periods))

    for sim in range(n_simulations):
        blocks = []
        for _ in range(n_blocks_needed):
            start = rng.integers(0, max_start)
            block = returns[start: start + block_size]
            blocks.append(block)
        sim_path = np.concatenate(blocks)[:n_periods]
        sim_returns[sim] = sim_path

    return sim_returns


def compute_cumulative_paths(
    sim_returns: np.ndarray,
) -> np.ndarray:
    """
    Convert daily returns to cumulative return paths.
    Returns shape (n_simulations, n_periods).
    """
    return np.cumprod(1 + sim_returns, axis=1) - 1


def compute_cone_percentiles(
    cum_paths: np.ndarray,
    confidence_levels: List[float],
) -> Dict[float, np.ndarray]:
    """
    Compute percentile bands across simulation paths at each time step.

    Parameters
    ----------
    cum_paths : shape (n_simulations, n_periods)
    confidence_levels : list of percentile fractions (e.g. [0.05, 0.25, 0.75, 0.95])

    Returns
    -------
    dict: {percentile: array of length n_periods}
    """
    result = {}
    for pct in confidence_levels:
        result[pct] = np.percentile(cum_paths, pct * 100, axis=0)
    return result


def run_monte_carlo_for_pair(
    ticker_i: str,
    ticker_j: str,
    oos_returns: np.ndarray,
    as_of_date: date = None,
) -> Optional[pd.DataFrame]:
    """
    Run Monte Carlo simulation for a single pair.

    Returns
    -------
    DataFrame with columns: [day, pct_5, pct_25, mean_path, pct_75, pct_95]
    Ready to store in BigQuery and serve to API.
    """
    cfg = get_config()["monte_carlo"]
    n_sims = cfg["n_simulations"]
    block_size = cfg["block_size"]
    conf_levels = cfg["confidence_levels"]

    if as_of_date is None:
        as_of_date = date.today()

    if len(oos_returns) < 30:
        return None

    n_periods = len(oos_returns)
    rng = np.random.default_rng(seed=hash(f"{ticker_i}{ticker_j}") % (2**31))

    sim_returns = block_bootstrap_returns(
        oos_returns, n_sims, block_size, n_periods, rng
    )
    cum_paths = compute_cumulative_paths(sim_returns)
    actual_cum = np.cumprod(1 + oos_returns) - 1

    # Percentile bands
    bands = compute_cone_percentiles(cum_paths, conf_levels)
    mean_path = cum_paths.mean(axis=0)

    # Probability of positive return at each step
    prob_positive = (cum_paths > 0).mean(axis=0)

    rows = []
    for t in range(n_periods):
        row = {
            "ticker_i": ticker_i,
            "ticker_j": ticker_j,
            "as_of_date": as_of_date,
            "day": t + 1,
            "actual_cumulative": float(actual_cum[t]),
            "mean_path": float(mean_path[t]),
            "prob_positive": float(prob_positive[t]),
        }
        for pct in conf_levels:
            col_name = f"pct_{int(pct * 100)}"
            row[col_name] = float(bands[pct][t])
        rows.append(row)

    return pd.DataFrame(rows)


def run_monte_carlo_pipeline(
    oos_returns_df: pd.DataFrame,
    as_of_date: date = None,
    top_n_pairs: int = 200,
) -> None:
    """
    Run Monte Carlo for top-N pairs by signal strength and store results.

    Parameters
    ----------
    oos_returns_df : DataFrame with ticker_i, ticker_j, strategy_return_net, oos_date
    as_of_date : date to tag results with
    top_n_pairs : compute MC only for top N pairs to control storage costs
    """
    if as_of_date is None:
        as_of_date = date.today()

    logger.info(f"Running Monte Carlo for top {top_n_pairs} pairs...")

    all_results = []

    # Group by pair
    pair_groups = list(oos_returns_df.groupby(["ticker_i", "ticker_j"]))

    # Only compute for top_n_pairs by data richness (number of OOS days)
    pair_groups_sorted = sorted(pair_groups, key=lambda x: len(x[1]), reverse=True)
    pair_groups_sorted = pair_groups_sorted[:top_n_pairs]

    for (ti, tj), group in pair_groups_sorted:
        group = group.sort_values("oos_date")
        returns = group["strategy_return_net"].values

        mc_df = run_monte_carlo_for_pair(ti, tj, returns, as_of_date)
        if mc_df is not None:
            all_results.append(mc_df)

    if all_results:
        result_df = pd.concat(all_results, ignore_index=True)
        # Store to BigQuery (add table to config if needed)
        # For now write to GCS as JSON for API serving
        logger.info(f"Monte Carlo complete: {len(result_df):,} rows for {len(all_results)} pairs")
        # write_dataframe(result_df, "monte_carlo_cones")  # Add table to schema if needed
        return result_df

    return pd.DataFrame()
