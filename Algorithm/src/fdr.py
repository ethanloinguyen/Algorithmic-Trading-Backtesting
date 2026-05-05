"""
fdr.py
------
Benjamini-Hochberg FDR correction for multiple testing.

Applied cross-sectionally across all pair-lag combinations
within a given rolling window.

Key design decision:
    - Only best lag per pair is kept BEFORE FDR (reduces test burden)
    - FDR controls the expected fraction of false discoveries
      among all declared significant pairs
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from Algorithm.src.config_loader import get_config

logger = logging.getLogger(__name__)


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg procedure for FDR control.

    Parameters
    ----------
    p_values : array of p-values (length m)
    alpha : FDR level (default 0.05)

    Returns
    -------
    (significant, q_values)
    significant : boolean array, True if pair is significant
    q_values : BH-adjusted p-values (q-values)
    """
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=bool), np.array([])

    # Rank p-values (1-indexed)
    order = np.argsort(p_values)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)

    # BH threshold: p_(k) <= k/m * alpha
    sorted_p = p_values[order]
    bh_thresholds = (np.arange(1, m + 1) / m) * alpha

    # Find largest k where p_(k) <= threshold
    below = sorted_p <= bh_thresholds
    if below.any():
        k_max = np.where(below)[0].max()
    else:
        k_max = -1

    significant_sorted = np.zeros(m, dtype=bool)
    if k_max >= 0:
        significant_sorted[: k_max + 1] = True

    # Map back to original order
    significant = significant_sorted[ranks - 1]

    # Compute q-values (Storey-style: q_i = min over j>=i of p_(j)*m/j)
    q_sorted = np.zeros(m)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        q_sorted[i] = min(sorted_p[i] * m / (i + 1), running_min)
        running_min = q_sorted[i]
    q_values = q_sorted[ranks - 1]

    return significant, q_values


def apply_fdr_to_window(raw_df: pd.DataFrame, window_start) -> pd.DataFrame:
    """
    Apply BH FDR correction to all pair results for a given window.

    Input DataFrame columns:
        ticker_i, ticker_j, lag, dcor, p_value, permutations_used

    Strategy:
        1. For each pair (i,j), keep only the lag with highest dCor
           (where dCor is not None)
        2. Apply BH correction across all kept pairs
        3. Return filtered DataFrame with q_values and significant flag

    Returns
    -------
    DataFrame with columns:
        window_start, ticker_i, ticker_j, lag, dcor, q_value, significant
    """
    cfg = get_config()
    alpha = cfg["fdr"]["alpha"]

    if raw_df.empty:
        logger.warning(f"Empty raw results for window {window_start}")
        return pd.DataFrame()

    # Drop rows where dCor is null (insufficient data)
    df = raw_df.dropna(subset=["dcor"]).copy()
    df = df[df["dcor"] > 0].copy()

    if df.empty:
        logger.warning(f"No valid dCor values for window {window_start}")
        return pd.DataFrame()

    # ── Step 1: Keep best lag per pair ───────────────────────────────────
    # Best = highest dCor value
    best_lag_idx = df.groupby(["ticker_i", "ticker_j"])["dcor"].idxmax()
    df_best = df.loc[best_lag_idx].copy()

    logger.info(f"Window {window_start}: {len(df_best):,} unique pairs with valid dCor")

    # ── Step 2: BH FDR correction ─────────────────────────────────────────
    p_values = df_best["p_value"].values
    significant, q_values = benjamini_hochberg(p_values, alpha=alpha)

    df_best["q_value"] = q_values
    df_best["significant"] = significant
    df_best["window_start"] = window_start

    n_sig = significant.sum()
    logger.info(
        f"Window {window_start}: {n_sig:,} / {len(df_best):,} pairs significant "
        f"at FDR q < {alpha}"
    )

    # ── Return all pairs (significant + not) for completeness ─────────────
    base_cols = ["window_start", "ticker_i", "ticker_j", "lag",
                 "dcor", "q_value", "significant"]
    if "pearson_corr" in df_best.columns:
        base_cols.append("pearson_corr")
    out = df_best[base_cols].reset_index(drop=True)

    return out


def apply_fdr_pipeline(window_start, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper: run FDR on raw pair results and return filtered DataFrame.
    """
    from Algorithm.src.bq_io import write_dataframe
    filtered = apply_fdr_to_window(raw_df, window_start)
    if not filtered.empty:
        write_dataframe(filtered, "pair_results_filtered")
    return filtered
