"""
stability.py
------------
Computes cross-window stability metrics for each pair.

For each pair (i,j) that appeared in at least one window:
    - mean_dcor       : Mean dCor across windows where significant
    - variance_dcor   : Variance of dCor across windows
    - frequency       : Fraction of windows where pair was significant
    - half_life       : Estimated persistence via exponential decay fit
    - sharpness       : Concentration of dCor mass at single lag
    - n_windows       : Number of windows observed

These become the features X_ij for the OOS regression.
"""

import logging
from datetime import date
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from src.bq_io import (read_all_pair_results_filtered, write_dataframe,
                       get_client, full_table)
from src.config_loader import get_config
from src.dcor_engine import compute_sharpness

logger = logging.getLogger(__name__)


# ── Half-Life Estimation ─────────────────────────────────────────────────────

def _exponential_decay(t: np.ndarray, A: float, lam: float) -> np.ndarray:
    """Model: f(t) = A * exp(-lambda * t)"""
    return A * np.exp(-lam * t)


def estimate_half_life(dcor_series: np.ndarray, window_indices: np.ndarray) -> Tuple[Optional[float], float, bool]:
    """
    Fit exponential decay to dCor values across rolling windows.

    Parameters
    ----------
    dcor_series : dCor values over time (for significant windows only)
    window_indices : integer indices of windows (0, 1, 2, ...)

    Returns
    -------
    (half_life_days, fit_r2, is_stable)
    half_life_days : ln(2) / lambda, in units of steps (multiply by step_days for calendar)
    fit_r2 : R² of exponential fit
    is_stable : True if fit_r2 >= min_r2_for_stable threshold
    """
    cfg = get_config()
    min_r2 = cfg["half_life"]["min_r2_for_stable"]
    max_hl = cfg["half_life"]["max_half_life_days"]
    step_days = cfg["windows"]["step_days"]

    if len(dcor_series) < 3:
        return None, 0.0, False

    t = window_indices.astype(float)
    y = dcor_series.astype(float)

    # Normalize t to [0, n]
    t = t - t.min()

    try:
        popt, _ = curve_fit(
            _exponential_decay, t, y,
            p0=[y.max(), 0.1],
            bounds=([0, 1e-6], [np.inf, 10.0]),
            maxfev=2000
        )
        A, lam = popt
        y_pred = _exponential_decay(t, A, lam)

        # R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Half-life in window steps, then convert to calendar days
        hl_steps = np.log(2) / lam
        hl_days = hl_steps * step_days

        # Cap at max
        hl_days = min(hl_days, max_hl)

        is_stable = r2 >= min_r2

        return float(hl_days), float(r2), is_stable

    except (RuntimeError, ValueError):
        return None, 0.0, False


# ── Sharpness Across Windows ─────────────────────────────────────────────────

def compute_mean_sharpness(pair_rows: pd.DataFrame) -> float:
    """
    Compute mean sharpness across all windows for a pair.
    Sharpness per window requires the full lag profile — stored in pair_results_raw.
    Here we use a simplified version: sharpness of the single best lag vs others.
    For full entropy-based sharpness, see dcor_engine.compute_sharpness().
    """
    # If we only have the best lag per pair (after FDR), sharpness is computed
    # during the pair job and stored in pair_results_raw.
    # Here we average it across windows.
    if "sharpness" in pair_rows.columns:
        return float(pair_rows["sharpness"].mean())
    return 0.0


# ── Core Stability Computation ───────────────────────────────────────────────

def compute_stability_metrics() -> pd.DataFrame:
    """
    Compute stability metrics for all pairs across all historical windows.

    Steps:
    1. Load all FDR-filtered pair results
    2. For each pair, compute: mean_dcor, variance_dcor, frequency,
       half_life, sharpness
    3. Write to stability_metrics table

    Returns
    -------
    DataFrame with one row per pair.
    """
    logger.info("Computing stability metrics across all windows...")
    cfg = get_config()

    # Load all significant pairs across windows
    filtered = read_all_pair_results_filtered()
    if filtered.empty:
        logger.warning("No filtered pair results found. Cannot compute stability.")
        return pd.DataFrame()

    # Also pull sharpness from raw results
    client = get_client()
    raw_query = f"""
        SELECT window_start, ticker_i, ticker_j, lag, dcor, sharpness
        FROM `{full_table('pair_results_raw')}`
        WHERE dcor IS NOT NULL
    """
    raw_df = client.query(raw_query).to_dataframe()

    # Total number of windows
    all_windows = sorted(filtered["window_start"].unique())
    total_windows = len(all_windows)
    window_index_map = {w: i for i, w in enumerate(all_windows)}

    logger.info(f"Processing {total_windows} windows, "
                f"{filtered[['ticker_i','ticker_j']].drop_duplicates().shape[0]:,} unique pairs")

    # Merge sharpness into filtered
    if not raw_df.empty:
        filtered = filtered.merge(
            raw_df[["window_start", "ticker_i", "ticker_j", "sharpness"]],
            on=["window_start", "ticker_i", "ticker_j"],
            how="left"
        )

    stability_rows = []

    for (ticker_i, ticker_j), group in filtered.groupby(["ticker_i", "ticker_j"]):
        group = group.sort_values("window_start")

        # Assign window indices for decay fitting
        win_indices = np.array([window_index_map[w] for w in group["window_start"]])
        dcor_vals = group["dcor"].values

        # Frequency: fraction of all windows where this pair was significant
        frequency = len(group) / total_windows

        # Mean and variance of dCor across significant windows
        mean_dcor = float(dcor_vals.mean())
        variance_dcor = float(dcor_vals.var()) if len(dcor_vals) > 1 else 0.0

        # Half-life
        hl, hl_r2, hl_stable = estimate_half_life(dcor_vals, win_indices)

        # Best lag (mode across significant windows)
        best_lag = int(group["lag"].mode().iloc[0]) if "lag" in group.columns else None

        # Sharpness (mean across windows)
        sharpness = compute_mean_sharpness(group)

        stability_rows.append({
            "ticker_i": ticker_i,
            "ticker_j": ticker_j,
            "best_lag": best_lag,
            "mean_dcor": mean_dcor,
            "variance_dcor": variance_dcor,
            "frequency": frequency,
            "half_life": hl if hl is not None else -1.0,  # -1 = fit failed
            "half_life_r2": hl_r2,
            "half_life_stable": hl_stable,
            "sharpness": sharpness,
            "n_windows_observed": len(group),
            "last_updated": date.today(),
        })

    result = pd.DataFrame(stability_rows)
    logger.info(f"Stability metrics computed for {len(result):,} pairs")

    write_dataframe(result, "stability_metrics", write_disposition="WRITE_TRUNCATE")
    return result
