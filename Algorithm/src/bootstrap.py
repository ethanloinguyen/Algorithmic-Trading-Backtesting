"""
bootstrap.py
------------
1. Cross-sectional OOS regression: Y_ij ~ X_ij features
   Learns β weights that map stability features → predicted Sharpe

2. Bootstrap CI: resample pairs 1000x, refit regression each time
   Returns mean weights + 95% confidence intervals

3. Signal Strength normalization:
   Winsorize at 5th/95th percentile
   Scale to 0-100

Features X_ij:
    - mean_dcor
    - variance_dcor
    - frequency
    - half_life
    - sharpness

Target Y_ij:
    - oos_sharpe_net (primary)
"""

import logging
from datetime import date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

from Algorithm.src.bq_io import write_dataframe, mark_model_weights_current
from Algorithm.src.config_loader import get_config

logger = logging.getLogger(__name__)

FEATURES = ["mean_dcor", "variance_dcor", "frequency", "half_life", "sharpness"]


def _prepare_regression_data(
    stability_df: pd.DataFrame,
    oos_sharpe_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Merge stability features with OOS Sharpe.
    Returns (X, y, merged_df) after cleaning.
    """
    merged = stability_df.merge(
        oos_sharpe_df[["ticker_i", "ticker_j", "oos_sharpe_net"]],
        on=["ticker_i", "ticker_j"],
        how="inner"
    )

    # Remove pairs with failed half-life fit (half_life = -1)
    merged = merged[merged["half_life"] > 0].copy()
    merged = merged.dropna(subset=FEATURES + ["oos_sharpe_net"])

    if len(merged) < 20:
        raise ValueError(f"Insufficient pairs for regression: {len(merged)}")

    X = merged[FEATURES].values
    y = merged["oos_sharpe_net"].values

    return X, y, merged


def fit_oos_regression(
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Fit OLS regression: y = β0 + β1*X1 + ... + βk*Xk

    Returns
    -------
    (coefficients_array, r2, f_statistic)
    coefficients_array includes intercept as first element.
    """
    X_with_const = np.column_stack([np.ones(len(X)), X])
    coeffs, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)

    # R²
    y_pred = X_with_const @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # F-statistic
    n, k = X.shape
    if n > k + 1 and ss_tot > 0:
        ms_reg = (ss_tot - ss_res) / k
        ms_res = ss_res / (n - k - 1)
        f_stat = ms_reg / ms_res if ms_res > 0 else 0.0
    else:
        f_stat = 0.0

    return coeffs, float(r2), float(f_stat)


def bootstrap_weights(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, dict]:
    """
    Bootstrap confidence intervals for regression weights.
    Resamples pairs with replacement, refits regression each time.

    Returns
    -------
    Dict with feature names as keys:
        {
            "intercept": {"mean": ..., "ci_lower": ..., "ci_upper": ...},
            "mean_dcor": {"mean": ..., "ci_lower": ..., "ci_upper": ...},
            ...
        }
    """
    cfg = get_config()
    n_bootstrap = cfg["model"]["bootstrap_n"]
    rng = np.random.default_rng(seed)

    n_pairs = len(X)
    all_coeffs = []

    logger.info(f"Running {n_bootstrap} bootstrap iterations on {n_pairs} pairs...")

    for b in range(n_bootstrap):
        # Resample pairs with replacement
        idx = rng.integers(0, n_pairs, size=n_pairs)
        X_boot = X[idx]
        y_boot = y[idx]

        try:
            coeffs, _, _ = fit_oos_regression(X_boot, y_boot)
            all_coeffs.append(coeffs)
        except Exception:
            continue

    if not all_coeffs:
        raise ValueError("All bootstrap iterations failed.")

    all_coeffs = np.array(all_coeffs)  # shape: (n_bootstrap, k+1)

    alpha = 1 - confidence_level
    lo_pct = alpha / 2 * 100
    hi_pct = (1 - alpha / 2) * 100

    feature_names = ["intercept"] + FEATURES
    results = {}
    for i, name in enumerate(feature_names):
        boot_vals = all_coeffs[:, i]
        results[name] = {
            "mean": float(np.mean(boot_vals)),
            "ci_lower": float(np.percentile(boot_vals, lo_pct)),
            "ci_upper": float(np.percentile(boot_vals, hi_pct)),
            "std": float(np.std(boot_vals)),
        }

    logger.info("Bootstrap complete.")
    return results


def run_model_refit(
    stability_df: pd.DataFrame,
    oos_sharpe_df: pd.DataFrame,
    model_version: str = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Full model refit pipeline:
    1. Prepare regression data
    2. Fit OLS
    3. Bootstrap CIs
    4. Store weights to BigQuery
    5. Mark as current

    Returns
    -------
    (weights_df, feature_weights_dict)
    feature_weights_dict: {feature_name: weight} for computing stability_score
    """
    if model_version is None:
        model_version = date.today().strftime("v%Y_%m")

    logger.info(f"Fitting model version: {model_version}")

    X, y, merged = _prepare_regression_data(stability_df, oos_sharpe_df)
    coeffs, r2, f_stat = fit_oos_regression(X, y)
    ci_dict = bootstrap_weights(X, y)

    logger.info(f"Model fit: R²={r2:.4f}, F={f_stat:.2f}, n_pairs={len(merged)}")

    # Prepare weights DataFrame for BigQuery
    feature_names = ["intercept"] + FEATURES
    rows = []
    for i, name in enumerate(feature_names):
        rows.append({
            "model_version": model_version,
            "refit_date": date.today(),
            "feature": name,
            "weight": float(coeffs[i]),
            "ci_lower": ci_dict[name]["ci_lower"],
            "ci_upper": ci_dict[name]["ci_upper"],
            "r2": r2,
            "f_statistic": f_stat,
            "n_pairs": len(merged),
            "is_current": True,
        })

    weights_df = pd.DataFrame(rows)

    # Mark all existing as not current, then write new
    try:
        mark_model_weights_current("__none__")  # mark all False
    except Exception:
        pass

    write_dataframe(weights_df, "model_weights", write_disposition="WRITE_APPEND")
    mark_model_weights_current(model_version)

    feature_weights = {FEATURES[i]: float(coeffs[i + 1]) for i in range(len(FEATURES))}
    feature_weights["intercept"] = float(coeffs[0])

    logger.info(f"Model weights stored: {feature_weights}")
    return weights_df, feature_weights


def compute_predicted_sharpe(
    stability_df: pd.DataFrame,
    feature_weights: Dict[str, float],
) -> pd.Series:
    """
    Apply frozen β weights to compute predicted Sharpe for each pair.
    stability_score = β0 + β1*mean_dcor + β2*variance_dcor + ...

    Returns
    -------
    Series indexed like stability_df with predicted Sharpe values.
    """
    intercept = feature_weights.get("intercept", 0.0)
    result = np.full(len(stability_df), intercept)
    for feat in FEATURES:
        if feat in feature_weights and feat in stability_df.columns:
            result += feature_weights[feat] * stability_df[feat].fillna(0).values
    return pd.Series(result, index=stability_df.index)


def compute_signal_strength(
    predicted_sharpe: pd.Series,
    lo_pct: float = 5,
    hi_pct: float = 95,
) -> pd.Series:
    """
    Normalize predicted Sharpe to 0-100 Signal Strength.
    Winsorizes at 5th/95th percentile to prevent extreme compression.

    Formula:
        SS = 100 * (sharpe - sharpe_min_win) / (sharpe_max_win - sharpe_min_win)
    """
    vals = predicted_sharpe.values.copy()

    # Winsorize
    lower = np.percentile(vals, lo_pct)
    upper = np.percentile(vals, hi_pct)
    vals_win = np.clip(vals, lower, upper)

    denom = upper - lower
    if denom <= 1e-10:
        return pd.Series(np.full(len(vals), 50.0), index=predicted_sharpe.index)

    signal_strength = 100.0 * (vals_win - lower) / denom
    signal_strength = np.clip(signal_strength, 0.0, 100.0)

    return pd.Series(signal_strength, index=predicted_sharpe.index)
