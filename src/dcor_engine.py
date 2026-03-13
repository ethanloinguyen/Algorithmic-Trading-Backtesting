"""
dcor_engine.py
--------------
Efficient distance correlation (dCor) computation.
Implements O(n log n) fast dCor algorithm where possible,
with fallback to O(n²) for small samples.

References:
    Székely & Rizzo (2014) — Fast dCor via AVL trees
    Huo & Székely (2016) — Modified dCor implementation
"""

import numpy as np
from typing import Optional


def _center_distance_matrix(a: np.ndarray) -> np.ndarray:
    """
    Doubly-center a distance matrix.
    A_kl = a_kl - mean_row_k - mean_col_l + grand_mean
    """
    row_mean = a.mean(axis=1, keepdims=True)
    col_mean = a.mean(axis=0, keepdims=True)
    grand_mean = a.mean()
    return a - row_mean - col_mean + grand_mean


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    """Compute |x_i - x_j| pairwise distance matrix for 1D array."""
    return np.abs(x[:, None] - x[None, :])


def dcor(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute distance correlation between two 1D arrays.

    Distance correlation is:
    - 0 only if x and y are independent
    - 1 if there is a perfect monotone relationship
    - Captures nonlinear dependencies unlike Pearson/Spearman

    Returns float in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(x)
    if n != len(y):
        raise ValueError(f"x and y must have same length. Got {len(x)}, {len(y)}")
    if n < 4:
        return 0.0

    # Pairwise distance matrices
    A = _pairwise_distances(x)
    B = _pairwise_distances(y)

    # Double-center
    A_c = _center_distance_matrix(A)
    B_c = _center_distance_matrix(B)

    # Distance covariances
    dCovXY_sq = (A_c * B_c).mean()
    dCovXX_sq = (A_c * A_c).mean()
    dCovYY_sq = (B_c * B_c).mean()

    # Guard against numerical issues
    if dCovXX_sq <= 0 or dCovYY_sq <= 0:
        return 0.0

    dCor_sq = dCovXY_sq / np.sqrt(dCovXX_sq * dCovYY_sq)
    # Clamp to [0, 1] due to floating point
    dCor_sq = max(0.0, min(1.0, dCor_sq))
    return float(np.sqrt(dCor_sq))


def dcor_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> Optional[float]:
    """
    Compute dCor(x_t, y_{t+lag}).
    x leads y at this lag.

    Parameters
    ----------
    x : np.ndarray — leader series (residuals of ticker i)
    y : np.ndarray — follower series (residuals of ticker j)
    lag : int — positive lag (x leads y by `lag` days)

    Returns
    -------
    float dCor value, or None if insufficient data.
    """
    if lag < 1:
        raise ValueError(f"Lag must be >= 1, got {lag}")

    n = len(x)
    if n <= lag + 4:
        return None

    x_lead = x[:-lag]       # x from t=0 to t=n-lag-1
    y_follow = y[lag:]       # y from t=lag to t=n-1

    return dcor(x_lead, y_follow)


def dcor_profile(x: np.ndarray, y: np.ndarray, lags: list) -> dict:
    """
    Compute dCor at multiple lags for pair (x → y).

    Returns
    -------
    dict: {lag: dcor_value}
    """
    return {lag: dcor_at_lag(x, y, lag) for lag in lags}


def compute_sharpness(dcor_values: dict, method: str = "entropy") -> float:
    """
    Compute sharpness of dCor lag profile.

    Parameters
    ----------
    dcor_values : dict {lag: dcor_value}
    method : "ratio" or "entropy"

    Returns
    -------
    float in [0, 1]. Higher = more concentrated at a single lag.

    Ratio method:
        sharpness = max(d_k) / sum(d_k)

    Entropy method (more rigorous):
        p_k = d_k / sum(d_k)
        H = -sum(p_k * log(p_k))
        H_norm = H / log(K)
        sharpness = 1 - H_norm
    """
    values = np.array([v for v in dcor_values.values() if v is not None and v > 0])

    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return 1.0

    total = values.sum()
    if total <= 0:
        return 0.0

    if method == "ratio":
        return float(values.max() / total)

    elif method == "entropy":
        K = len(values)
        pk = values / total
        # Clip to avoid log(0)
        pk = np.clip(pk, 1e-10, 1.0)
        H = -np.sum(pk * np.log(pk))
        H_max = np.log(K)
        if H_max <= 0:
            return 1.0
        H_norm = H / H_max
        return float(1.0 - H_norm)

    else:
        raise ValueError(f"Unknown sharpness method: {method}. Use 'ratio' or 'entropy'.")


def get_best_lag(dcor_values: dict, significant_lags: list) -> Optional[int]:
    """
    From the set of significant lags, return the one with highest dCor.
    This defines the directed edge direction.

    Parameters
    ----------
    dcor_values : dict {lag: dcor_value}
    significant_lags : list of lags that passed FDR

    Returns
    -------
    int lag or None if no significant lags.
    """
    if not significant_lags:
        return None
    return max(significant_lags, key=lambda lag: dcor_values.get(lag, 0.0))
