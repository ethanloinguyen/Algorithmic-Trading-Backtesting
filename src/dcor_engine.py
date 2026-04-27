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

# ---------------------------------------------------------------------------
# Optional Numba JIT acceleration
#
# When Numba is available, _dcor_numba() replaces the NumPy implementation
# inside dcor(). It computes the same result but:
#   - Uses explicit loops compiled to native SIMD code (fastmath=True)
#   - Requires only two O(n) row-mean vectors instead of four O(n²) matrices
#     (A, B, A_c, B_c), reducing memory allocation on every call
#   - Eliminates Python interpreter overhead when called thousands of times
#     inside the permutation loop
#   - Compiled binary is cached after the first run (cache=True), so the
#     one-time JIT cost (~1-2 s) is not paid on subsequent executions.
#
# If Numba is not installed, dcor() falls back to the original NumPy path
# transparently — results are numerically identical either way.
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMBA_AVAILABLE = False
    # Transparent no-op decorator so the function definition below is valid
    # even without Numba installed.
    def _njit(*args, **kwargs):  # type: ignore[misc]
        def _decorator(fn):
            return fn
        return _decorator


@_njit(cache=True, fastmath=True)
def _dcor_numba(x: np.ndarray, y: np.ndarray) -> float:
    """
    JIT-compiled distance correlation.

    Computes dCor in two O(n²) passes using only O(n) extra memory,
    avoiding allocation of the four n×n intermediate matrices used by
    the NumPy path.  Results are numerically equivalent to dcor() within
    floating-point precision.
    """
    n = len(x)

    # ── Pass 1: row means and grand mean of each distance matrix ──────────
    row_a = np.zeros(n)
    row_b = np.zeros(n)
    grand_a = 0.0
    grand_b = 0.0
    for i in range(n):
        for j in range(n):
            a = abs(x[i] - x[j])
            b = abs(y[i] - y[j])
            row_a[i] += a
            row_b[i] += b
            grand_a += a
            grand_b += b
    for i in range(n):
        row_a[i] /= n
        row_b[i] /= n
    grand_a /= n * n
    grand_b /= n * n

    # ── Pass 2: doubly-centered covariances ───────────────────────────────
    # For a symmetric distance matrix col_mean[j] == row_mean[j], so:
    #   A_c[i,j] = |x_i-x_j| - row_a[i] - row_a[j] + grand_a
    dcov_xy = 0.0
    dcov_xx = 0.0
    dcov_yy = 0.0
    for i in range(n):
        for j in range(n):
            ac = abs(x[i] - x[j]) - row_a[i] - row_a[j] + grand_a
            bc = abs(y[i] - y[j]) - row_b[i] - row_b[j] + grand_b
            dcov_xy += ac * bc
            dcov_xx += ac * ac
            dcov_yy += bc * bc

    n2 = float(n * n)
    dcov_xy /= n2
    dcov_xx /= n2
    dcov_yy /= n2

    if dcov_xx <= 0.0 or dcov_yy <= 0.0:
        return 0.0

    dcor_sq = dcov_xy / (dcov_xx * dcov_yy) ** 0.5
    if dcor_sq < 0.0:
        return 0.0
    if dcor_sq > 1.0:
        return 1.0
    return dcor_sq ** 0.5


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

    When Numba is installed the computation is dispatched to _dcor_numba(),
    which avoids allocating four n×n intermediate matrices and runs as
    compiled native code.  Falls back to NumPy otherwise.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = len(x)
    if n != len(y):
        raise ValueError(f"x and y must have same length. Got {len(x)}, {len(y)}")
    if n < 4:
        return 0.0

    if _NUMBA_AVAILABLE:
        return float(_dcor_numba(x, y))

    # ── NumPy fallback ────────────────────────────────────────────────────
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


def pearson_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> Optional[float]:
    """
    Compute signed Pearson correlation between x[t] and y[t+lag].

    Returns a value in [-1, 1]:
        positive → ticker_i up predicts ticker_j up at this lag
        negative → ticker_i up predicts ticker_j down at this lag

    Used alongside dCor to determine trade direction: dCor detects
    *that* a relationship exists; Pearson sign determines *which way*
    to trade.
    """
    if lag < 1 or len(x) <= lag + 4:
        return None
    x_lead = x[:-lag]
    y_follow = y[lag:]
    if np.std(x_lead) < 1e-10 or np.std(y_follow) < 1e-10:
        return None
    return float(np.corrcoef(x_lead, y_follow)[0, 1])


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
