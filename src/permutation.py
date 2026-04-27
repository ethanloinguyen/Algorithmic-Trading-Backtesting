"""
permutation.py
--------------
Adaptive 3-tier block permutation testing for dCor significance.

Tier 1: 100 permutations → stop if p > 0.20 (clearly null)
Tier 2: extend to 500    → stop if p > 0.10
Tier 3: extend to 1000   → hard ceiling

Block permutation preserves weekly autocorrelation structure.
Block size: 5 trading days (configurable).

Budget guard: tracks and alerts if >10% of pairs hit tier 3.
"""

import logging
from typing import Tuple

import numpy as np

from src.config_loader import get_config
from src.dcor_engine import dcor_at_lag

logger = logging.getLogger(__name__)


def block_shuffle(series: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Shuffle a time series by permuting contiguous blocks.
    Preserves within-block autocorrelation structure.

    Parameters
    ----------
    series : 1D array to shuffle
    block_size : number of consecutive observations per block
    rng : numpy random Generator

    Returns
    -------
    Shuffled array of same length.
    """
    n = len(series)
    n_blocks = int(np.ceil(n / block_size))

    # Create blocks (last block may be smaller)
    blocks = [series[i * block_size: (i + 1) * block_size] for i in range(n_blocks)]

    # Shuffle block order
    rng.shuffle(blocks)

    # Reconstruct and trim to original length
    shuffled = np.concatenate(blocks)[:n]
    return shuffled


def adaptive_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    lag: int,
    observed_dcor: float,
    rng: np.random.Generator = None,
    max_permutations: int = None,
) -> Tuple[float, int]:
    """
    Adaptive 3-tier permutation test for dCor at a given lag.

    Strategy:
        - Permute x (the leading series) using block shuffles
        - Keep y (follower) fixed
        - Count how often permuted dCor >= observed dCor

    Parameters
    ----------
    x : leader residual series
    y : follower residual series
    lag : lag being tested
    observed_dcor : pre-computed dCor(x, y, lag)
    rng : numpy random Generator (for reproducibility)

    Returns
    -------
    (p_value, n_permutations_used)
    """
    cfg = get_config()["permutation"]
    tier1_n = cfg["tier1_n"]
    tier1_cutoff = cfg["tier1_cutoff"]
    tier2_n = cfg["tier2_n"]
    tier2_cutoff = cfg["tier2_cutoff"]
    tier3_n = cfg["tier3_n"]
    block_size = cfg["block_size"]

    # Cap all tier budgets at max_permutations when provided (e.g. synthetic
    # health check uses a smaller budget than the main pipeline).
    if max_permutations is not None:
        tier1_n = min(tier1_n, max_permutations)
        tier2_n = min(tier2_n, max_permutations)
        tier3_n = min(tier3_n, max_permutations)

    if rng is None:
        rng = np.random.default_rng()
    if observed_dcor is None or np.isnan(observed_dcor):
        return 1.0, 0

    # Pre-align y once (fixed across all permutations)
    y_follow = y[lag:]
    n = len(x) - lag

    def run_batch(n_perms: int) -> np.ndarray:
        """Generate n_perms block-shuffled dCor values as a numpy array."""
        null_dcors = np.empty(n_perms)
        for i in range(n_perms):
            x_perm = block_shuffle(x, block_size, rng)
            null_dcors[i] = dcor_at_lag(x_perm, y, lag) or 0.0
        return null_dcors

    # Tier 1
    null1 = run_batch(tier1_n)
    exceed1 = np.sum(null1 >= observed_dcor)
    p1 = (exceed1 + 1) / (tier1_n + 1)
    if p1 > tier1_cutoff:
        return float(p1), tier1_n

    # Tier 2
    null2 = run_batch(tier2_n - tier1_n)
    exceed2 = exceed1 + np.sum(null2 >= observed_dcor)
    p2 = (exceed2 + 1) / (tier2_n + 1)
    if p2 > tier2_cutoff:
        return float(p2), tier2_n

    # Tier 3
    null3 = run_batch(tier3_n - tier2_n)
    exceed3 = exceed2 + np.sum(null3 >= observed_dcor)
    p3 = (exceed3 + 1) / (tier3_n + 1)
    return float(p3), tier3_n


def test_pair_all_lags(
    x: np.ndarray,
    y: np.ndarray,
    lags: list,
    rng: np.random.Generator = None,
    max_permutations: int = None,
) -> dict:
    """
    For a pair (x → y), compute dCor and p-value at all lags.
    Uses adaptive permutation at each lag independently.

    Tier 0 pre-screen: if observed dCor at a lag is below
    permutation.tier0_dcor_threshold (default 0.05), the permutation
    test is skipped entirely and p_value=1.0 is returned for that lag.
    For n≈247 observations the expected dCor under independence is
    ~1/√n ≈ 0.064, so any pair below 0.05 would stop at Tier 1 with
    p > 0.5 regardless — skipping costs nothing in statistical power.

    Parameters
    ----------
    x : leader residuals
    y : follower residuals
    lags : list of int lags to test

    Returns
    -------
    dict: {lag: {"dcor": float, "p_value": float, "permutations_used": int}}
    """
    from src.dcor_engine import dcor_at_lag
    if rng is None:
        rng = np.random.default_rng()

    tier0_threshold = get_config()["permutation"].get("tier0_dcor_threshold", 0.05)

    results = {}
    for lag in lags:
        observed = dcor_at_lag(x, y, lag)
        if observed is None:
            results[lag] = {"dcor": None, "p_value": 1.0, "permutations_used": 0}
            continue

        # Tier 0: skip permutation for clearly null lags
        if observed < tier0_threshold:
            results[lag] = {"dcor": observed, "p_value": 1.0, "permutations_used": 0}
            continue

        p_val, n_perms = adaptive_permutation_test(
            x, y, lag, observed, rng=rng, max_permutations=max_permutations
        )
        results[lag] = {
            "dcor": observed,
            "p_value": p_val,
            "permutations_used": n_perms,
        }

    return results


def check_budget_guard(tier3_count: int, total_count: int) -> bool:
    """
    Returns True if tier3_fraction exceeds budget guard threshold.
    Logs a warning if so.
    """
    if total_count == 0:
        return False
    cfg = get_config()["permutation"]
    fraction = tier3_count / total_count
    threshold = cfg["max_tier3_fraction"]
    if fraction > threshold:
        logger.warning(
            f"BUDGET ALERT: {fraction:.1%} of pairs hit tier 3 permutations "
            f"(threshold: {threshold:.1%}). Review compute costs."
        )
        return True
    return False
