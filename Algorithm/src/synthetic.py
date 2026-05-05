"""
synthetic.py
------------
Fixed-seed synthetic data health check.
Runs monthly to verify pipeline integrity.

Design:
    - 50 planted pairs with known lag-2 nonlinear relationship
    - 450 null pairs (pure noise, some with volatility clustering)
    - Fixed seed (42) for month-to-month comparability

Planted relationship (threshold regime, nonlinear):
    B_t = 0.5 * A_{t-2} + eps    if A_{t-2} > 0
    B_t = eps                      otherwise

This creates nonlinear dependence dCor detects but Pearson misses.

Evaluates:
    - True Positive Rate (TPR): correctly identified planted pairs
    - False Positive Rate (FPR): null pairs incorrectly flagged
    - Stability ranking accuracy: planted pairs should rank higher

Alerts if TPR drops below threshold or FPR exceeds threshold.
"""

import logging
from datetime import date
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from Algorithm.src.bq_io import log_synthetic_health
from Algorithm.src.config_loader import get_config
from Algorithm.src.dcor_engine import dcor_at_lag, compute_sharpness, dcor_profile
from Algorithm.src.fdr import benjamini_hochberg
from Algorithm.src.permutation import test_pair_all_lags

logger = logging.getLogger(__name__)


def generate_synthetic_universe(
    n_obs: int = 300,
    n_planted: int = 50,
    n_null: int = 450,
    planted_lag: int = 2,
    noise_std: float = 1.0,
    volatility_clustering: bool = True,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], list, list]:
    """
    Generate synthetic time series for health check.
    Uses FIXED seed for reproducibility across monthly runs.

    Returns
    -------
    (series_dict, planted_pairs, null_pairs)
    series_dict: {ticker: np.ndarray of returns}
    planted_pairs: list of (ticker_i, ticker_j) tuples with real signal
    null_pairs: list of (ticker_i, ticker_j) tuples with no signal
    """
    rng = np.random.default_rng(seed)
    cfg = get_config()["synthetic"]

    series = {}
    planted_pairs = []
    null_pairs = []

    # ── Generate planted pairs ────────────────────────────────────────────
    for p in range(n_planted):
        ticker_A = f"PLANT_A_{p:03d}"
        ticker_B = f"PLANT_B_{p:03d}"

        # Leader A: AR(1) process
        A = np.zeros(n_obs)
        eps_A = rng.normal(0, noise_std, n_obs)
        for t in range(1, n_obs):
            A[t] = 0.3 * A[t - 1] + eps_A[t]

        # Follower B: threshold regime nonlinear relationship with A at lag 2
        B = np.zeros(n_obs)
        eps_B = rng.normal(0, noise_std, n_obs)
        for t in range(planted_lag, n_obs):
            if A[t - planted_lag] > 0:
                B[t] = 0.5 * A[t - planted_lag] + eps_B[t]
            else:
                B[t] = eps_B[t]

        series[ticker_A] = A
        series[ticker_B] = B
        planted_pairs.append((ticker_A, ticker_B))

    # ── Generate null pairs ────────────────────────────────────────────────
    for n in range(n_null):
        ticker_X = f"NULL_X_{n:03d}"
        ticker_Y = f"NULL_Y_{n:03d}"

        if volatility_clustering and n % 3 == 0:
            # GARCH-like: some null pairs have volatility clustering
            X = np.zeros(n_obs)
            sigma = np.ones(n_obs)
            for t in range(1, n_obs):
                sigma[t] = np.sqrt(0.1 + 0.3 * X[t-1]**2 + 0.6 * sigma[t-1]**2)
                X[t] = sigma[t] * rng.normal()
            Y = rng.normal(0, noise_std, n_obs)
        else:
            X = rng.normal(0, noise_std, n_obs)
            Y = rng.normal(0, noise_std, n_obs)

        series[ticker_X] = X
        series[ticker_Y] = Y
        null_pairs.append((ticker_X, ticker_Y))

    return series, planted_pairs, null_pairs


def run_synthetic_health_check() -> Dict:
    """
    Run the full synthetic health check.

    Steps:
    1. Generate fixed-seed synthetic universe
    2. Compute dCor + adaptive permutation for all pairs
    3. Apply BH FDR
    4. Evaluate TPR, FPR, ranking accuracy
    5. Log results to BigQuery

    Returns
    -------
    dict with health check results and alert flags
    """
    cfg = get_config()["synthetic"]
    alert_cfg = get_config()["synthetic"]

    logger.info("Running synthetic health check (fixed seed)...")

    n_planted = cfg["n_planted_pairs"]
    n_null = cfg["n_null_pairs"]
    planted_lag = cfg["planted_lag"]
    seed = cfg["random_seed"]
    noise_std = cfg["planted_noise_std"]
    vol_cluster = cfg["volatility_clustering"]

    # Generate synthetic data
    series, planted_pairs, null_pairs = generate_synthetic_universe(
        n_obs=300,
        n_planted=n_planted,
        n_null=n_null,
        planted_lag=planted_lag,
        noise_std=noise_std,
        volatility_clustering=vol_cluster,
        seed=seed,
    )

    all_pairs = planted_pairs + null_pairs
    planted_set = set(planted_pairs)

    lags = get_config()["lags"]["lag_list"]
    rng = np.random.default_rng(seed + 1)  # Separate seed for permutation RNG

    # Use a reduced permutation budget for the health check — enough precision
    # to evaluate TPR/FPR without burning the full main-pipeline budget.
    max_perms = cfg.get("permutation_n", 200)

    # ── Compute dCor + permutation for all pairs ───────────────────────────
    pair_results = []
    for (ti, tj) in all_pairs:
        x = series[ti]
        y = series[tj]
        lag_results = test_pair_all_lags(x, y, lags, rng=rng, max_permutations=max_perms)

        # Keep best lag
        best_lag = max(lag_results.keys(), key=lambda l: lag_results[l]["dcor"] or 0.0)
        best = lag_results[best_lag]

        pair_results.append({
            "ticker_i": ti,
            "ticker_j": tj,
            "lag": best_lag,
            "dcor": best["dcor"],
            "p_value": best["p_value"],
            "is_planted": (ti, tj) in planted_set,
        })

    results_df = pd.DataFrame(pair_results).dropna(subset=["dcor"])

    # ── Apply BH FDR ──────────────────────────────────────────────────────
    p_values = results_df["p_value"].values
    alpha = get_config()["fdr"]["alpha"]
    significant, q_values = benjamini_hochberg(p_values, alpha=alpha)

    results_df["significant"] = significant
    results_df["q_value"] = q_values

    # ── Evaluate performance ──────────────────────────────────────────────
    planted_mask = results_df["is_planted"]
    null_mask = ~planted_mask

    # TPR: among planted pairs, fraction correctly flagged significant
    n_planted_detected = (planted_mask & results_df["significant"]).sum()
    tpr = n_planted_detected / n_planted if n_planted > 0 else 0.0

    # FPR: among null pairs, fraction incorrectly flagged significant
    n_null_false_positive = (null_mask & results_df["significant"]).sum()
    fpr = n_null_false_positive / n_null if n_null > 0 else 0.0

    # Stability ranking accuracy:
    # Are planted pairs ranked higher (by dCor) than null pairs?
    results_df["rank"] = results_df["dcor"].rank(ascending=False)
    planted_mean_rank = results_df.loc[planted_mask, "rank"].mean()
    null_mean_rank = results_df.loc[null_mask, "rank"].mean()
    # Accuracy: fraction of planted pairs in top (n_planted) by rank
    top_n_mask = results_df["rank"] <= n_planted
    rank_accuracy = (top_n_mask & planted_mask).sum() / n_planted if n_planted > 0 else 0.0

    # ── Alert logic ───────────────────────────────────────────────────────
    alert_tpr = tpr < alert_cfg["min_tpr"]
    alert_fpr = fpr > alert_cfg["max_fpr"]

    if alert_tpr:
        logger.warning(f"SYNTHETIC ALERT: TPR={tpr:.2%} below threshold {alert_cfg['min_tpr']:.2%}")
    if alert_fpr:
        logger.warning(f"SYNTHETIC ALERT: FPR={fpr:.2%} above threshold {alert_cfg['max_fpr']:.2%}")

    status = "PASS"
    if alert_tpr or alert_fpr:
        status = "WARN"

    result = {
        "run_date": date.today(),
        "true_positive_rate": float(tpr),
        "false_positive_rate": float(fpr),
        "stability_rank_accuracy": float(rank_accuracy),
        "n_planted": n_planted,
        "n_null": n_null,
        "alert_tpr": bool(alert_tpr),
        "alert_fpr": bool(alert_fpr),
        "status": status,
    }

    logger.info(
        f"Synthetic health check: TPR={tpr:.2%}, FPR={fpr:.2%}, "
        f"Rank Acc={rank_accuracy:.2%}, Status={status}"
    )

    log_synthetic_health(result)
    return result
