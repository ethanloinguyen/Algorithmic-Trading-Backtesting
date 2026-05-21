# backend/app/services/pipeline_service.py
"""
Hierarchical clustering → Monte Carlo risk assessment pipeline.

Import strategy
---------------
hierarchical.py  lives in  <repo>/model/  and has no internal relative imports,
                 so we load it by file path (importlib) — avoids polluting
                 sys.path with the `model/` directory, which would shadow
                 <repo>/monte-carlo/model.py.

mc_engine.py     lives in  <repo>/monte-carlo/  and uses bare imports
                 (from data import …, from model import …).  We add that
                 directory to sys.path so all its siblings resolve correctly.
"""
from __future__ import annotations

import importlib.util
import logging
import pathlib
import sys

logger = logging.getLogger(__name__)

# ── Path setup (runs once at import time) ─────────────────────────────────────
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MC_DIR    = _REPO_ROOT / "monte-carlo"

if str(_MC_DIR) not in sys.path:
    sys.path.insert(0, str(_MC_DIR))

# Load hierarchical by file path so `model/` stays off sys.path
_HIER_PATH = _REPO_ROOT / "model" / "hierarchical.py"
_hier_spec  = importlib.util.spec_from_file_location("hierarchical", str(_HIER_PATH))
_hier_mod   = importlib.util.module_from_spec(_hier_spec)
_hier_spec.loader.exec_module(_hier_mod)
run_clustering = _hier_mod.run_clustering

from mc_engine import run_portfolio_risk  # noqa: E402  (monte-carlo/ now on path)


def run_risk_pipeline(
    user_portfolio: list[str],
    horizon_days: int = 63,
    n_sims: int = 1000,
    target_return: float = 0.10,
    confidence_levels: list[float] | None = None,
    seed: int = 42,
    bq_client=None,
) -> dict:
    """
    Run the full hierarchical clustering → Monte Carlo risk pipeline.

    Step 1 — hierarchical.run_clustering():
        Queries BigQuery for stocks decorrelated from user_portfolio,
        clusters them via K-Medoids, and selects one stock per sector.

    Step 2 — mc_engine.run_portfolio_risk():
        Simulates Monte Carlo paths for the recommended tickers and
        returns per-stock and portfolio-level risk metrics.

    Parameters
    ----------
    user_portfolio     : tickers already held by the user
    horizon_days       : simulation horizon (21 / 63 / 126 / 252 trading days)
    n_sims             : Monte Carlo simulation paths
    target_return      : decimal return threshold for probability metric
    confidence_levels  : VaR/CVaR confidence levels (default [0.95, 0.99])
    seed               : random seed for reproducibility
    bq_client          : optional pre-existing BigQuery client

    Returns
    -------
    {
      "user_portfolio":  [...],
      "recommendations": [
          {sector, stock, cluster, is_medoid, avg_dcor_to_portfolio,
           mean_intra_dist, n_sector_candidates, cluster_size}, ...
      ],
      "risk": {
          "tickers":       [...],
          "missing":       [...],
          "weights":       {ticker: float},
          "horizon_days":  int,
          "n_simulations": int,
          "per_stock":     {ticker: {var_95, cvar_95, ...}},
          "portfolio":     {var_95, cvar_95, diversification_benefit_95, ...},
      }
    }
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    logger.info("Pipeline step 1: running hierarchical clustering for %s", user_portfolio)
    recommendations = run_clustering(
        user_portfolio=user_portfolio,
        bq_client=bq_client,
    )

    rec_tickers = recommendations["stock"].tolist()
    if not rec_tickers:
        raise ValueError("Hierarchical clustering returned no recommendations.")

    logger.info("Pipeline step 2a: running Monte Carlo per-stock risk for recommendations %s", rec_tickers)
    rec_risk = run_portfolio_risk(
        tickers=rec_tickers,
        horizon_days=horizon_days,
        n_sims=n_sims,
        target_return=target_return,
        confidence_levels=confidence_levels,
        seed=seed,
    )

    logger.info("Pipeline step 2b: running Monte Carlo portfolio risk for user holdings %s", list(user_portfolio))
    user_risk = run_portfolio_risk(
        tickers=list(user_portfolio),
        horizon_days=horizon_days,
        n_sims=n_sims,
        target_return=target_return,
        confidence_levels=confidence_levels,
        seed=seed,
    )

    # per_stock metrics come from the clustering recommendations;
    # portfolio-level metrics (VaR, CVaR, drawdown, etc.) come from the user's own holdings.
    risk = {
        **rec_risk,
        "portfolio": user_risk["portfolio"],
    }

    return {
        "user_portfolio":  user_portfolio,
        "recommendations": recommendations.to_dict("records"),
        "risk":            risk,
    }