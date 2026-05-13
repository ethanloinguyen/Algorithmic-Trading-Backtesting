"""
mc_engine.py
------------
Single entry point for the Monte Carlo portfolio risk simulation.

Usage
-----
from mc_engine import run_portfolio_risk

result = run_portfolio_risk(
    tickers=['AAPL', 'MSFT', 'NVDA', 'JPM', 'JNJ'],
    weights=[0.25, 0.25, 0.20, 0.15, 0.15],  # optional, defaults to equal weight
    horizon_days=63,                           # 21 / 63 / 126 / 252
    n_sims=1000,
    target_return=0.10,
    confidence_levels=[0.95, 0.99],
)

Output structure
----------------
{
  "tickers":      [...],          # tickers that were successfully simulated
  "missing":      [...],          # tickers that could not be fetched
  "weights":      {ticker: w},    # weights used (normalised)
  "horizon_days": 63,
  "n_simulations": 1000,
  "per_stock": {
      "AAPL": {
          "var_95":                        float,   # e.g. -0.142
          "cvar_95":                       float,   # e.g. -0.187
          "tail_risk_ratio_95":            float,   # cvar/var, > 1 = fatter tail
          "var_99":                        float,
          "cvar_99":                       float,
          "tail_risk_ratio_99":            float,
          "expected_max_drawdown":         float,   # e.g. -0.118
          "worst_case_max_drawdown_p95":   float,   # e.g. -0.231
          "prob_loss":                     float,   # e.g. 0.44
          "prob_return_above_10pct":       float,   # e.g. 0.21
          "skewness":                      float,
          "avg_recovery_days":             float | None,
          "sortino_ratio_historical_252d": float | None,
      },
      ...
  },
  "portfolio": {
      "var_95":                            float,
      "cvar_95":                           float,
      "tail_risk_ratio_95":                float,
      "var_99":                            float,
      "cvar_99":                           float,
      "tail_risk_ratio_99":                float,
      "expected_max_drawdown":             float,
      "worst_case_max_drawdown_p95":       float,
      "prob_loss":                         float,
      "prob_return_above_10pct":           float,
      "skewness":                          float,
      "avg_recovery_days":                 float | None,
      "sortino_ratio_historical_252d":     float | None,
      "diversification_benefit_95":        float,   # positive = diversification helps
      "diversification_benefit_99":        float,
      "risk_contribution_per_stock_95":    {ticker: float},
      "risk_contribution_per_stock_99":    {ticker: float},
  }
}
"""

import numpy as np

from data import fetch_returns
from model import fit_params
from simulate import run_simulation
from risk import compute_stock_metrics, compute_portfolio_metrics


def run_portfolio_risk(
    tickers: list,
    weights: list = None,
    horizon_days: int = 63,
    n_sims: int = 1000,
    target_return: float = 0.10,
    confidence_levels: list = None,
    seed: int = 42,
    include_simulation_data: bool = False,
) -> dict:
    """
    Run a full Monte Carlo portfolio risk assessment.

    Parameters
    ----------
    tickers           : list of ticker symbols (e.g. ['AAPL', 'MSFT', 'NVDA'])
    weights           : portfolio weights aligned with tickers; defaults to equal weight.
                        Will be re-normalised to sum to 1.0 automatically.
    horizon_days      : simulation horizon in trading days (21 / 63 / 126 / 252)
    n_sims            : number of simulation paths (1000 recommended minimum)
    target_return     : decimal return threshold for probability metric (e.g. 0.10 = 10%)
    confidence_levels : VaR / CVaR confidence levels (default [0.95, 0.99])
    seed              : random seed for reproducibility

    Returns
    -------
    Structured dict — see module docstring for full output schema.
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    # --- Data ---
    returns, available, missing = fetch_returns(tickers)

    if not available:
        raise ValueError(f"No valid tickers found. Requested: {tickers}")

    # --- Weights ---
    weights_arr = _resolve_weights(tickers, available, weights)

    # --- Model ---
    params = fit_params(returns)

    # --- Simulation ---
    paths = run_simulation(params, n_steps=horizon_days, n_sims=n_sims, seed=seed)

    # --- Risk metrics ---
    stock_metrics = compute_stock_metrics(
        paths,
        available,
        params["sortino_historical"],
        target_return=target_return,
        confidence_levels=confidence_levels,
    )

    portfolio_metrics = compute_portfolio_metrics(
        paths,
        weights_arr,
        available,
        returns,
        target_return=target_return,
        confidence_levels=confidence_levels,
    )

    result = {
        "tickers": available,
        "missing": missing,
        "weights": dict(zip(available, weights_arr.tolist())),
        "horizon_days": horizon_days,
        "n_simulations": n_sims,
        "per_stock": stock_metrics,
        "portfolio": portfolio_metrics,
    }

    if include_simulation_data:
        # Attach raw arrays needed by visualize.py — prefixed with underscore
        # to signal they are internal data, not output metrics.
        result["_paths"] = paths
        result["_params"] = params
        result["_weights_arr"] = weights_arr

    return result


def _resolve_weights(
    requested_tickers: list,
    available_tickers: list,
    weights: list,
) -> np.ndarray:
    """
    Resolve and normalise portfolio weights for the available tickers.

    If weights are provided, those corresponding to unavailable tickers are
    dropped and the remainder re-normalised to sum to 1.0.
    If no weights are provided, equal weight is applied.
    """
    n = len(available_tickers)

    if weights is None:
        return np.full(n, 1.0 / n)

    if len(weights) != len(requested_tickers):
        raise ValueError(
            f"Length of weights ({len(weights)}) must match "
            f"length of tickers ({len(requested_tickers)})."
        )

    ticker_to_weight = dict(zip(requested_tickers, weights))
    w = np.array([ticker_to_weight[t] for t in available_tickers], dtype=float)

    total = w.sum()
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")

    return w / total


if __name__ == "__main__":
    import json

    result = run_portfolio_risk(
        tickers=["AAPL", "NVDA", "JPM", "JNJ", "MSFT", "GOOG"],
        horizon_days=63,
        n_sims=1000,
        target_return=0.10,
    )
    print(json.dumps(result, indent=2))