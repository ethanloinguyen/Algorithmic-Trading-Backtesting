"""
simulate.py
-----------
Runs a correlated Geometric Brownian Motion simulation with t-distributed
innovations and zero drift.

Key design decisions:
  - Zero drift: the simulation is a conservative stress test, not a return
    forecast. Estimating drift from a short historical window introduces more
    noise than signal, so we set it to zero.
  - T-distributed innovations: each stock draws from a fitted t-distribution
    (degrees of freedom from model.py) rather than a normal distribution.
    This produces fat tails consistent with observed equity return behaviour.
  - Cholesky correlation: innovations are linearly mixed via the Cholesky
    factor of the historical correlation matrix so stocks move together
    realistically across all simulated paths.
  - Ito correction (-0.5 * sigma^2 * dt): standard GBM drift correction that
    ensures the log-price process is unbiased under zero drift.
"""

import numpy as np


def run_simulation(
    params: dict,
    n_steps: int,
    n_sims: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate correlated stock price paths.

    Parameters
    ----------
    params  : dict returned by model.fit_params()
    n_steps : number of trading days to simulate (e.g. 21, 63, 126, 252)
    n_sims  : number of independent simulation paths
    seed    : random seed for reproducibility

    Returns
    -------
    paths : np.ndarray, shape (n_sims, n_steps + 1, n_stocks)
        Normalised price paths — each stock starts at 1.0.
        Terminal value - 1.0 = cumulative return over the horizon.
    """
    rng = np.random.default_rng(seed)

    n_stocks = len(params["tickers"])
    sig = params["daily_sigma"]        # shape (n_stocks,)
    dof_list = params["dof"]           # list of floats, one per stock
    L = params["L"]                    # Cholesky factor, shape (n_stocks, n_stocks)
    dt = 1.0  # one step = one trading day; sig is already daily volatility

    paths = np.ones((n_sims, n_steps + 1, n_stocks))

    for t in range(n_steps):
        z_corr = _draw_correlated_innovations(rng, dof_list, L, n_sims, n_stocks)

        # GBM step: zero drift with Ito correction
        r_t = -0.5 * sig ** 2 * dt + sig * np.sqrt(dt) * z_corr
        paths[:, t + 1, :] = paths[:, t, :] * np.exp(r_t)

    return paths


def _draw_correlated_innovations(
    rng: np.random.Generator,
    dof_list: list,
    L: np.ndarray,
    n_sims: int,
    n_stocks: int,
) -> np.ndarray:
    """
    Draw t-distributed innovations per stock and mix them via Cholesky to
    produce correlated, fat-tailed shocks.

    Each stock's innovation is drawn from t(df_i) and standardised to unit
    variance (t-distribution has variance df/(df-2)) before the Cholesky mix.
    This ensures that the per-stock volatility scaling in the GBM step remains
    correct regardless of degrees of freedom.

    Returns shape (n_sims, n_stocks).
    """
    z_raw = np.zeros((n_sims, n_stocks))
    for i, df in enumerate(dof_list):
        samples = rng.standard_t(df, size=n_sims)
        # Standardise to unit variance
        z_raw[:, i] = samples / np.sqrt(df / (df - 2))

    # Apply Cholesky to induce cross-stock correlation
    return (L @ z_raw.T).T