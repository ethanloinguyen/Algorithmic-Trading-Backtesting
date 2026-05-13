"""
model.py
--------
Fits statistical parameters from historical returns needed to drive the simulation:
  - Per-stock annualised volatility
  - Per-stock t-distribution degrees of freedom (MLE) for fat-tail modelling
  - Correlation matrix with positive semi-definite correction
  - Cholesky factor for inducing cross-stock correlation
  - Historical Sortino ratio per stock (backward-looking calibration metric)

Note: drift (mu) is NOT used in the simulation — it is set to zero in simulate.py
so the model acts as a conservative stress test rather than a return forecast.
Mu is retained here only to compute the historical Sortino ratio.
"""

import numpy as np
import pandas as pd
from scipy import stats


def fit_params(returns: pd.DataFrame) -> dict:
    """
    Estimate all model parameters from a DataFrame of daily log returns.

    Parameters
    ----------
    returns : DataFrame, shape (n_days, n_stocks), daily log returns

    Returns
    -------
    dict with keys:
        tickers            : list of ticker symbols
        daily_sigma        : np.ndarray, daily volatility per stock
        sigma_annual       : np.ndarray, annualised volatility per stock
        dof                : list of floats, t-dist degrees of freedom per stock
        corr               : np.ndarray, PSD-corrected correlation matrix
        L                  : np.ndarray, Cholesky factor of correlation matrix
        sortino_historical : dict {ticker: float}, 252-day historical Sortino ratio
    """
    tickers = list(returns.columns)

    daily_sigma = returns.std().values
    sigma_annual = daily_sigma * np.sqrt(252)
    mu_annual = returns.mean().values * 252

    dof = _fit_dof(returns, tickers)
    corr, L = _fit_correlation(returns)
    sortino = _compute_sortino(returns, tickers, mu_annual)

    return {
        "tickers": tickers,
        "daily_sigma": daily_sigma,
        "sigma_annual": sigma_annual,
        "dof": dof,
        "corr": corr,
        "L": L,
        "sortino_historical": sortino,
    }


def _fit_dof(returns: pd.DataFrame, tickers: list) -> list:
    """
    Fit t-distribution degrees of freedom per stock via MLE on standardised returns.
    Enforces df >= 2.5 so that variance remains finite and well-behaved.
    """
    dof = []
    for ticker in tickers:
        standardised = returns[ticker] / returns[ticker].std()
        # Fix loc=0 and scale=1 — only estimate df
        df, _, _ = stats.t.fit(standardised, floc=0, fscale=1)
        dof.append(max(float(df), 2.5))
    return dof


def _fit_correlation(returns: pd.DataFrame) -> tuple:
    """
    Estimate the correlation matrix and return it with its Cholesky factor.

    Applies a nearest positive semi-definite correction (eigenvalue clipping)
    to handle near-singular matrices that arise with highly correlated assets.
    """
    C = returns.corr().values.copy()

    # PSD correction: clip negative eigenvalues to a small positive floor
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-8)
    C_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Re-normalise diagonal to exactly 1.0 after correction
    d = np.sqrt(np.diag(C_psd))
    C_psd = C_psd / d[:, None] / d[None, :]

    L = np.linalg.cholesky(C_psd)
    return C_psd, L


def _compute_sortino(
    returns: pd.DataFrame,
    tickers: list,
    mu_annual: np.ndarray,
) -> dict:
    """
    Compute the historical Sortino ratio for each stock over the lookback window.

    Sortino = annualised mean return / annualised downside deviation
    Downside deviation uses only negative daily returns (target = 0).

    This is a backward-looking metric derived from historical data, not from
    the simulation. It is included to give users context on past risk-adjusted
    efficiency, separately from the forward-looking simulation risk metrics.
    """
    sortino = {}
    for i, ticker in enumerate(tickers):
        r = returns[ticker]
        downside = r[r < 0].std() * np.sqrt(252)
        if downside > 1e-10:
            sortino[ticker] = float(mu_annual[i] / downside)
        else:
            sortino[ticker] = None
    return sortino