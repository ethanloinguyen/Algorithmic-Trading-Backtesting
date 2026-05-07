import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

router = APIRouter(prefix="/api/montecarlo", tags=["montecarlo"])

TRAIN_START = "2015-01-01"
TRAIN_END   = "2019-12-31"
TEST_START  = "2020-01-01"
TEST_END    = "2024-12-31"
N_SIMS      = 500
MAX_LAG     = 5          # maximum lag days considered for VAR matrix

@lru_cache(maxsize=128)
def _download(symbol: str) -> pd.DataFrame:
    """Download adjusted daily closes, cached per symbol."""
    data = yf.download(symbol, start=TRAIN_START, end=TEST_END,
                       auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    data = data.dropna()
    return data


def _log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def _extract_params(train_ret: pd.Series) -> dict:
    """Extract annualised GBM drift and volatility from training log-returns."""
    mu    = float(train_ret.mean() * 252)
    sigma = float(train_ret.std()  * np.sqrt(252))
    return {"mu": mu, "sigma": sigma}


def _run_gbm(mu: float, sigma: float, S0: float,
             n_steps: int, n_sims: int = N_SIMS) -> np.ndarray:
    """
    Simulate n_sims GBM paths of length n_steps+1 starting at S0.
    Returns array of shape (n_sims, n_steps+1).
    """
    dt    = 1 / 252
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = S0
    for t in range(n_steps):
        z = np.random.normal(0, 1, n_sims)
        r = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        paths[:, t+1] = paths[:, t] * np.exp(r)
    return paths


def _run_var_gbm(mu: float, sigma: float, S0_leader: float, S0_follower: float,
                 beta: float, lag: int, n_steps: int,
                 n_sims: int = N_SIMS) -> np.ndarray:
    """
    Simulate follower paths with VAR lead-lag nudge from leader.
    Leader itself is simulated as pure GBM.
    Returns follower paths of shape (n_sims, n_steps+1).
    """
    dt = 1 / 252

    # Leader paths (pure GBM — it doesn't receive a nudge)
    leader_paths = np.zeros((n_sims, n_steps + 1))
    leader_paths[:, 0] = S0_leader
    leader_ret   = np.zeros((n_sims, n_steps + lag + 1))

    follower_paths = np.zeros((n_sims, n_steps + 1))
    follower_paths[:, 0] = S0_follower

    # Simulate both simultaneously so leader returns are available at each lag
    for sim in range(n_sims):
        l_price = S0_leader
        f_price = S0_follower
        l_rets  = []

        for t in range(n_steps):
            # Leader GBM step
            zl = np.random.normal()
            rl = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * zl
            l_price *= np.exp(rl)
            l_rets.append(rl)

            # Follower GBM step + VAR nudge from leader's lagged return
            zf      = np.random.normal()
            rf_base = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * zf
            var_c   = beta * l_rets[-lag] if len(l_rets) >= lag else 0.0
            f_price *= np.exp(rf_base + var_c)

            leader_paths[sim, t+1]   = l_price
            follower_paths[sim, t+1] = f_price

    return follower_paths


def _percentile_bands(paths: np.ndarray) -> dict:
    """Compute percentile bands across simulations at each time step."""
    return {
        "p5":  np.percentile(paths, 5,  axis=0),
        "p16": np.percentile(paths, 16, axis=0),
        "p50": np.percentile(paths, 50, axis=0),
        "p84": np.percentile(paths, 84, axis=0),
        "p95": np.percentile(paths, 95, axis=0),
    }


def _coverage(actual: np.ndarray, bands: dict) -> float:
    """Fraction of actual prices that fall inside the 1σ (p16–p84) band."""
    n = min(len(actual), len(bands["p16"]))
    if n == 0:
        return 0.0
    inside = (actual[:n] >= bands["p16"][:n]) & (actual[:n] <= bands["p84"][:n])
    return float(inside.mean())


def _build_result(symbol: str, train_prices: pd.Series,
                  test_prices: pd.Series, bands: dict,
                  mu: float, sigma: float) -> dict:
    """
    Serialize train prices and percentile bands into the JSON shape
    expected by ConeChart in MonteCarloLab.tsx.
    """
    train_out = [
        {"day": i - len(train_prices), "date": str(d)[:10], "price": round(float(p), 2)}
        for i, (d, p) in enumerate(train_prices.items())
    ]

    actual_arr = test_prices.values
    band_out   = []
    for i in range(len(bands["p50"])):
        actual = round(float(actual_arr[i]), 2) if i < len(actual_arr) else None
        band_out.append({
            "day":    i,
            "date":   str(test_prices.index[i])[:10] if i < len(test_prices) else None,
            "p5":     round(float(bands["p5"][i]),  2),
            "p16":    round(float(bands["p16"][i]), 2),
            "p50":    round(float(bands["p50"][i]), 2),
            "p84":    round(float(bands["p84"][i]), 2),
            "p95":    round(float(bands["p95"][i]), 2),
            "actual": actual,
        })

    coverage = _coverage(actual_arr, bands)

    return {
        "symbol":       symbol,
        "train":        train_out,
        "bands":        band_out,
        "mu_annual":    round(mu,    4),
        "sigma_annual": round(sigma, 4),
        "n_sims":       N_SIMS,
        "coverage_1s":  round(coverage, 4),
    }

@router.get("/single")
def montecarlo_single(symbol: str = Query(..., description="Stock ticker, e.g. AAPL")):
    np.random.seed(42)
    symbol = symbol.upper().strip()

    try:
        prices = _download(symbol)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download {symbol}: {e}")

    if len(prices) < 200:
        raise HTTPException(status_code=400, detail=f"Insufficient price history for {symbol}.")

    train = prices[TRAIN_START:TRAIN_END]
    test  = prices[TEST_START:TEST_END]

    if len(train) < 100 or len(test) < 10:
        raise HTTPException(status_code=400, detail="Not enough train/test data.")

    train_ret = _log_returns(train)
    params    = _extract_params(train_ret)
    S0        = float(test.iloc[0])

    paths  = _run_gbm(params["mu"], params["sigma"], S0, len(test), N_SIMS)
    bands  = _percentile_bands(paths)
    result = _build_result(symbol, train, test, bands, params["mu"], params["sigma"])

    return JSONResponse(result)

@router.get("/pair")
def montecarlo_pair(
    leader:   str = Query(..., description="Leader ticker, e.g. WM"),
    follower: str = Query(..., description="Follower ticker, e.g. WMB"),
):
    np.random.seed(42)
    leader   = leader.upper().strip()
    follower = follower.upper().strip()

    if leader == follower:
        raise HTTPException(status_code=400, detail="Leader and follower must be different.")

    # Download prices for both
    try:
        l_prices = _download(leader)
        f_prices = _download(follower)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Align on common trading days
    common = l_prices.index.intersection(f_prices.index)
    l_prices = l_prices.loc[common]
    f_prices = f_prices.loc[common]

    if len(l_prices) < 200:
        raise HTTPException(status_code=400, detail="Insufficient shared price history.")

    # Train / test split
    l_train = l_prices[TRAIN_START:TRAIN_END]
    f_train = f_prices[TRAIN_START:TRAIN_END]
    l_test  = l_prices[TEST_START:TEST_END]
    f_test  = f_prices[TEST_START:TEST_END]

    if len(l_train) < 100 or len(f_test) < 10:
        raise HTTPException(status_code=400, detail="Not enough train/test data.")

    # Extract GBM params from follower training returns
    f_train_ret = _log_returns(f_train)
    l_train_ret = _log_returns(l_train)
    params      = _extract_params(f_train_ret)

    # Find the best lag (1–MAX_LAG) by highest |Pearson ρ| on training data.
    best_lag     = 1
    best_abs_rho = 0.0
    best_slope   = 0.0
    best_rho     = 0.0

    for lag in range(1, MAX_LAG + 1):
        rl = l_train_ret.iloc[:-lag].values
        rf = f_train_ret.iloc[lag:].values
        if len(rl) < 30:
            continue
        rho, _ = stats.pearsonr(rl, rf)
        if abs(rho) > best_abs_rho:
            best_abs_rho = abs(rho)
            best_lag     = lag
            best_rho     = rho
            slope, _, _, _, _ = stats.linregress(rl, rf)
            best_slope = slope  # carries correct sign + real-return-unit magnitude

    S0_leader   = float(l_test.iloc[0])
    S0_follower = float(f_test.iloc[0])
    n_steps     = len(f_test)

    # WITHOUT lead-lag — pure GBM on follower
    paths_wo = _run_gbm(params["mu"], params["sigma"], S0_follower, n_steps, N_SIMS)
    bands_wo = _percentile_bands(paths_wo)

    # WITH lead-lag — VAR-MC using OLS beta
    paths_wl = _run_var_gbm(
        params["mu"], params["sigma"],
        S0_leader, S0_follower,
        beta    = best_slope,
        lag     = best_lag,
        n_steps = n_steps,
        n_sims  = N_SIMS,
    )
    bands_wl = _percentile_bands(paths_wl)

    result = {
        "leader":   leader,
        "follower": follower,
        "lag":      best_lag,
        "beta":     round(best_slope, 6),
        "pearson":  round(best_rho,   4),
        "without":  _build_result(follower, f_train, f_test, bands_wo, params["mu"], params["sigma"]),
        "with_ll":  _build_result(follower, f_train, f_test, bands_wl, params["mu"], params["sigma"]),
    }

    return JSONResponse(result)


