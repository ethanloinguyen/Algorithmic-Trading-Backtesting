"""
risk.py
-------
Computes risk metrics from simulated price paths, at both per-stock and
portfolio levels.

Per-stock metrics (computed from individual stock paths):
  - VaR 95% / 99%
  - CVaR 95% / 99%
  - Tail Risk Ratio (CVaR / VaR) per confidence level
  - Expected max drawdown
  - Worst-case max drawdown (95th percentile across simulations)
  - Probability of loss
  - Probability of exceeding a target return
  - Skewness of terminal return distribution
  - Average time to recovery from max drawdown trough
  - Sortino ratio (historical, from model.py — not simulation-derived)

Portfolio-level metrics (computed from weighted aggregate paths):
  - All per-stock metrics applied to the portfolio return stream
  - Diversification benefit: reduction in VaR vs weighted sum of individual VaRs
  - Risk contribution per stock: each holding's share of portfolio tail losses
"""

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_CONFIDENCE_LEVELS = [0.95, 0.99]


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def terminal_returns(paths: np.ndarray) -> np.ndarray:
    """
    Extract cumulative returns at the end of the horizon.

    Parameters
    ----------
    paths : shape (n_sims, n_steps + 1, n_stocks), normalised to start at 1.0

    Returns
    -------
    shape (n_sims, n_stocks) — each value is final_price - 1.0
    """
    return paths[:, -1, :] - 1.0


def aggregate_portfolio_paths(paths: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Combine per-stock paths into a single portfolio path using fixed weights.

    Portfolio cumulative return at step t =
        sum_i( weight_i * (stock_i_price_t / stock_i_price_0 - 1) )

    This assumes daily rebalancing to maintain target weights, which is the
    standard assumption for portfolio risk aggregation.

    Parameters
    ----------
    paths   : shape (n_sims, n_steps + 1, n_stocks)
    weights : shape (n_stocks,), must sum to 1.0

    Returns
    -------
    portfolio_paths : shape (n_sims, n_steps + 1, 1)
        Structured as a single-stock paths array so all metric functions
        accept it without modification.
    """
    cum_returns = paths - 1.0
    port_cum = (cum_returns * weights[np.newaxis, np.newaxis, :]).sum(axis=2)
    return (1.0 + port_cum)[:, :, np.newaxis]


# ---------------------------------------------------------------------------
# Core metric functions (operate on terminal_returns of shape (n_sims, k))
# ---------------------------------------------------------------------------

def compute_var(term_ret: np.ndarray, confidence: float) -> np.ndarray:
    """
    Value at Risk: the confidence-th worst percentile of terminal returns.

    Returns shape (k,) — negative values indicate losses.
    VaR at 95% is the 5th percentile of the distribution.
    """
    return np.percentile(term_ret, (1.0 - confidence) * 100.0, axis=0)


def compute_cvar(term_ret: np.ndarray, confidence: float) -> np.ndarray:
    """
    Conditional Value at Risk (Expected Shortfall): mean return of all
    simulations falling at or below VaR.

    Returns shape (k,) — more negative than VaR, characterises tail severity.
    """
    var = compute_var(term_ret, confidence)
    k = term_ret.shape[1]
    cvar = np.zeros(k)
    for i in range(k):
        tail = term_ret[:, i][term_ret[:, i] <= var[i]]
        cvar[i] = float(tail.mean()) if len(tail) > 0 else var[i]
    return cvar


def compute_max_drawdown(paths: np.ndarray) -> tuple:
    """
    Per-simulation maximum peak-to-trough decline for each stock/portfolio.

    Parameters
    ----------
    paths : shape (n_sims, n_steps + 1, k)

    Returns
    -------
    expected_mdd   : shape (k,), mean max drawdown across simulations (negative)
    worst_case_mdd : shape (k,), 5th percentile — the worst 5% of simulations
    """
    running_peak = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_peak) / running_peak
    mdd_per_sim = drawdowns.min(axis=1)                          # (n_sims, k)
    expected_mdd = mdd_per_sim.mean(axis=0)
    worst_case_mdd = np.percentile(mdd_per_sim, 5.0, axis=0)    # 5th percentile
    return expected_mdd, worst_case_mdd


def compute_prob_loss(term_ret: np.ndarray) -> np.ndarray:
    """Fraction of simulations ending below the starting price. Shape (k,)."""
    return (term_ret < 0.0).mean(axis=0)


def compute_prob_target(term_ret: np.ndarray, target: float) -> np.ndarray:
    """Fraction of simulations exceeding the target return. Shape (k,)."""
    return (term_ret >= target).mean(axis=0)


def compute_skewness(term_ret: np.ndarray) -> np.ndarray:
    """
    Skewness of the terminal return distribution. Shape (k,).
    Negative skew indicates a longer left tail — bad outcomes are more extreme
    than good ones, even if the median is positive.
    """
    return stats.skew(term_ret, axis=0)


def compute_time_to_recovery(paths: np.ndarray) -> np.ndarray:
    """
    Average number of steps to recover from the maximum drawdown trough,
    per stock/portfolio. Shape (k,).

    For each simulation:
      1. Find the trough (time of minimum price)
      2. Find the running peak just before the trough
      3. Count steps after the trough until price first returns to that peak

    Returns NaN for stocks where fewer than 10% of simulations recover
    within the horizon — indicates the drawdown is unlikely to be temporary.
    """
    n_sims, n_steps_p1, k = paths.shape
    avg_recovery = np.full(k, np.nan)

    for i in range(k):
        stock_paths = paths[:, :, i]
        recovery_times = []

        for sim in range(n_sims):
            path = stock_paths[sim]
            trough_t = int(np.argmin(path))
            if trough_t == 0:
                continue
            peak_before = path[:trough_t].max()
            post_trough = path[trough_t:]
            recovered = np.where(post_trough >= peak_before)[0]
            if len(recovered) > 0:
                recovery_times.append(int(recovered[0]))

        recovery_rate = len(recovery_times) / n_sims
        if recovery_rate >= 0.10:
            avg_recovery[i] = float(np.mean(recovery_times))

    return avg_recovery


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _metrics_from_paths(
    paths: np.ndarray,
    confidence_levels: list,
    target_return: float,
) -> dict:
    """
    Compute all simulation-derived risk metrics from a paths array.
    Works for both per-stock (k = n_stocks) and portfolio (k = 1) paths.
    """
    term_ret = terminal_returns(paths)
    expected_mdd, worst_case_mdd = compute_max_drawdown(paths)
    prob_loss = compute_prob_loss(term_ret)
    prob_target = compute_prob_target(term_ret, target_return)
    skew = compute_skewness(term_ret)
    recovery = compute_time_to_recovery(paths)

    metrics = {}
    for idx in range(paths.shape[2]):
        tr = term_ret[:, idx: idx + 1]
        entry = {
            "prob_loss": float(prob_loss[idx]),
            f"prob_return_above_{int(target_return * 100)}pct": float(prob_target[idx]),
            "expected_max_drawdown": float(expected_mdd[idx]),
            "worst_case_max_drawdown_p95": float(worst_case_mdd[idx]),
            "skewness": float(skew[idx]),
            "avg_recovery_days": (
                None if np.isnan(recovery[idx]) else float(recovery[idx])
            ),
        }
        for conf in confidence_levels:
            label = int(conf * 100)
            var = float(compute_var(tr, conf)[0])
            cvar = float(compute_cvar(tr, conf)[0])
            tail_ratio = float(cvar / var) if abs(var) > 1e-10 else 1.0
            entry[f"var_{label}"] = var
            entry[f"cvar_{label}"] = cvar
            entry[f"tail_risk_ratio_{label}"] = tail_ratio

        metrics[idx] = entry

    return metrics


def compute_stock_metrics(
    paths: np.ndarray,
    tickers: list,
    sortino_historical: dict,
    target_return: float = 0.10,
    confidence_levels: list = None,
) -> dict:
    """
    Compute all risk metrics for each individual stock.

    Parameters
    ----------
    paths               : shape (n_sims, n_steps + 1, n_stocks)
    tickers             : list of ticker symbols aligned with last axis of paths
    sortino_historical  : {ticker: float} from model.fit_params()
    target_return       : probability of exceeding this return is reported
    confidence_levels   : list of floats e.g. [0.95, 0.99]

    Returns
    -------
    dict: {ticker: {metric_name: value, ...}}
    """
    if confidence_levels is None:
        confidence_levels = DEFAULT_CONFIDENCE_LEVELS

    raw = _metrics_from_paths(paths, confidence_levels, target_return)

    result = {}
    for i, ticker in enumerate(tickers):
        entry = raw[i]
        entry["sortino_ratio_historical_252d"] = sortino_historical.get(ticker)
        result[ticker] = entry

    return result


def compute_portfolio_metrics(
    paths: np.ndarray,
    weights: np.ndarray,
    tickers: list,
    returns_historical: pd.DataFrame,
    target_return: float = 0.10,
    confidence_levels: list = None,
) -> dict:
    """
    Compute portfolio-level risk metrics including diversification benefit
    and per-stock risk contribution.

    Parameters
    ----------
    paths                : shape (n_sims, n_steps + 1, n_stocks)
    weights              : shape (n_stocks,), must sum to 1.0
    tickers              : list of ticker symbols
    returns_historical   : DataFrame of historical daily log returns (for Sortino)
    target_return        : probability of exceeding this return is reported
    confidence_levels    : list of floats e.g. [0.95, 0.99]

    Returns
    -------
    dict of portfolio-level metrics
    """
    if confidence_levels is None:
        confidence_levels = DEFAULT_CONFIDENCE_LEVELS

    port_paths = aggregate_portfolio_paths(paths, weights)
    raw = _metrics_from_paths(port_paths, confidence_levels, target_return)
    result = raw[0]

    # Historical portfolio Sortino
    port_hist_returns = (returns_historical * weights).sum(axis=1)
    ann_mean = port_hist_returns.mean() * 252
    downside = port_hist_returns[port_hist_returns < 0].std() * np.sqrt(252)
    result["sortino_ratio_historical_252d"] = (
        float(ann_mean / downside) if downside > 1e-10 else None
    )

    # Diversification benefit and risk contribution per confidence level
    term_ret_stocks = terminal_returns(paths)    # (n_sims, n_stocks)
    term_ret_port = terminal_returns(port_paths)  # (n_sims, 1)

    for conf in confidence_levels:
        label = int(conf * 100)

        # Diversification benefit:
        # positive value = portfolio VaR is less severe than weighted sum of individual VaRs
        individual_vars = compute_var(term_ret_stocks, conf)           # (n_stocks,)
        weighted_sum_var = float((individual_vars * weights).sum())
        portfolio_var = float(compute_var(term_ret_port, conf)[0])
        result[f"diversification_benefit_{label}"] = float(
            portfolio_var - weighted_sum_var
        )

        # Risk contribution per stock:
        # each stock's weighted average return in the portfolio's tail scenarios
        tail_mask = term_ret_port[:, 0] <= portfolio_var
        risk_contrib = {}
        if tail_mask.any():
            for i, ticker in enumerate(tickers):
                stock_tail = term_ret_stocks[tail_mask, i]
                risk_contrib[ticker] = float(weights[i] * stock_tail.mean())
        result[f"risk_contribution_per_stock_{label}"] = risk_contrib

    return result