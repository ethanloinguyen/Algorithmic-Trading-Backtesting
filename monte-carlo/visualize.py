"""
visualize.py
------------
Generates charts from a Monte Carlo risk result produced by mc_engine.py.

Requires the result to have been produced with include_simulation_data=True
so that the raw paths and params arrays are available.

Five chart types are provided:

  1. Fan Chart          — simulated price paths with percentile bands per stock
  2. Return Distributions — terminal return histograms with VaR / CVaR marked
  3. Risk Metrics Comparison — VaR, CVaR, max drawdown across all stocks
  4. Portfolio Decomposition — risk contribution per stock + diversification benefit
  5. Correlation Heatmap — estimated cross-stock correlation matrix

Usage
-----
    from mc_engine import run_portfolio_risk
    from visualize import plot_all

    result = run_portfolio_risk(
        tickers=['AAPL', 'MSFT', 'NVDA', 'JPM', 'JNJ'],
        horizon_days=63,
        n_sims=1000,
        include_simulation_data=True,
    )
    plot_all(result, output_dir='charts')
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from risk import aggregate_portfolio_paths, terminal_returns

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
BG          = "#ffffff"
PANEL_BG    = "#f7f7f7"
SPINE_CLR   = "#cccccc"
TICK_CLR    = "#555555"
TEXT_CLR    = "#111111"
BAND_CLR    = "#1a56db"
PATH_CLR    = "#aaaaaa"
ACTUAL_CLR  = "#e02020"
POSITIVE    = "#16a34a"
NEGATIVE    = "#dc2626"
NEUTRAL     = "#6b7280"
ANNOT_BG    = "#ffffff"
ANNOT_BD    = "#cccccc"

FONT = {"fontfamily": "monospace"}


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_color(SPINE_CLR)
    ax.tick_params(colors=TICK_CLR, labelsize=8)
    ax.xaxis.label.set_color(TICK_CLR)
    ax.yaxis.label.set_color(TICK_CLR)


def _save(fig, path):
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved -> {path}")


def _require_simulation_data(result: dict):
    if "_paths" not in result:
        raise ValueError(
            "result must be produced with include_simulation_data=True "
            "to use this visualisation."
        )


# ---------------------------------------------------------------------------
# Chart 1 — Fan Chart
# ---------------------------------------------------------------------------

def plot_fan_charts(result: dict, output_dir: str = "charts") -> str:
    """
    Simulated price path fan charts for every stock plus the portfolio.

    Shows 50 individual sample paths (grey) with percentile band overlays:
      - Outer band (5th–95th percentile): light blue fill
      - Inner band (16th–84th percentile): medium blue fill
      - Median path (50th percentile): solid blue line

    VaR 95% is annotated on each panel.
    """
    _require_simulation_data(result)

    paths      = result["_paths"]           # (n_sims, n_steps+1, n_stocks)
    weights    = result["_weights_arr"]
    tickers    = result["tickers"]
    horizon    = result["horizon_days"]

    port_paths = aggregate_portfolio_paths(paths, weights)
    all_paths  = np.concatenate([paths, port_paths], axis=2)
    all_labels = tickers + ["Portfolio"]

    n_panels = len(all_labels)
    ncols    = min(3, n_panels)
    nrows    = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4 * nrows),
        facecolor=BG,
        squeeze=False,
    )
    fig.suptitle(
        "Monte Carlo Simulation — Price Path Fan Charts",
        color=TEXT_CLR, fontsize=13, fontweight="bold", **FONT, y=1.01,
    )

    x = np.arange(horizon + 1)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(all_paths.shape[0], size=min(50, all_paths.shape[0]), replace=False)

    all_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    for ax in all_axes[n_panels:]:
        ax.set_visible(False)

    for idx, (label, ax) in enumerate(zip(all_labels, all_axes)):
        _style_ax(ax)
        stock_paths = all_paths[:, :, idx]   # (n_sims, n_steps+1)

        # Sample paths
        for si in sample_idx:
            ax.plot(x, stock_paths[si] - 1, color=PATH_CLR, lw=0.4, alpha=0.5, zorder=1)

        # Percentile bands
        p5   = np.percentile(stock_paths, 5,  axis=0) - 1
        p16  = np.percentile(stock_paths, 16, axis=0) - 1
        p50  = np.percentile(stock_paths, 50, axis=0) - 1
        p84  = np.percentile(stock_paths, 84, axis=0) - 1
        p95  = np.percentile(stock_paths, 95, axis=0) - 1

        ax.fill_between(x, p5,  p95,  alpha=0.12, color=BAND_CLR, zorder=2)
        ax.fill_between(x, p16, p84,  alpha=0.25, color=BAND_CLR, zorder=3)
        ax.plot(x, p50, color=BAND_CLR, lw=2.0, label="Median", zorder=4)
        ax.axhline(0, color=SPINE_CLR, lw=0.8, ls="--", zorder=1)

        # VaR annotation
        if label == "Portfolio":
            var_val = result["portfolio"].get("var_95")
        else:
            var_val = result["per_stock"][label].get("var_95")

        if var_val is not None:
            ax.text(
                0.03, 0.05,
                f"VaR 95%: {var_val:.1%}",
                transform=ax.transAxes, fontsize=8, color=NEGATIVE,
                **FONT,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=ANNOT_BG,
                          edgecolor=ANNOT_BD, alpha=0.9),
            )

        color = BAND_CLR if label == "Portfolio" else TEXT_CLR
        ax.set_title(label, color=color, fontsize=10, fontweight="bold", **FONT)
        ax.set_xlabel("Trading Days", fontsize=8, **FONT)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "1_fan_charts.png")
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Chart 2 — Terminal Return Distributions
# ---------------------------------------------------------------------------

def plot_return_distributions(result: dict, output_dir: str = "charts") -> str:
    """
    Histogram of terminal returns for every stock plus the portfolio.

    VaR 95% is marked as a red vertical line.
    The CVaR region (mean of all returns left of VaR) is shaded in red.
    The probability of loss is annotated on each panel.
    """
    _require_simulation_data(result)

    paths   = result["_paths"]
    weights = result["_weights_arr"]
    tickers = result["tickers"]

    port_paths  = aggregate_portfolio_paths(paths, weights)
    all_paths   = np.concatenate([paths, port_paths], axis=2)
    all_labels  = tickers + ["Portfolio"]

    n_panels = len(all_labels)
    ncols    = min(3, n_panels)
    nrows    = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4 * nrows),
        facecolor=BG,
        squeeze=False,
    )
    fig.suptitle(
        "Terminal Return Distributions with VaR and CVaR",
        color=TEXT_CLR, fontsize=13, fontweight="bold", **FONT, y=1.01,
    )

    term = terminal_returns(all_paths)   # (n_sims, n_stocks+1)

    all_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    for ax in all_axes[n_panels:]:
        ax.set_visible(False)

    for idx, (label, ax) in enumerate(zip(all_labels, all_axes)):
        _style_ax(ax)
        r = term[:, idx]

        if label == "Portfolio":
            metrics = result["portfolio"]
        else:
            metrics = result["per_stock"][label]

        var  = metrics.get("var_95", np.percentile(r, 5))
        cvar = metrics.get("cvar_95", r[r <= var].mean())

        # Histogram
        ax.hist(r, bins=60, color=BAND_CLR, alpha=0.6, edgecolor="none", zorder=2)

        # CVaR shaded region
        ax.hist(
            r[r <= var], bins=30,
            color=NEGATIVE, alpha=0.7, edgecolor="none", zorder=3,
        )

        # VaR vertical line
        ax.axvline(var, color=NEGATIVE, lw=2.0, ls="--", zorder=4, label=f"VaR 95%: {var:.1%}")

        # CVaR vertical line
        ax.axvline(cvar, color="#7f1d1d", lw=1.5, ls=":", zorder=4, label=f"CVaR 95%: {cvar:.1%}")

        ax.axvline(0, color=SPINE_CLR, lw=0.8, zorder=1)

        # Annotations
        prob_loss = metrics.get("prob_loss", (r < 0).mean())
        ax.text(
            0.97, 0.95,
            f"P(loss): {prob_loss:.1%}\nVaR 95%:  {var:.1%}\nCVaR 95%: {cvar:.1%}",
            transform=ax.transAxes, fontsize=7.5, color=TEXT_CLR,
            ha="right", va="top", **FONT,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=ANNOT_BG,
                      edgecolor=ANNOT_BD, alpha=0.92),
        )

        color = BAND_CLR if label == "Portfolio" else TEXT_CLR
        ax.set_title(label, color=color, fontsize=10, fontweight="bold", **FONT)
        ax.set_xlabel("Cumulative Return", fontsize=8, **FONT)
        ax.set_ylabel("Frequency", fontsize=8, **FONT)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.legend(fontsize=7, framealpha=0.9, facecolor=ANNOT_BG,
                  edgecolor=ANNOT_BD, labelcolor=TEXT_CLR)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "2_return_distributions.png")
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Chart 3 — Risk Metrics Comparison
# ---------------------------------------------------------------------------

def plot_risk_comparison(result: dict, output_dir: str = "charts") -> str:
    """
    Grouped bar chart comparing VaR 95%, CVaR 95%, and Expected Max Drawdown
    across all individual stocks and the portfolio.

    Lets users immediately see which holdings dominate the risk profile.
    """
    tickers = result["tickers"]
    labels  = tickers + ["Portfolio"]

    var_vals  = []
    cvar_vals = []
    mdd_vals  = []

    for t in tickers:
        m = result["per_stock"][t]
        var_vals.append(abs(m.get("var_95", 0)))
        cvar_vals.append(abs(m.get("cvar_95", 0)))
        mdd_vals.append(abs(m.get("expected_max_drawdown", 0)))

    pm = result["portfolio"]
    var_vals.append(abs(pm.get("var_95", 0)))
    cvar_vals.append(abs(pm.get("cvar_95", 0)))
    mdd_vals.append(abs(pm.get("expected_max_drawdown", 0)))

    x     = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5), facecolor=BG)
    fig.suptitle(
        "Risk Metrics Comparison",
        color=TEXT_CLR, fontsize=13, fontweight="bold", **FONT,
    )
    _style_ax(ax)

    bars1 = ax.bar(x - width, var_vals,  width, label="VaR 95%",               color="#1a56db", alpha=0.85)
    bars2 = ax.bar(x,         cvar_vals, width, label="CVaR 95%",              color="#e02020", alpha=0.85)
    bars3 = ax.bar(x + width, mdd_vals,  width, label="Exp. Max Drawdown",     color="#f59e0b", alpha=0.85)

    # Value labels on bars
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.1%}", ha="center", va="bottom",
                fontsize=7, color=TEXT_CLR, **FONT,
            )

    # Highlight portfolio bars with a border
    n = len(labels) - 1
    for bars in (bars1, bars2, bars3):
        bars[n].set_edgecolor(TEXT_CLR)
        bars[n].set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, **FONT, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Loss magnitude (absolute value)", fontsize=9, **FONT)
    ax.legend(fontsize=8, framealpha=0.9, facecolor=ANNOT_BG,
              edgecolor=ANNOT_BD, labelcolor=TEXT_CLR)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "3_risk_comparison.png")
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Chart 4 — Portfolio Decomposition
# ---------------------------------------------------------------------------

def plot_portfolio_decomposition(result: dict, output_dir: str = "charts") -> str:
    """
    Two-panel portfolio-level chart:

    Left  — Risk contribution per stock: each holding's weighted average loss
            in the portfolio's worst 5% of simulations. Shows which positions
            are driving tail risk.

    Right — Diversification benefit: side-by-side comparison of the weighted
            sum of individual VaRs vs the actual portfolio VaR, illustrating
            the risk reduction from holding a correlated basket.
    """
    tickers = result["tickers"]
    pm      = result["portfolio"]
    weights = result["weights"]

    contrib = pm.get("risk_contribution_per_stock_95", {})
    contrib_vals  = [abs(contrib.get(t, 0)) for t in tickers]
    contrib_total = sum(contrib_vals) or 1.0
    contrib_pct   = [v / contrib_total for v in contrib_vals]

    div_benefit   = pm.get("diversification_benefit_95", 0)
    portfolio_var = abs(pm.get("var_95", 0))
    naive_var     = portfolio_var + div_benefit   # = weighted sum of individual VaRs

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(13, 5), facecolor=BG
    )
    fig.suptitle(
        "Portfolio Risk Decomposition",
        color=TEXT_CLR, fontsize=13, fontweight="bold", **FONT,
    )

    # Left: horizontal bar chart of risk contribution
    _style_ax(ax_left)
    colours = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(tickers)))
    bars = ax_left.barh(tickers, contrib_pct, color=colours, edgecolor="none")

    for bar, pct in zip(bars, contrib_pct):
        ax_left.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1%}", va="center", fontsize=8, color=TEXT_CLR, **FONT,
        )

    ax_left.set_xlabel("Share of Portfolio Tail Loss (VaR 95%)", fontsize=9, **FONT)
    ax_left.set_title("Risk Contribution per Stock", fontsize=10,
                      fontweight="bold", color=TEXT_CLR, **FONT)
    ax_left.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_left.invert_yaxis()

    # Right: diversification benefit comparison
    _style_ax(ax_right)
    bar_labels = ["Naive\n(sum of individual VaRs)", "Portfolio\n(actual VaR)"]
    bar_values = [naive_var, portfolio_var]
    bar_colors = [NEUTRAL, BAND_CLR]

    bars2 = ax_right.bar(bar_labels, bar_values, width=0.45,
                         color=bar_colors, alpha=0.85, edgecolor="none")

    for bar, val in zip(bars2, bar_values):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2, val + 0.002,
            f"{val:.1%}", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=TEXT_CLR, **FONT,
        )

    # Annotate the benefit arrow
    if div_benefit > 0:
        ax_right.annotate(
            f"Diversification\nbenefit: {div_benefit:.1%}",
            xy=(1, portfolio_var), xytext=(1, naive_var),
            xycoords="data", textcoords="data",
            arrowprops=dict(arrowstyle="<->", color=POSITIVE, lw=1.5),
            ha="center", fontsize=9, color=POSITIVE, **FONT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=ANNOT_BG,
                      edgecolor=ANNOT_BD, alpha=0.92),
        )

    ax_right.set_ylabel("VaR 95% magnitude", fontsize=9, **FONT)
    ax_right.set_title("Diversification Benefit", fontsize=10,
                       fontweight="bold", color=TEXT_CLR, **FONT)
    ax_right.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "4_portfolio_decomposition.png")
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Chart 5 — Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(result: dict, output_dir: str = "charts") -> str:
    """
    Annotated heatmap of the estimated cross-stock correlation matrix.

    Shows users which holdings are highly correlated (and therefore provide
    less diversification benefit than the weights alone suggest) versus which
    holdings genuinely diversify risk.
    """
    _require_simulation_data(result)

    corr    = result["_params"]["corr"]
    tickers = result["tickers"]
    n       = len(tickers)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)), facecolor=BG)
    fig.suptitle(
        "Cross-Stock Correlation Matrix (252-day lookback)",
        color=TEXT_CLR, fontsize=13, fontweight="bold", **FONT,
    )
    _style_ax(ax)

    cmap = plt.cm.RdYlGn
    im   = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # Cell annotations
    for i in range(n):
        for j in range(n):
            val   = corr[i, j]
            color = "white" if abs(val) > 0.6 else TEXT_CLR
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, **FONT)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=9, **FONT)
    ax.set_yticklabels(tickers, fontsize=9, **FONT)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors=TICK_CLR)
    cbar.set_label("Correlation", fontsize=9, color=TICK_CLR, **FONT)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "5_correlation_heatmap.png")
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Chart 6 — Tail Highlight (presentation explainer)
# ---------------------------------------------------------------------------

def plot_tail_highlight(
    result: dict,
    output_dir: str = "charts",
    subject: str = "Portfolio",
) -> str:
    """
    Single-panel presentation chart that makes the purpose of Monte Carlo
    immediately clear to any audience.

    All 1,000 simulated paths are drawn as thin grey lines. The worst 5%
    of paths — those ending below the VaR 95% threshold — are overlaid in
    red. An annotated arrow points directly to the tail region with the
    label 'This is what we measure', and the VaR value is called out
    explicitly on the threshold line.

    Parameters
    ----------
    result  : dict from run_portfolio_risk(include_simulation_data=True)
    subject : ticker symbol to plot, or 'Portfolio' (default)
    """
    _require_simulation_data(result)

    raw_paths   = result["_paths"]          # (n_sims, n_steps+1, n_stocks)
    weights_arr = result["_weights_arr"]
    tickers     = result["tickers"]
    horizon     = result["horizon_days"]
    n_sims      = result["n_simulations"]

    # Resolve subject paths and metrics
    if subject == "Portfolio":
        port = aggregate_portfolio_paths(raw_paths, weights_arr)
        stock_paths = port[:, :, 0]         # (n_sims, n_steps+1)
        metrics     = result["portfolio"]
        title_label = "Portfolio"
    elif subject in tickers:
        idx         = tickers.index(subject)
        stock_paths = raw_paths[:, :, idx]
        metrics     = result["per_stock"][subject]
        title_label = subject
    else:
        raise ValueError(f"subject '{subject}' not found. Choose from: {['Portfolio'] + tickers}")

    # Convert to cumulative return (start at 0%)
    cum_returns = stock_paths - 1.0         # (n_sims, n_steps+1)
    terminal    = cum_returns[:, -1]
    var_95      = metrics.get("var_95", float(np.percentile(terminal, 5)))
    cvar_95     = metrics.get("cvar_95", float(terminal[terminal <= var_95].mean()))

    # Identify tail paths (worst 5%)
    tail_mask   = terminal <= var_95
    safe_mask   = ~tail_mask

    x = np.arange(horizon + 1)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    _style_ax(ax)

    # --- All safe paths (grey, very thin) ---
    for i in np.where(safe_mask)[0]:
        ax.plot(x, cum_returns[i], color=PATH_CLR, lw=0.35, alpha=0.25, zorder=1)

    # --- Tail paths (red, slightly thicker) ---
    for i in np.where(tail_mask)[0]:
        ax.plot(x, cum_returns[i], color=NEGATIVE, lw=0.6, alpha=0.55, zorder=2)

    # --- Median path ---
    median_path = np.median(cum_returns, axis=0)
    ax.plot(x, median_path, color=BAND_CLR, lw=2.2, zorder=4, label="Median path")

    # --- VaR threshold line ---
    ax.axhline(
        var_95, color=NEGATIVE, lw=1.6, ls="--", zorder=3,
        label=f"VaR 95%  {var_95:.1%}",
    )

    # --- Zero line ---
    ax.axhline(0, color=SPINE_CLR, lw=0.8, ls=":", zorder=1)

    # --- Shade the tail terminal region ---
    # Vertical band at the last step highlighting where tail paths land
    tail_min = float(terminal[tail_mask].min()) if tail_mask.any() else var_95
    ax.fill_betweenx(
        [tail_min * 1.05, var_95],
        horizon - 0.5, horizon + 0.5,
        color=NEGATIVE, alpha=0.18, zorder=2,
    )

    # --- Arrow annotation: "This is what we measure" ---
    # Arrow tip points into the tail path cluster, roughly halfway through horizon
    mid_x      = horizon * 0.65
    tail_paths_mid = cum_returns[tail_mask, int(mid_x)] if tail_mask.any() else [var_95]
    arrow_tip_y   = float(np.median(tail_paths_mid))
    arrow_label_y = arrow_tip_y - 0.12  # label sits below the tip

    ax.annotate(
        "This is what\nwe measure",
        xy=(mid_x, arrow_tip_y),
        xytext=(mid_x - horizon * 0.18, arrow_label_y),
        fontsize=10, color=NEGATIVE, fontweight="bold", **FONT,
        ha="center", va="top",
        arrowprops=dict(
            arrowstyle="-|>",
            color=NEGATIVE,
            lw=1.8,
            connectionstyle="arc3,rad=-0.25",
        ),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=ANNOT_BG,
            edgecolor=NEGATIVE,
            alpha=0.95,
            linewidth=1.5,
        ),
        zorder=6,
    )

    # --- VaR + CVaR summary box ---
    n_tail = int(tail_mask.sum())
    ax.text(
        0.97, 0.97,
        (
            f"Simulations:  {n_sims:,}\n"
            f"Tail paths:   {n_tail} ({n_tail/n_sims:.0%})\n"
            f"VaR  95%:  {var_95:.1%}\n"
            f"CVaR 95%:  {cvar_95:.1%}"
        ),
        transform=ax.transAxes,
        fontsize=9, color=TEXT_CLR, **FONT,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=ANNOT_BG,
                  edgecolor=ANNOT_BD, alpha=0.95),
        zorder=6,
    )

    # --- Legend ---
    ax.legend(
        fontsize=9, framealpha=0.9, facecolor=ANNOT_BG,
        edgecolor=ANNOT_BD, labelcolor=TEXT_CLR, loc="upper left",
    )

    ax.set_title(
        f"Monte Carlo Risk Simulation — {title_label}  "
        f"({horizon}-day horizon, {n_sims:,} simulations)",
        color=TEXT_CLR, fontsize=12, fontweight="bold", **FONT, pad=10,
    )
    ax.set_xlabel("Trading Days", fontsize=10, **FONT)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Cumulative Return", fontsize=10, **FONT)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "0_tail_highlight.png")
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Generate all charts
# ---------------------------------------------------------------------------

def plot_all(result: dict, output_dir: str = "charts") -> list:
    """
    Generate all six charts and save them to output_dir.

    Parameters
    ----------
    result     : dict returned by run_portfolio_risk(include_simulation_data=True)
    output_dir : directory to save PNG files (created if it does not exist)

    Returns
    -------
    List of file paths for the generated charts.
    """
    print(f"\nGenerating charts -> {os.path.abspath(output_dir)}/")
    paths = [
        plot_tail_highlight(result, output_dir),
        plot_fan_charts(result, output_dir),
        plot_return_distributions(result, output_dir),
        plot_risk_comparison(result, output_dir),
        plot_portfolio_decomposition(result, output_dir),
        plot_correlation_heatmap(result, output_dir),
    ]
    print(f"\nAll charts saved ({len(paths)} files).\n")
    return paths


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from mc_engine import run_portfolio_risk

    result = run_portfolio_risk(
        tickers=["AAPL", "MSFT", "NVDA", "JPM", "JNJ", "GOOG"],
        horizon_days=63,
        n_sims=1000,
        include_simulation_data=True,
    )
    plot_all(result, output_dir="charts")