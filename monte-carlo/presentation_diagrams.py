"""
presentation_diagrams.py
------------------------
Standalone presentation diagrams for the Monte Carlo risk feature.
Requires no real data — all diagrams are purely illustrative.

Generates four PNG files:
  diag_1_data_input.png       — data source decision flow
  diag_2_simulation.png       — tensor shape collapsing to fan chart
  diag_3_pipeline.png         — end-to-end pipeline architecture
  diag_4_output_structure.png — JSON output + frontend wireframe

Run from the monte-carlo directory:
    python presentation_diagrams.py
"""

import os
import warnings

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from scipy import stats

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

OUTPUT_DIR = "charts"

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
BG          = "#ffffff"
PANEL_BG    = "#f5f7fa"
DATA_CLR    = "#1a56db"   # blue   — data / input
COMPUTE_CLR = "#16a34a"   # green  — computation
OUTPUT_CLR  = "#d97706"   # amber  — output
NEUTRAL_CLR = "#6b7280"   # grey   — neutral boxes
NEGATIVE    = "#dc2626"   # red    — risk / warning
TEXT_CLR    = "#111111"
LIGHT_TEXT  = "#6b7280"
WHITE       = "#ffffff"
SPINE_CLR   = "#cccccc"
FONT        = {"fontfamily": "monospace"}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _box(ax, cx, cy, w, h, text, fill, text_color=WHITE,
         fontsize=9, linewidth=1.5, zorder=3, alpha=1.0):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.03",
        facecolor=fill, edgecolor=WHITE,
        linewidth=linewidth, zorder=zorder, alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold", **FONT, zorder=zorder + 1,
            multialignment="center")


def _diamond(ax, cx, cy, w, h, text, fill, text_color=WHITE, fontsize=9, zorder=3):
    pts = np.array([
        [cx,       cy + h / 2],
        [cx + w / 2, cy],
        [cx,       cy - h / 2],
        [cx - w / 2, cy],
    ])
    patch = Polygon(pts, closed=True,
                    facecolor=fill, edgecolor=WHITE,
                    linewidth=1.5, zorder=zorder)
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold", **FONT, zorder=zorder + 1,
            multialignment="center")


def _arrow(ax, x1, y1, x2, y2, color=NEUTRAL_CLR, lw=1.8,
           label="", label_side="top", connectionstyle="arc3,rad=0.0"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            connectionstyle=connectionstyle,
        ),
        zorder=5,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        offset = 0.03 if label_side == "top" else -0.03
        ax.text(mx, my + offset, label,
                ha="center", va="bottom" if label_side == "top" else "top",
                fontsize=7.5, color=color, **FONT, zorder=6)


def _save(fig, filename, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Diagram 1 — Data Input Flow
# ---------------------------------------------------------------------------

def diag_data_input(output_dir=OUTPUT_DIR) -> str:
    """
    Two-column diagram:
      Left  — user inputs (ticker list, weights, horizon, n_sims)
      Right — data source decision tree (BigQuery → yfinance fallback)
    """
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor(BG)

    fig.suptitle("Data Input — Sources and Decision Flow",
                 color=TEXT_CLR, fontsize=14, fontweight="bold", **FONT, y=0.97)

    # ── Left column: User Inputs ──────────────────────────────────────────────
    ax.text(1.8, 6.4, "USER INPUT", ha="center", fontsize=10,
            color=DATA_CLR, fontweight="bold", **FONT)

    inputs = [
        ("tickers", "['AAPL', 'MSFT', 'NVDA', ...]"),
        ("weights", "[0.25, 0.25, 0.20, ...]"),
        ("horizon_days", "63"),
        ("n_sims", "1000"),
    ]
    for i, (key, val) in enumerate(inputs):
        cy = 5.5 - i * 0.9
        _box(ax, 1.0, cy, 0.9, 0.55, key,   DATA_CLR, fontsize=8.5)
        _box(ax, 2.6, cy, 1.8, 0.55, val,   PANEL_BG,
             text_color=TEXT_CLR, fontsize=8)
        _arrow(ax, 1.45, cy, 1.65, cy, color=DATA_CLR, lw=1.2)

    # Brace / bracket to group inputs
    ax.annotate("", xy=(3.7, 3.1), xytext=(3.7, 5.8),
                arrowprops=dict(arrowstyle="-|>", color=DATA_CLR, lw=2.0))
    ax.text(3.85, 4.45, "passed to\nfetch_returns()",
            ha="left", va="center", fontsize=8,
            color=DATA_CLR, **FONT)

    # ── Divider ───────────────────────────────────────────────────────────────
    ax.axvline(4.8, color=SPINE_CLR, lw=1.0, ls="--", ymin=0.05, ymax=0.95)

    # ── Right column: Decision Tree ───────────────────────────────────────────
    ax.text(9.5, 6.4, "DATA SOURCE RESOLUTION", ha="center", fontsize=10,
            color=TEXT_CLR, fontweight="bold", **FONT)

    # Node positions (cx, cy)
    n_bq_query  = (9.5, 5.6)
    n_bq_check  = (9.5, 4.5)
    n_bq_ok     = (7.2, 3.3)
    n_yf        = (11.8, 3.3)
    n_yf_check  = (11.8, 2.3)
    n_merge     = (9.5, 1.3)

    _box(ax, *n_bq_query, 3.6, 0.6,
         "Query BigQuery\nyfinance_stocks_data.market_data",
         DATA_CLR, fontsize=8.5)

    _diamond(ax, *n_bq_check, 3.8, 0.8,
             "≥ 60 rows\nreturned?", DATA_CLR, fontsize=8.5)

    _box(ax, *n_bq_ok, 2.8, 0.6,
         "Use BigQuery\nlog_return column", COMPUTE_CLR, fontsize=8.5)

    _box(ax, *n_yf, 2.8, 0.6,
         "Fallback:\nyfinance.download()", NEUTRAL_CLR, fontsize=8.5)

    _diamond(ax, *n_yf_check, 2.8, 0.7,
             "≥ 60 rows?", NEUTRAL_CLR, fontsize=8.5)

    _box(ax, *n_merge, 4.2, 0.65,
         "Merge on date index  →  252-day returns DataFrame",
         OUTPUT_CLR, text_color=WHITE, fontsize=8.5)

    # Arrows
    _arrow(ax, 9.5, 5.3,  9.5, 4.9,  color=DATA_CLR)
    # Yes branch (left)
    _arrow(ax, 7.6, 4.5,  7.2, 3.62, color=COMPUTE_CLR,
           label="Yes", label_side="top")
    # No branch (right)
    _arrow(ax, 11.4, 4.5, 11.8, 3.62, color=NEUTRAL_CLR,
           label="No", label_side="top")
    _arrow(ax, 11.8, 2.97, 11.8, 2.65, color=NEUTRAL_CLR)
    # yfinance success path
    ax.annotate("", xy=(11.0, 1.3), xytext=(11.8, 1.97),
                arrowprops=dict(arrowstyle="-|>", color=NEUTRAL_CLR,
                                lw=1.8, connectionstyle="arc3,rad=0.0"))
    # BigQuery success to merge
    ax.annotate("", xy=(8.0, 1.3), xytext=(7.2, 2.97),
                arrowprops=dict(arrowstyle="-|>", color=COMPUTE_CLR,
                                lw=1.8, connectionstyle="arc3,rad=0.0"))

    # "Missing" label on yfinance fail
    ax.text(13.1, 2.3, "ticker\n→ missing[ ]",
            ha="left", va="center", fontsize=7.5,
            color=NEGATIVE, **FONT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                      edgecolor=NEGATIVE, alpha=0.9))
    ax.annotate("", xy=(13.1, 2.3), xytext=(12.6, 2.3),
                arrowprops=dict(arrowstyle="-|>", color=NEGATIVE, lw=1.2))

    ax.text(4.9, 3.5, "← primary\n    path",
            ha="left", va="center", fontsize=7.5, color=COMPUTE_CLR, **FONT)
    ax.text(4.9, 2.5, "← fallback\n    path",
            ha="left", va="center", fontsize=7.5, color=NEUTRAL_CLR, **FONT)

    plt.tight_layout()
    return _save(fig, "diag_1_data_input.png", output_dir)


# ---------------------------------------------------------------------------
# Diagram 2 — Simulation Mechanics (tensor → fan chart)
# ---------------------------------------------------------------------------

def diag_simulation_mechanics(output_dir=OUTPUT_DIR) -> str:
    """
    Left panel  — 2D perspective drawing of the 3D paths tensor
                  axes labelled: n_sims, n_steps, n_stocks
    Right panel — fan chart of a single stock's simulated paths
                  (synthetically generated — no real data needed)
    """
    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    fig.suptitle("Simulation Output — From Tensor to Fan Chart",
                 color=TEXT_CLR, fontsize=14, fontweight="bold", **FONT, y=0.98)

    ax_left  = fig.add_axes([0.03, 0.08, 0.40, 0.82])
    ax_right = fig.add_axes([0.55, 0.08, 0.43, 0.82])

    ax_left.set_xlim(0, 10)
    ax_left.set_ylim(0, 8)
    ax_left.axis("off")
    ax_left.set_facecolor(BG)

    # ── 3D cube (2-point perspective) ────────────────────────────────────────
    # Front face corners (bottom-left origin)
    ox, oy   = 1.5, 1.5
    fw, fh   = 4.0, 4.5      # front width, height
    dx, dy   = 2.2, 1.8      # perspective offset

    # Vertices
    fl = (ox,         oy)
    fr = (ox + fw,    oy)
    tr = (ox + fw,    oy + fh)
    tl = (ox,         oy + fh)
    bl = (ox + dx,    oy + dy)
    br = (ox + fw + dx, oy + dy)
    mr = (ox + fw + dx, oy + fh + dy)
    ml = (ox + dx,    oy + fh + dy)

    face_fill = DATA_CLR + "22"   # very transparent
    edge_kw   = dict(color=DATA_CLR, lw=1.8, zorder=2)

    # Front face
    front_pts = plt.Polygon([fl, fr, tr, tl], closed=True,
                             facecolor=DATA_CLR + "18", edgecolor=DATA_CLR,
                             lw=1.8, zorder=2)
    ax_left.add_patch(front_pts)

    # Top face
    top_pts = plt.Polygon([tl, tr, mr, ml], closed=True,
                           facecolor=DATA_CLR + "28", edgecolor=DATA_CLR,
                           lw=1.8, zorder=2)
    ax_left.add_patch(top_pts)

    # Right face
    right_pts = plt.Polygon([fr, br, mr, tr], closed=True,
                              facecolor=DATA_CLR + "20", edgecolor=DATA_CLR,
                              lw=1.8, zorder=2)
    ax_left.add_patch(right_pts)

    # Back edges (dashed)
    for p1, p2 in [(fl, bl), (bl, br), (bl, ml)]:
        ax_left.plot([p1[0], p2[0]], [p1[1], p2[1]],
                     color=DATA_CLR, lw=1.2, ls="--", zorder=1, alpha=0.5)

    # Axis labels
    ax_left.text(ox + fw / 2, oy - 0.45, "n_stocks  (5)",
                 ha="center", va="top", fontsize=9,
                 color=DATA_CLR, fontweight="bold", **FONT)

    ax_left.text(ox - 0.3, oy + fh / 2, "n_sims\n(1,000)",
                 ha="right", va="center", fontsize=9,
                 color=DATA_CLR, fontweight="bold", **FONT)

    ax_left.annotate(
        "n_steps\n(64)", xy=mr, xytext=(mr[0] + 0.5, mr[1] + 0.2),
        fontsize=9, color=DATA_CLR, fontweight="bold", **FONT, ha="left",
        arrowprops=dict(arrowstyle="-", color=DATA_CLR, lw=1.0),
    )

    ax_left.text(ox + fw / 2 + dx / 2, oy + fh / 2 + dy / 2,
                 "paths\narray",
                 ha="center", va="center", fontsize=11,
                 color=DATA_CLR, fontweight="bold", **FONT, alpha=0.9)

    # Array shape annotation
    ax_left.text(5.0, 0.5,
                 "shape: (1000 simulations  ×  64 time steps  ×  5 stocks)",
                 ha="center", va="bottom", fontsize=8.5,
                 color=LIGHT_TEXT, **FONT)

    # ── Centre arrow ─────────────────────────────────────────────────────────
    fig.text(0.475, 0.50,
             "select\nstock  i\n────────►",
             ha="center", va="center", fontsize=10,
             color=NEUTRAL_CLR, fontweight="bold", **FONT)

    # ── Right panel: fan chart of synthetic paths ─────────────────────────────
    ax_right.set_facecolor(PANEL_BG)
    for sp in ax_right.spines.values():
        sp.set_color(SPINE_CLR)
    ax_right.tick_params(colors=LIGHT_TEXT, labelsize=8)

    rng     = np.random.default_rng(7)
    n_paths = 120
    n_steps = 63
    dt      = 1 / 252
    sigma   = 0.22

    returns = rng.standard_t(5, size=(n_paths, n_steps)) / np.sqrt(5 / 3)
    returns *= sigma * np.sqrt(dt)
    cum     = np.cumprod(1 + returns, axis=1)
    cum     = np.hstack([np.ones((n_paths, 1)), cum]) - 1.0   # start at 0%

    x = np.arange(n_steps + 1)

    for i in range(n_paths):
        ax_right.plot(x, cum[i], color="#aaaaaa", lw=0.45, alpha=0.35, zorder=1)

    p5  = np.percentile(cum, 5,  axis=0)
    p16 = np.percentile(cum, 16, axis=0)
    p50 = np.percentile(cum, 50, axis=0)
    p84 = np.percentile(cum, 84, axis=0)
    p95 = np.percentile(cum, 95, axis=0)

    ax_right.fill_between(x, p5,  p95,  alpha=0.10, color=DATA_CLR, zorder=2)
    ax_right.fill_between(x, p16, p84,  alpha=0.22, color=DATA_CLR, zorder=3)
    ax_right.plot(x, p50, color=DATA_CLR, lw=2.2, zorder=4, label="Median")
    ax_right.axhline(0, color=SPINE_CLR, lw=0.8, ls=":", zorder=1)

    ax_right.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_right.set_xlabel("Trading Days", fontsize=9, color=LIGHT_TEXT, **FONT)
    ax_right.set_ylabel("Cumulative Return", fontsize=9, color=LIGHT_TEXT, **FONT)
    ax_right.set_title("Single Stock — 1,000 Simulated Paths",
                       fontsize=10, color=TEXT_CLR, fontweight="bold", **FONT)

    # Annotate bands
    mid = n_steps // 2
    ax_right.annotate(
        "5th – 95th\npercentile band",
        xy=(mid, p95[mid]),
        xytext=(mid + 8, p95[mid] + 0.04),
        fontsize=7.5, color=DATA_CLR, **FONT,
        arrowprops=dict(arrowstyle="-|>", color=DATA_CLR, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                  edgecolor=DATA_CLR, alpha=0.9),
    )
    ax_right.annotate(
        "16th – 84th\npercentile band",
        xy=(mid, p16[mid]),
        xytext=(mid + 8, p16[mid] - 0.06),
        fontsize=7.5, color=DATA_CLR + "cc", **FONT,
        arrowprops=dict(arrowstyle="-|>", color=DATA_CLR, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                  edgecolor=DATA_CLR, alpha=0.9),
    )

    return _save(fig, "diag_2_simulation.png", output_dir)


# ---------------------------------------------------------------------------
# Diagram 3 — Pipeline Architecture
# ---------------------------------------------------------------------------

def diag_pipeline_architecture(output_dir=OUTPUT_DIR) -> str:
    """
    Clean left-to-right five-box flow diagram.
    Blue = data, green = computation, amber = output.
    Labels on arrows describe what flows between stages.
    """
    fig, ax = plt.subplots(figsize=(16, 5.5), facecolor=BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_facecolor(BG)

    fig.suptitle("Monte Carlo Risk Pipeline — Architecture",
                 color=TEXT_CLR, fontsize=14, fontweight="bold", **FONT, y=0.97)

    # Box definitions: (cx, cy, w, h, title, subtitle, color)
    boxes = [
        (1.5,  2.75, 2.2, 2.6, "User\nInput",
         "tickers\nweights\nhorizon_days\nn_sims",           DATA_CLR),
        (4.5,  2.75, 2.2, 2.6, "data.py",
         "Fetch Returns\n─────────\nBigQuery\n+ yfinance fallback", DATA_CLR),
        (7.5,  2.75, 2.2, 2.6, "model.py",
         "Fit Parameters\n─────────\nvolatility\nt-dist dof\nCholesky", COMPUTE_CLR),
        (10.5, 2.75, 2.2, 2.6, "simulate.py",
         "Generate Paths\n─────────\n1,000 × horizon\n× n_stocks",   COMPUTE_CLR),
        (13.5, 2.75, 2.2, 2.6, "risk.py",
         "Compute Metrics\n─────────\nVaR  CVaR  MDD\nSortino  Skew", OUTPUT_CLR),
    ]

    for cx, cy, w, h, title, subtitle, color in boxes:
        # Outer coloured header box
        header = FancyBboxPatch(
            (cx - w / 2, cy + 0.1), w, h / 2 - 0.1,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="none", zorder=3,
        )
        ax.add_patch(header)
        ax.text(cx, cy + h / 4 + 0.1, title,
                ha="center", va="center", fontsize=11,
                color=WHITE, fontweight="bold", **FONT, zorder=4)

        # Lower body box (light)
        body = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h / 2 + 0.1,
            boxstyle="round,pad=0.04",
            facecolor=PANEL_BG, edgecolor=color,
            linewidth=1.5, zorder=3,
        )
        ax.add_patch(body)
        ax.text(cx, cy - h / 4, subtitle,
                ha="center", va="center", fontsize=7.5,
                color=TEXT_CLR, **FONT, zorder=4,
                multialignment="center")

    # Arrows + flow labels
    flow_labels = [
        "tickers,\nparams",
        "252-day\nlog returns",
        "sigma, dof,\nCholesky L",
        "(n_sims ×\nn_steps × k)\npaths array",
    ]
    for i, label in enumerate(flow_labels):
        x1 = boxes[i][0]     + boxes[i][2] / 2
        x2 = boxes[i + 1][0] - boxes[i + 1][2] / 2
        cy = 2.75
        _arrow(ax, x1 + 0.05, cy, x2 - 0.05, cy,
               color=NEUTRAL_CLR, lw=2.0, label=label, label_side="top")

    # Output arrow + box
    _arrow(ax, 14.6, 2.75, 15.3, 2.75, color=OUTPUT_CLR, lw=2.2)
    _box(ax, 15.75, 2.75, 0.7, 1.0, "JSON\nOutput",
         OUTPUT_CLR, fontsize=8.5)

    # Legend
    legend_items = [
        (DATA_CLR,    "Data / Input"),
        (COMPUTE_CLR, "Computation"),
        (OUTPUT_CLR,  "Output"),
    ]
    for i, (color, label) in enumerate(legend_items):
        lx = 0.3 + i * 2.8
        patch = FancyBboxPatch((lx, 0.2), 0.45, 0.35,
                                boxstyle="round,pad=0.03",
                                facecolor=color, edgecolor="none", zorder=3)
        ax.add_patch(patch)
        ax.text(lx + 0.6, 0.38, label, va="center", fontsize=8.5,
                color=TEXT_CLR, **FONT)

    return _save(fig, "diag_3_pipeline.png", output_dir)


# ---------------------------------------------------------------------------
# Diagram 4 — Output Structure (JSON + frontend wireframe)
# ---------------------------------------------------------------------------

def diag_output_structure(output_dir=OUTPUT_DIR) -> str:
    """
    Split layout:
      Left  — abbreviated JSON output with colour-coded field groups
      Right — wireframe mockup of the frontend portfolio risk panel
    """
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(16, 8), facecolor=BG,
        gridspec_kw={"width_ratios": [1, 1]},
    )
    fig.suptitle("Output Structure — JSON Response & Frontend Panel",
                 color=TEXT_CLR, fontsize=14, fontweight="bold", **FONT, y=0.98)

    # ── Left: annotated JSON ──────────────────────────────────────────────────
    ax_left.axis("off")
    ax_left.set_facecolor(PANEL_BG)
    ax_left.set_xlim(0, 10)
    ax_left.set_ylim(0, 10)

    ax_left.add_patch(FancyBboxPatch(
        (0.2, 0.2), 9.6, 9.4,
        boxstyle="round,pad=0.1",
        facecolor=PANEL_BG, edgecolor=SPINE_CLR, linewidth=1.5,
    ))

    ax_left.text(5.0, 9.5, "JSON Response (abbreviated)",
                 ha="center", fontsize=10, color=TEXT_CLR,
                 fontweight="bold", **FONT)

    json_lines = [
        ("{",                                              TEXT_CLR,   0),
        ('"tickers":   ["AAPL", "MSFT", "NVDA", ...],',  NEUTRAL_CLR, 1),
        ('"weights":   {"AAPL": 0.25, ...},',             NEUTRAL_CLR, 1),
        ('"horizon_days": 63,',                            NEUTRAL_CLR, 1),
        ('"per_stock": {',                                 TEXT_CLR,   1),
        ('"AAPL": {',                                      TEXT_CLR,   2),
        ('"var_95":   -0.142,',                            DATA_CLR,   3),
        ('"cvar_95":  -0.187,',                            DATA_CLR,   3),
        ('"tail_risk_ratio_95": 1.32,',                    DATA_CLR,   3),
        ('"expected_max_drawdown":       -0.118,',         COMPUTE_CLR, 3),
        ('"worst_case_max_drawdown_p95": -0.231,',         COMPUTE_CLR, 3),
        ('"prob_loss":  0.44,',                            OUTPUT_CLR, 3),
        ('"avg_recovery_days":  18.4,',                    OUTPUT_CLR, 3),
        ('"sortino_ratio_historical_252d": 0.81', NEUTRAL_CLR, 3),
        ("},  ...",                                         TEXT_CLR,   2),
        ("},",                                              TEXT_CLR,   1),
        ('"portfolio": {',                                 TEXT_CLR,   1),
        ('"var_95":   -0.098,',                            DATA_CLR,   2),
        ('"cvar_95":  -0.127,',                            DATA_CLR,   2),
        ('"diversification_benefit_95": 0.044,',           COMPUTE_CLR, 2),
        ('"risk_contribution_per_stock_95": {',            OUTPUT_CLR, 2),
        ('"AAPL": -0.031,  "MSFT": -0.022, ...', OUTPUT_CLR, 3),
        ("}",                                              OUTPUT_CLR, 2),
        ("}",                                              TEXT_CLR,   1),
        ("}",                                              TEXT_CLR,   0),
    ]

    y_start = 8.9
    line_h  = 0.32
    indent  = 0.35

    for i, (text, color, depth) in enumerate(json_lines):
        ax_left.text(
            0.5 + depth * indent, y_start - i * line_h,
            text, va="center", fontsize=7.8,
            color=color, **FONT,
        )

    # Colour legend
    for i, (color, label) in enumerate([
        (DATA_CLR,    "VaR / CVaR / Tail Risk"),
        (COMPUTE_CLR, "Drawdown / Diversification"),
        (OUTPUT_CLR,  "Probability / Contribution"),
    ]):
        lx, ly = 0.5, 0.85 - i * 0.38
        ax_left.add_patch(FancyBboxPatch(
            (lx, ly - 0.1), 0.3, 0.22,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="none",
        ))
        ax_left.text(lx + 0.45, ly, label, va="center",
                     fontsize=7.5, color=TEXT_CLR, **FONT)

    # ── Right: frontend wireframe ─────────────────────────────────────────────
    ax_right.axis("off")
    ax_right.set_facecolor(BG)
    ax_right.set_xlim(0, 10)
    ax_right.set_ylim(0, 10)

    ax_right.text(5.0, 9.5, "Frontend Panel (wireframe)",
                  ha="center", fontsize=10, color=TEXT_CLR,
                  fontweight="bold", **FONT)

    # Portfolio summary card
    ax_right.add_patch(FancyBboxPatch(
        (0.3, 7.0), 9.4, 2.3,
        boxstyle="round,pad=0.1",
        facecolor=DATA_CLR + "15", edgecolor=DATA_CLR,
        linewidth=1.5,
    ))
    ax_right.text(5.0, 9.1, "Portfolio Risk Summary",
                  ha="center", fontsize=9.5, color=DATA_CLR,
                  fontweight="bold", **FONT)

    summary_items = [
        ("VaR 95%", "-9.8%"),
        ("CVaR 95%", "-12.7%"),
        ("Exp. Drawdown", "-8.3%"),
        ("Div. Benefit", "+4.4%"),
    ]
    for i, (label, val) in enumerate(summary_items):
        cx = 1.5 + i * 2.3
        ax_right.add_patch(FancyBboxPatch(
            (cx - 0.95, 7.15), 1.9, 1.6,
            boxstyle="round,pad=0.08",
            facecolor=BG, edgecolor=SPINE_CLR, linewidth=1.0,
        ))
        ax_right.text(cx, 8.1, val, ha="center", va="center",
                      fontsize=12, color=DATA_CLR,
                      fontweight="bold", **FONT)
        ax_right.text(cx, 7.45, label, ha="center", va="center",
                      fontsize=7.5, color=LIGHT_TEXT, **FONT)

    # Individual stock cards
    ax_right.text(0.4, 6.7, "Individual Holdings", fontsize=9,
                  color=TEXT_CLR, fontweight="bold", **FONT)

    for i, (ticker, var, prob) in enumerate([
        ("AAPL", "-14.2%", "44%"),
        ("MSFT", "-12.8%", "41%"),
        ("NVDA", "-22.1%", "48%"),
    ]):
        cx = 1.7 + i * 3.0
        ax_right.add_patch(FancyBboxPatch(
            (cx - 1.3, 4.9), 2.6, 1.55,
            boxstyle="round,pad=0.08",
            facecolor=PANEL_BG, edgecolor=SPINE_CLR, linewidth=1.0,
        ))
        ax_right.text(cx, 6.25, ticker, ha="center", fontsize=10,
                      color=TEXT_CLR, fontweight="bold", **FONT)
        ax_right.text(cx, 5.75, f"VaR 95%: {var}",
                      ha="center", fontsize=8, color=DATA_CLR, **FONT)
        ax_right.text(cx, 5.3, f"P(loss): {prob}",
                      ha="center", fontsize=8, color=NEUTRAL_CLR, **FONT)

    # Risk contribution bar chart wireframe
    ax_right.text(0.4, 4.6, "Risk Contribution per Stock (VaR 95%)",
                  fontsize=9, color=TEXT_CLR, fontweight="bold", **FONT)

    ax_right.add_patch(FancyBboxPatch(
        (0.3, 1.0), 9.4, 3.35,
        boxstyle="round,pad=0.1",
        facecolor=PANEL_BG, edgecolor=SPINE_CLR, linewidth=1.0,
    ))

    bar_data = [
        ("NVDA",  0.72, NEGATIVE),
        ("AAPL",  0.51, OUTPUT_CLR),
        ("MSFT",  0.38, DATA_CLR),
        ("JPM",   0.25, COMPUTE_CLR),
        ("JNJ",   0.14, NEUTRAL_CLR),
    ]
    max_val  = max(v for _, v, _ in bar_data)
    bar_h    = 0.42
    y_start  = 3.85

    for j, (label, val, color) in enumerate(bar_data):
        by    = y_start - j * 0.52
        bw    = val / max_val * 6.5
        ax_right.add_patch(FancyBboxPatch(
            (1.5, by - bar_h / 2), bw, bar_h,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="none", alpha=0.85,
        ))
        ax_right.text(1.3, by, label, ha="right", va="center",
                      fontsize=8, color=TEXT_CLR, **FONT)
        ax_right.text(1.55 + bw, by, f"{val:.0%}", ha="left", va="center",
                      fontsize=7.5, color=TEXT_CLR, **FONT)

    plt.tight_layout()
    return _save(fig, "diag_4_output_structure.png", output_dir)


# ---------------------------------------------------------------------------
# Diagram 5 — Model Fitting (fat tails · correlation · Cholesky)
# ---------------------------------------------------------------------------

def diag_model_fitting(output_dir=OUTPUT_DIR) -> str:
    """
    Three-panel diagram:
      Panel 1 — histogram of synthetic daily returns overlaid with
                normal PDF and t-distribution PDF
      Panel 2 — 5×5 correlation matrix heatmap
      Panel 3 — Cholesky decomposition schematic (C = L · Lᵀ)
    All panels use synthetic data — no real prices required.
    """
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), facecolor=BG)
    fig.suptitle(
        "Model Fitting — Fat Tails · Correlation · Cholesky Decomposition",
        color=TEXT_CLR, fontsize=14, fontweight="bold", **FONT, y=0.99,
    )

    # ── Panel 1: Return histogram vs normal vs t ──────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(PANEL_BG)
    for sp in ax1.spines.values():
        sp.set_color(SPINE_CLR)

    dof   = 5
    scale = 0.013
    raw   = rng.standard_t(dof, size=3000) * scale
    raw   = raw[np.abs(raw) < 0.12]          # clip extreme outliers for display

    x = np.linspace(-0.09, 0.09, 500)
    mu_n, sig_n  = raw.mean(), raw.std()
    pdf_normal   = stats.norm.pdf(x, mu_n, sig_n)
    pdf_t        = stats.t.pdf(x, dof, loc=0, scale=scale)

    ax1.hist(raw, bins=65, density=True,
             color=DATA_CLR, alpha=0.30, edgecolor="none", label="Simulated returns")
    ax1.plot(x, pdf_normal, color=NEUTRAL_CLR, lw=2.0, ls="--", label="Normal fit")
    ax1.plot(x, pdf_t,      color=NEGATIVE,    lw=2.2,           label=f"t-dist  (ν={dof})")

    # Annotate where the t-dist wins in the tail
    tail_x   = 0.060
    y_n_tail = stats.norm.pdf(tail_x, mu_n, sig_n)
    y_t_tail = stats.t.pdf(tail_x, dof, loc=0, scale=scale)
    mid_y    = (y_n_tail + y_t_tail) / 2
    ax1.annotate(
        "heavier tail\nfitted here",
        xy=(tail_x, mid_y),
        xytext=(0.042, max(pdf_normal) * 0.55),
        fontsize=7.5, color=NEGATIVE, **FONT,
        arrowprops=dict(arrowstyle="-|>", color=NEGATIVE, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                  edgecolor=NEGATIVE, alpha=0.9),
    )

    ax1.set_title("Return Distribution\n(Normal vs t-distribution)",
                  fontsize=10, color=TEXT_CLR, fontweight="bold", **FONT, pad=8)
    ax1.set_xlabel("Daily Return", fontsize=9, color=LIGHT_TEXT, **FONT)
    ax1.set_ylabel("Density",      fontsize=9, color=LIGHT_TEXT, **FONT)
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax1.tick_params(colors=LIGHT_TEXT, labelsize=8)
    ax1.legend(fontsize=8, framealpha=0.9, loc="upper left",
               facecolor=BG, edgecolor=SPINE_CLR, labelcolor=TEXT_CLR)

    # ── Panel 2: Correlation heatmap ──────────────────────────────────────────
    ax2 = axes[1]
    for sp in ax2.spines.values():
        sp.set_color(SPINE_CLR)

    tickers_5 = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ"]
    C = np.array([
        [1.00, 0.72, 0.61, 0.38, 0.21],
        [0.72, 1.00, 0.58, 0.41, 0.19],
        [0.61, 0.58, 1.00, 0.32, 0.15],
        [0.38, 0.41, 0.32, 1.00, 0.29],
        [0.21, 0.19, 0.15, 0.29, 1.00],
    ])

    im = ax2.imshow(C, cmap="RdYlBu_r", vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(5):
        for j in range(5):
            txt_clr = WHITE if C[i, j] > 0.55 else TEXT_CLR
            ax2.text(j, i, f"{C[i, j]:.2f}",
                     ha="center", va="center", fontsize=9,
                     color=txt_clr, fontweight="bold", **FONT)

    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    ax2.set_xticklabels(tickers_5, fontsize=9, color=TEXT_CLR, **FONT)
    ax2.set_yticklabels(tickers_5, fontsize=9, color=TEXT_CLR, **FONT)
    ax2.set_title("Correlation Matrix\n(5 × 5 example)",
                  fontsize=10, color=TEXT_CLR, fontweight="bold", **FONT, pad=8)

    cb = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8, colors=LIGHT_TEXT)

    # ── Panel 3: Cholesky schematic ───────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 8.5)
    ax3.axis("off")
    ax3.set_facecolor(BG)
    ax3.set_title("Cholesky Decomposition\nC = L · Lᵀ",
                  fontsize=10, color=TEXT_CLR, fontweight="bold", **FONT, pad=8)

    # Compute actual Cholesky from C
    eigvals, eigvecs = np.linalg.eigh(C)
    C_psd = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T
    d     = np.sqrt(np.diag(C_psd))
    C_psd = C_psd / d[:, None] / d[None, :]
    L_mat = np.linalg.cholesky(C_psd)

    n         = 5
    cell      = 0.72
    top_y     = 7.8
    start_C_x = 0.25
    eq_x      = start_C_x + n * cell + 0.40
    start_L_x = eq_x + 0.60

    # ── C matrix ─────────────────────────────────────────────────────────────
    ax3.text(start_C_x + n * cell / 2, top_y + 0.35,
             "C  (correlation)", ha="center", fontsize=8,
             color=DATA_CLR, fontweight="bold", **FONT)

    for i in range(n):
        for j in range(n):
            cx = start_C_x + j * cell + cell / 2
            cy = top_y - i * cell - cell / 2
            intensity = C[i, j]
            fc = plt.cm.RdYlBu_r(intensity * 0.95)
            ax3.add_patch(FancyBboxPatch(
                (cx - cell / 2 + 0.03, cy - cell / 2 + 0.03),
                cell - 0.06, cell - 0.06,
                boxstyle="square,pad=0.01",
                facecolor=fc, edgecolor=WHITE, linewidth=0.8, zorder=3,
            ))
            txt_clr = WHITE if intensity > 0.55 else TEXT_CLR
            ax3.text(cx, cy, f"{intensity:.1f}", ha="center", va="center",
                     fontsize=6.5, color=txt_clr, fontweight="bold",
                     **FONT, zorder=4)

    # ── "=" sign ──────────────────────────────────────────────────────────────
    ax3.text(eq_x, top_y - n * cell / 2, "=",
             ha="center", va="center", fontsize=22, color=TEXT_CLR, **FONT)

    # ── L matrix (lower-triangular) ───────────────────────────────────────────
    ax3.text(start_L_x + n * cell / 2, top_y + 0.35,
             "L  (lower triangle)", ha="center", fontsize=8,
             color=COMPUTE_CLR, fontweight="bold", **FONT)

    l_max = L_mat.max()
    for i in range(n):
        for j in range(n):
            cx = start_L_x + j * cell + cell / 2
            cy = top_y - i * cell - cell / 2
            if j > i:
                ax3.add_patch(FancyBboxPatch(
                    (cx - cell / 2 + 0.03, cy - cell / 2 + 0.03),
                    cell - 0.06, cell - 0.06,
                    boxstyle="square,pad=0.01",
                    facecolor="#eeeeee", edgecolor=WHITE,
                    linewidth=0.8, zorder=3,
                ))
                ax3.text(cx, cy, "0", ha="center", va="center",
                         fontsize=6.5, color=LIGHT_TEXT, **FONT, zorder=4)
            else:
                val = L_mat[i, j]
                fc  = plt.cm.Greens(0.25 + (val / l_max) * 0.70)
                ax3.add_patch(FancyBboxPatch(
                    (cx - cell / 2 + 0.03, cy - cell / 2 + 0.03),
                    cell - 0.06, cell - 0.06,
                    boxstyle="square,pad=0.01",
                    facecolor=fc, edgecolor=WHITE,
                    linewidth=0.8, zorder=3,
                ))
                txt_clr = WHITE if val / l_max > 0.55 else TEXT_CLR
                ax3.text(cx, cy, f"{val:.2f}", ha="center", va="center",
                         fontsize=6.0, color=txt_clr,
                         fontweight="bold", **FONT, zorder=4)

    # ── Usage annotation ──────────────────────────────────────────────────────
    bot_y = top_y - n * cell - 0.15
    ax3.text(5.0, bot_y - 0.15,
             "z_corr  =  L  ·  z_raw",
             ha="center", va="top", fontsize=9,
             color=COMPUTE_CLR, fontweight="bold", **FONT,
             bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL_BG,
                       edgecolor=COMPUTE_CLR, alpha=0.95))
    ax3.text(5.0, bot_y - 0.90,
             "Multiplying independent shocks z_raw\nby L injects the empirical correlations\nbetween stocks into each simulation step.",
             ha="center", va="top", fontsize=7.5, color=LIGHT_TEXT, **FONT,
             multialignment="center")

    plt.tight_layout()
    return _save(fig, "diag_5_model_fitting.png", output_dir)


# ---------------------------------------------------------------------------
# Generate all diagrams
# ---------------------------------------------------------------------------

def generate_all(output_dir=OUTPUT_DIR) -> list:
    print(f"\nGenerating presentation diagrams -> {os.path.abspath(output_dir)}/")
    paths = [
        diag_data_input(output_dir),
        diag_simulation_mechanics(output_dir),
        diag_pipeline_architecture(output_dir),
        diag_output_structure(output_dir),
        diag_model_fitting(output_dir),
    ]
    print(f"\nAll diagrams saved ({len(paths)} files).\n")
    return paths


if __name__ == "__main__":
    generate_all()