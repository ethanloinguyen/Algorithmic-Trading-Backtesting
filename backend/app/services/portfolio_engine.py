# backend/app/services/portfolio_engine.py
"""
Core scoring logic for the portfolio diversification tool.
No FastAPI or BQ imports — pure pandas + dataclasses.

Two recommendation groups:
    signal_recommendations       — stocks with detected lead-lag relationships to holdings
    independent_recommendations  — stocks with ZERO detected relationships (pure diversification)

Group A composite (must sum to 1.0):
    0.40 × signal_score        — frequency-weighted mean signal strength across connections
    0.15 × durability_score    — normalized half-life (how long the signal persists)
    0.15 × coverage_score      — fraction of user's holdings this stock connects to
    0.20 × sector_diversity    — how underrepresented this sector is in the portfolio
    0.10 × centrality_score    — eigenvector centrality in market network
                                  (gracefully zeroed if centrality data unavailable)

Group B composite:
    0.70 × sector_gap_score
    0.30 × centrality_score    (gracefully zeroed if unavailable)

Component 1 — dCor-filtered candidate pool (feeds Components 2 & 3):
    compute_dcor_filtered_pool() computes mean dCor between each candidate stock
    in the universe and the user's portfolio holdings.  Candidates above the dCor
    threshold are filtered out as too correlated.  The remaining low-dCor pool is
    returned sorted ascending by mean_dcor_to_portfolio (most independent first),
    ready for downstream clustering and Monte Carlo risk assessment.

    mean_dcor_to_portfolio(C) = mean({pair_dcor(C, p) for p in portfolio
                                      where pair (C, p) exists in network})

    Stocks with NO pairs in the network at all get dcor = 0.0 — no detected
    dependency means they automatically pass the filter.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

_HALF_LIFE_CAP_DAYS = 252.0


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class OverlapResult:
    ticker_leader:    str
    ticker_follower:  str
    best_lag:         int
    signal_strength:  float
    mean_dcor:        float
    oos_sharpe_net:   float
    predicted_sharpe: float
    sector_leader:    str
    sector_follower:  str
    frequency:        float
    half_life:        float
    sharpness:        float
    interpretation:   str


@dataclass
class Recommendation:
    """Group A — signal-connected stock."""
    ticker:                     str
    sector:                     str
    centrality:                 float
    composite_score:            float
    n_portfolio_relationships:  int
    best_relationship_strength: float
    mean_relationship_strength: float
    related_holdings:           list
    direction:                  str
    signal_score:               float
    centrality_score:           float
    sector_diversity_score:     float
    coverage_score:             float
    durability_score:           float
    reasoning:                  str


@dataclass
class IndependentRecommendation:
    """Group B — stock with zero detected relationship to any holding."""
    ticker:            str
    sector:            str
    centrality:        float
    composite_score:   float
    sector_gap_score:  float
    centrality_score:  float
    reasoning:         str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_tickers(tickers: list) -> list:
    return [t.strip().upper() for t in tickers if t.strip()]


def get_ticker_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive ticker → (sector, centrality) from the network DataFrame.
    Works directly from the pair rows — no separate metadata table needed.
    """
    if df.empty:
        return pd.DataFrame(columns=["ticker", "sector", "centrality"]).set_index("ticker")

    side_i = df[["ticker_i", "sector_i", "centrality_i"]].rename(
        columns={"ticker_i": "ticker", "sector_i": "sector", "centrality_i": "centrality"})
    side_j = df[["ticker_j", "sector_j", "centrality_j"]].rename(
        columns={"ticker_j": "ticker", "sector_j": "sector", "centrality_j": "centrality"})
    meta = pd.concat([side_i, side_j]).drop_duplicates("ticker")
    return meta.set_index("ticker")


def _sector_distribution(holdings: list, meta: pd.DataFrame) -> dict:
    dist: dict = {}
    for t in holdings:
        if t in meta.index:
            s = meta.loc[t, "sector"]
            dist[s] = dist.get(s, 0) + 1
    return dist


def _compute_durability_score(half_life: float) -> float:
    if half_life is None or half_life < 0:
        return 0.0
    return min(half_life, _HALF_LIFE_CAP_DAYS) / _HALF_LIFE_CAP_DAYS * 100.0


def _interpretation(row: pd.Series) -> str:
    lag  = int(row["best_lag"])
    sig  = round(float(row["signal_strength"]), 1)
    dcor = round(float(row["mean_dcor"]), 3)
    hl   = int(round(float(row["half_life"])))
    freq = int(round(float(row["frequency"]) * 100))
    return (
        f"{row['ticker_i']} consistently leads {row['ticker_j']} by {lag} trading "
        f"day{'s' if lag > 1 else ''} (signal strength {sig}/100, dCor={dcor}). "
        f"This relationship appeared in {freq}% of rolling windows over 15 years "
        f"with an estimated half-life of {hl} days."
    )


def _signal_reasoning(ticker, related, direction, sector, signal_score,
                      sector_diversity_score, n, durability_score) -> str:
    holdings_str = ", ".join(related[:3])
    if len(related) > 3:
        holdings_str += f" and {len(related) - 3} more"
    dir_phrase = {
        "leads_your_holdings":   f"leads {holdings_str}",
        "follows_your_holdings": f"follows {holdings_str}",
        "both":                  f"has bidirectional relationships with {holdings_str}",
    }.get(direction, f"relates to {holdings_str}")
    diversity_note = (
        f" Adding {ticker} introduces {sector} exposure not well represented in your portfolio."
        if sector_diversity_score > 15 else ""
    )
    durability_note = ""
    if durability_score >= 75:
        durability_note = " The signal has high persistence, suggesting a durable structural relationship."
    elif durability_score <= 25:
        durability_note = " Note: the signal decays relatively quickly — monitor for changes."
    return (
        f"{ticker} ({sector}) statistically {dir_phrase} with a frequency-weighted mean signal "
        f"strength of {round(signal_score, 1)}/100 across {n} of your holdings.{diversity_note}{durability_note}"
    )


def _independent_reasoning(
    ticker: str, sector: str, centrality: float,
    sector_gap_score: float, current_sectors: dict,
) -> str:
    centrality_pct = int(round(centrality * 100))
    sector_in_portfolio = sector in current_sectors
    sector_note = (
        f"{sector} is not currently represented in your portfolio, "
        f"making it a direct sector gap fill."
        if not sector_in_portfolio else
        f"You already hold {current_sectors.get(sector, 0)} stock(s) in {sector}, "
        f"but this adds further independent exposure within the sector."
    )
    return (
        f"{ticker} ({sector}) has no detected lead-lag relationship with any of your "
        f"holdings across 15 years of analysis — purely independent exposure. "
        f"{sector_note} "
        f"It ranks in the top {centrality_pct}% of stocks by market centrality, "
        f"meaning it carries real informational weight in the broader market network."
    )


# ── Function 1: Overlap analyzer ─────────────────────────────────────────────

def analyze_portfolio_overlap(
    user_tickers: list,
    network_df: pd.DataFrame,
    min_signal_strength: float = 0.0,
) -> list[OverlapResult]:
    tickers = set(_normalize_tickers(user_tickers))
    if len(tickers) < 2 or network_df.empty:
        return []

    mask = (
        network_df["ticker_i"].isin(tickers) &
        network_df["ticker_j"].isin(tickers) &
        (network_df["signal_strength"] >= min_signal_strength)
    )
    sub = network_df[mask].copy().sort_values("signal_strength", ascending=False)

    results = []
    for _, row in sub.iterrows():
        results.append(OverlapResult(
            ticker_leader=row["ticker_i"],
            ticker_follower=row["ticker_j"],
            best_lag=int(row["best_lag"]),
            signal_strength=round(float(row["signal_strength"]), 1),
            mean_dcor=round(float(row["mean_dcor"]), 4),
            oos_sharpe_net=round(float(row["oos_sharpe_net"]), 3),
            predicted_sharpe=round(float(row["predicted_sharpe"]), 3),
            sector_leader=row["sector_i"],
            sector_follower=row["sector_j"],
            frequency=round(float(row["frequency"]), 3),
            half_life=round(float(row["half_life"]), 1),
            sharpness=round(float(row["sharpness"]), 3),
            interpretation=_interpretation(row),
        ))
    return results


# ── Function 2: Signal-connected recommendations ──────────────────────────────

def get_signal_recommendations(
    user_tickers: list,
    network_df: pd.DataFrame,
    top_n: int = 10,
    min_signal_strength: float = 55.0,
    signal_weight: float = 0.40,
    durability_weight: float = 0.15,
    coverage_weight: float = 0.15,
    sector_diversity_weight: float = 0.20,
    centrality_weight: float = 0.10,
) -> list[Recommendation]:
    tickers = set(_normalize_tickers(user_tickers))
    if not tickers or network_df.empty:
        return []

    meta        = get_ticker_metadata(network_df)
    sector_dist = _sector_distribution(list(tickers), meta)
    total       = len(tickers)

    involves = network_df["ticker_i"].isin(tickers) | network_df["ticker_j"].isin(tickers)
    filtered = network_df[involves & (network_df["signal_strength"] >= min_signal_strength)].copy()
    if filtered.empty:
        return []

    # Check centrality availability — if all zero, redistribute weight to signal
    all_centrality = pd.concat([
        filtered["centrality_i"].fillna(0.0),
        filtered["centrality_j"].fillna(0.0),
    ])
    if all_centrality.max() == 0.0:
        signal_weight     = signal_weight + centrality_weight
        centrality_weight = 0.0

    candidates: dict = {}
    for _, row in filtered.iterrows():
        ti, tj = row["ticker_i"], row["ticker_j"]
        ti_in  = ti in tickers
        tj_in  = tj in tickers
        if ti_in and tj_in:
            continue
        if ti_in and not tj_in:
            cand, holding, direction = tj, ti, "follows_your_holdings"
        elif tj_in and not ti_in:
            cand, holding, direction = ti, tj, "leads_your_holdings"
        else:
            continue

        if cand not in candidates:
            candidates[cand] = {
                "signal_strengths": [],
                "frequencies":      [],
                "half_lives":       [],
                "related_holdings": [],
                "directions":       [],
                "centrality":       0.0,
                "sector":           "Unknown",
            }

        candidates[cand]["signal_strengths"].append(float(row["signal_strength"]))
        candidates[cand]["frequencies"].append(float(row.get("frequency", 0.5)))
        candidates[cand]["half_lives"].append(float(row.get("half_life", -1.0)))
        candidates[cand]["related_holdings"].append(holding)
        candidates[cand]["directions"].append(direction)

        if cand == ti:
            candidates[cand]["centrality"] = max(candidates[cand]["centrality"], float(row["centrality_i"]))
            candidates[cand]["sector"]     = row["sector_i"]
        else:
            candidates[cand]["centrality"] = max(candidates[cand]["centrality"], float(row["centrality_j"]))
            candidates[cand]["sector"]     = row["sector_j"]

    if not candidates:
        return []

    max_centrality = max(d["centrality"] for d in candidates.values()) or 1.0
    max_coverage   = max(len(set(d["related_holdings"])) for d in candidates.values()) or 1.0

    recs = []
    for ticker, data in candidates.items():
        sigs    = data["signal_strengths"]
        freqs   = data["frequencies"]
        hls     = data["half_lives"]
        related = list(dict.fromkeys(data["related_holdings"]))
        sector  = data["sector"]
        n_conn  = len(set(related))

        total_freq = sum(freqs)
        signal_score = float(np.average(sigs, weights=freqs)) if total_freq > 0 else float(np.mean(sigs))

        centrality_score       = (data["centrality"] / max_centrality) * 100.0
        coverage_score         = (n_conn / max_coverage) * 100.0
        sector_frac            = sector_dist.get(sector, 0) / total if total else 0
        sector_diversity_score = max(0.0, (1.0 - sector_frac) * 100.0)

        valid_hls      = [h for h in hls if h > 0]
        best_half_life = max(valid_hls) if valid_hls else -1.0
        durability_score = _compute_durability_score(best_half_life)

        composite = (
            signal_weight           * signal_score +
            durability_weight       * durability_score +
            coverage_weight         * coverage_score +
            sector_diversity_weight * sector_diversity_score +
            centrality_weight       * centrality_score
        )

        dir_counts = {d: data["directions"].count(d) for d in set(data["directions"])}
        direction  = "both" if len(dir_counts) > 1 else list(dir_counts.keys())[0]

        recs.append(Recommendation(
            ticker=ticker, sector=sector,
            centrality=round(data["centrality"], 4),
            composite_score=round(composite, 2),
            n_portfolio_relationships=n_conn,
            best_relationship_strength=round(float(np.max(sigs)), 1),
            mean_relationship_strength=round(signal_score, 1),
            related_holdings=related,
            direction=direction,
            signal_score=round(signal_score, 1),
            centrality_score=round(centrality_score, 1),
            sector_diversity_score=round(sector_diversity_score, 1),
            coverage_score=round(coverage_score, 1),
            durability_score=round(durability_score, 1),
            reasoning=_signal_reasoning(
                ticker, related, direction, sector,
                signal_score, sector_diversity_score, n_conn, durability_score,
            ),
        ))

    recs.sort(key=lambda r: r.composite_score, reverse=True)
    return recs[:top_n]


# ── Function 3: Truly independent recommendations ─────────────────────────────

def get_independent_recommendations(
    user_tickers: list,
    network_df: pd.DataFrame,
    top_n: int = 10,
    sector_gap_weight: float = 0.70,
    centrality_weight: float = 0.30,
    universe_meta_df: pd.DataFrame | None = None,
) -> list[IndependentRecommendation]:
    """
    Group B: stocks with ZERO detected lead-lag relationships to any user holding.

    universe_meta_df: optional compact DataFrame with columns [ticker, sector, centrality]
    representing the full ticker universe. If provided, used instead of network_df
    to build the candidate pool — this avoids needing to load the full network table.
    """
    tickers = set(_normalize_tickers(user_tickers))
    if not tickers:
        return []

    # Determine sector distribution from available metadata
    meta_for_dist = get_ticker_metadata(network_df)
    sector_dist   = _sector_distribution(list(tickers), meta_for_dist)
    total_holdings = len(tickers)

    # Build the universe from compact metadata if provided, else fall back to network_df
    if universe_meta_df is not None and not universe_meta_df.empty:
        # universe_meta_df has columns: ticker, sector, centrality
        universe_df = universe_meta_df.copy()
    else:
        # Fall back: derive from the pair rows we already have
        side_i = network_df[["ticker_i", "sector_i", "centrality_i"]].rename(
            columns={"ticker_i": "ticker", "sector_i": "sector", "centrality_i": "centrality"})
        side_j = network_df[["ticker_j", "sector_j", "centrality_j"]].rename(
            columns={"ticker_j": "ticker", "sector_j": "sector", "centrality_j": "centrality"})
        universe_df = pd.concat([side_i, side_j]).drop_duplicates("ticker")

    # Tickers connected to user holdings (from the targeted network_df slice)
    connected: set[str] = set()
    if not network_df.empty:
        involves_user = (
            network_df["ticker_i"].isin(tickers) | network_df["ticker_j"].isin(tickers)
        )
        related_df = network_df[involves_user]
        connected  = set(related_df["ticker_i"].tolist() + related_df["ticker_j"].tolist())
        connected -= tickers

    # Check centrality availability
    max_cent_val = universe_df["centrality"].max() if not universe_df.empty else 0.0
    if max_cent_val == 0.0:
        sector_gap_weight = 1.0
        centrality_weight = 0.0

    results = []
    for _, row in universe_df.iterrows():
        ticker = row["ticker"]
        if ticker in tickers or ticker in connected:
            continue

        sector = row["sector"]
        cent   = float(row["centrality"])

        sector_frac      = sector_dist.get(sector, 0) / total_holdings if total_holdings else 0
        sector_gap_score = max(0.0, (1.0 - sector_frac) * 100.0)
        centrality_score = (cent / max_cent_val * 100.0) if max_cent_val > 0 else 0.0

        composite = (
            sector_gap_weight  * sector_gap_score +
            centrality_weight  * centrality_score
        )

        results.append(IndependentRecommendation(
            ticker=ticker,
            sector=sector,
            centrality=round(cent, 4),
            composite_score=round(composite, 2),
            sector_gap_score=round(sector_gap_score, 1),
            centrality_score=round(centrality_score, 1),
            reasoning=_independent_reasoning(
                ticker, sector, cent, sector_gap_score, sector_dist
            ),
        ))

    results.sort(key=lambda r: r.composite_score, reverse=True)
    return results[:top_n]


# ── Function 4: Holdings sector map ──────────────────────────────────────────

def get_holdings_sectors(
    user_tickers: list,
    network_df: pd.DataFrame,
) -> dict[str, str]:
    tickers = _normalize_tickers(user_tickers)
    meta    = get_ticker_metadata(network_df)
    result  = {}
    for t in tickers:
        if t in meta.index:
            result[t] = str(meta.loc[t, "sector"])
    return result


# ── Component 1 — dCor-filtered candidate pool ────────────────────────────────

@dataclass
class DcorCandidate:
    """
    A candidate stock that passed the dCor threshold filter.
    Ready to be passed to Component 2 (clustering) and Component 3 (Monte Carlo).
    """
    ticker:                  str
    sector:                  str
    centrality:              float
    mean_dcor_to_portfolio:  float   # average pairwise dCor vs all user holdings
    n_portfolio_pairs:       int     # number of portfolio stocks this was paired with
    paired_holdings:         dict    # {holding_ticker: mean_dcor} for each detected pair
    reasoning:               str


def _dcor_candidate_reasoning(
    ticker: str,
    sector: str,
    mean_dcor: float,
    n_pairs: int,
    n_portfolio: int,
    is_zero_pair: bool = False,
) -> str:
    if is_zero_pair:
        return (
            f"{ticker} ({sector}) has no detected lead-lag pairs with any of your "
            f"{n_portfolio} portfolio holding(s) across 15 years of analysis. "
            f"It was selected for its high market network centrality, meaning it carries "
            f"real informational weight in the broader market — a strong independent candidate."
        )
    coverage_str = f"paired with {n_pairs}/{n_portfolio} portfolio stock(s) in the network"
    return (
        f"{ticker} ({sector}) has a mean distance correlation of {round(mean_dcor, 3)} "
        f"with your portfolio ({coverage_str}). "
        f"This is among the lowest measured dCor values across all candidates, "
        f"indicating low nonlinear dependence with your current holdings."
    )


def compute_dcor_filtered_pool(
    user_tickers: list[str],
    network_df: pd.DataFrame,
    universe_meta_df: pd.DataFrame | None = None,
    dcor_threshold: float = 0.3,
    max_network_candidates: int = 20,
    max_zero_pair_candidates: int = 80,
) -> list[DcorCandidate]:
    """
    Component 1: compute mean dCor between each candidate stock in the universe
    and the user's portfolio, then return a curated pool sorted ascending by
    mean_dcor_to_portfolio (most independent first).

    Why caps instead of a fixed threshold
    --------------------------------------
    All pairs that survive the pipeline's FDR correction and enter final_network
    already have statistically significant but naturally low mean_dcor values
    (typically 0.05–0.20).  An absolute 0.3 threshold therefore passes virtually
    every candidate in the ~1700-stock universe.  Instead we:

      - Network candidates  : keep the top `max_network_candidates` stocks with the
                              lowest measured mean dCor (already most independent).
      - Zero-pair candidates: keep the top `max_zero_pair_candidates` stocks by
                              market centrality — no detected relationship + high
                              market connectivity = high-quality independent pick.

    The absolute dcor_threshold is retained as a safety cap to exclude any
    pathologically high-dCor outliers that somehow survive the pipeline.

    Parameters
    ----------
    user_tickers              : tickers in the user's current portfolio
    network_df                : pairwise dCor rows from final_network that involve
                                at least one user holding
    universe_meta_df          : optional compact universe table (ticker, sector, centrality)
    dcor_threshold            : hard cap — paired candidates above this are always excluded
    max_network_candidates    : max paired candidates to return (lowest dCor first)
    max_zero_pair_candidates  : max zero-pair candidates to return (highest centrality first)

    Returns
    -------
    list[DcorCandidate] sorted ascending by mean_dcor_to_portfolio, then descending
    by centrality
    """
    portfolio = set(_normalize_tickers(user_tickers))
    if not portfolio:
        return []

    n_portfolio = len(portfolio)

    # ── Step 1: collect per-candidate dCor values from network pairs ──────────
    # network_df contains rows where ticker_i or ticker_j is a user holding.
    # For each such row the OTHER side is a candidate.
    # candidate_dcors  : {candidate: [dcor, ...]}  — for computing the mean
    # candidate_holding_dcors: {candidate: {holding: dcor}} — for UI display
    candidate_dcors: dict[str, list[float]] = {}
    candidate_holding_dcors: dict[str, dict[str, float]] = {}

    if not network_df.empty:
        for _, row in network_df.iterrows():
            ti = row["ticker_i"]
            tj = row["ticker_j"]
            dcor_val = float(row["mean_dcor"])

            ti_in_portfolio = ti in portfolio
            tj_in_portfolio = tj in portfolio

            # Both are portfolio holdings — skip (overlap, not a candidate)
            if ti_in_portfolio and tj_in_portfolio:
                continue

            # Determine which is the candidate and which is the holding
            if ti_in_portfolio:
                candidate, holding = tj, ti
            elif tj_in_portfolio:
                candidate, holding = ti, tj
            else:
                # Neither side is a portfolio holding — shouldn't appear in the
                # targeted network slice, but guard defensively
                continue

            candidate_dcors.setdefault(candidate, []).append(dcor_val)
            # Store per-holding dCor; if the same pair appears multiple times,
            # keep the minimum (most conservative independence estimate).
            ph = candidate_holding_dcors.setdefault(candidate, {})
            ph[holding] = min(ph.get(holding, float("inf")), round(dcor_val, 4))

    # ── Step 2: build metadata lookup (sector, centrality) ───────────────────
    meta = get_ticker_metadata(network_df)

    # Supplement with universe_meta_df when provided
    if universe_meta_df is not None and not universe_meta_df.empty:
        universe_index = universe_meta_df.set_index("ticker")
    else:
        universe_index = pd.DataFrame(columns=["sector", "centrality"])

    def _sector(ticker: str) -> str:
        if ticker in meta.index:
            return str(meta.loc[ticker, "sector"])
        if ticker in universe_index.index:
            return str(universe_index.loc[ticker, "sector"])
        return "Unknown"

    def _centrality(ticker: str) -> float:
        if ticker in meta.index:
            return float(meta.loc[ticker, "centrality"])
        if ticker in universe_index.index:
            return float(universe_index.loc[ticker, "centrality"])
        return 0.0

    # ── Step 3: compute mean dCor per candidate and apply hard cap ───────────
    network_candidates: list[DcorCandidate] = []

    for ticker, dcor_values in candidate_dcors.items():
        if ticker in portfolio:
            continue
        mean_dcor = float(np.mean(dcor_values))
        if mean_dcor > dcor_threshold:
            continue  # hard cap — exclude pathologically high dCor
        sec  = _sector(ticker)
        cent = _centrality(ticker)
        network_candidates.append(DcorCandidate(
            ticker=ticker,
            sector=sec,
            centrality=round(cent, 4),
            mean_dcor_to_portfolio=round(mean_dcor, 4),
            n_portfolio_pairs=len(dcor_values),
            paired_holdings=candidate_holding_dcors.get(ticker, {}),
            reasoning=_dcor_candidate_reasoning(
                ticker, sec, mean_dcor, len(dcor_values), n_portfolio,
                is_zero_pair=False,
            ),
        ))

    # Sort by ascending dCor, descending centrality — take only the top N most independent
    network_candidates.sort(key=lambda c: (c.mean_dcor_to_portfolio, -c.centrality))
    network_candidates = network_candidates[:max_network_candidates]

    # ── Step 4: zero-pair candidates — capped by centrality ──────────────────
    # Stocks with NO network pairs vs any portfolio holding are collected from
    # universe_meta_df, sorted by centrality descending, and capped at
    # max_zero_pair_candidates.  This ensures only market-relevant, well-connected
    # names are passed through rather than the entire ~1700-stock universe.
    zero_pair_candidates: list[DcorCandidate] = []

    if universe_meta_df is not None and not universe_meta_df.empty:
        seen = set(candidate_dcors.keys()) | portfolio
        zero_rows = [
            row for _, row in universe_meta_df.iterrows()
            if row["ticker"] not in seen
        ]
        # Sort by centrality descending — most network-connected first
        zero_rows.sort(key=lambda r: float(r["centrality"]), reverse=True)

        for row in zero_rows[:max_zero_pair_candidates]:
            ticker = row["ticker"]
            sector = str(row["sector"])
            cent   = float(row["centrality"])
            zero_pair_candidates.append(DcorCandidate(
                ticker=ticker,
                sector=sector,
                centrality=round(cent, 4),
                mean_dcor_to_portfolio=0.0,
                n_portfolio_pairs=0,
                paired_holdings={},
                reasoning=_dcor_candidate_reasoning(
                    ticker, sector, 0.0, 0, n_portfolio,
                    is_zero_pair=True,
                ),
            ))

    # ── Step 5: merge and final sort ─────────────────────────────────────────
    # Network candidates come first (they have measured dCor data), then zero-pair.
    # Within each group the sort applied above is preserved by Python's stable sort.
    results = network_candidates + zero_pair_candidates
    results.sort(key=lambda c: (c.mean_dcor_to_portfolio, -c.centrality))
    return results