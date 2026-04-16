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
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Half-life cap: signals persisting beyond 1 trading year treated equally
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
    direction:                  str   # "leads_your_holdings" | "follows_your_holdings" | "both"
    signal_score:               float
    centrality_score:           float
    sector_diversity_score:     float
    coverage_score:             float
    durability_score:           float  # NEW: normalized half-life 0-100
    reasoning:                  str


@dataclass
class IndependentRecommendation:
    """
    Group B — stock with zero detected relationship to any holding.
    Ranked by composite of sector gap (primary) and centrality (secondary).
    """
    ticker:            str
    sector:            str
    centrality:        float
    composite_score:   float   # 0-100: 70% sector gap + 30% centrality
    sector_gap_score:  float   # 100 if sector absent from portfolio, else scaled down
    centrality_score:  float   # normalized 0-100
    reasoning:         str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_tickers(tickers: list) -> list:
    return [t.strip().upper() for t in tickers if t.strip()]


def get_ticker_metadata(df: pd.DataFrame) -> pd.DataFrame:
    from app.services.mock_final_network import get_ticker_metadata as _meta
    return _meta(df)


def _sector_distribution(holdings: list, meta: pd.DataFrame) -> dict:
    dist: dict = {}
    for t in holdings:
        if t in meta.index:
            s = meta.loc[t, "sector"]
            dist[s] = dist.get(s, 0) + 1
    return dist


def _compute_durability_score(half_life: float) -> float:
    """
    Normalize half_life (days) to 0-100.
    Capped at _HALF_LIFE_CAP_DAYS (252 = 1 trading year).
    half_life of -1 means fit failed — treat as 0 durability.
    """
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
    if len(tickers) < 2:
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
    if not tickers:
        return []

    meta        = get_ticker_metadata(network_df)
    sector_dist = _sector_distribution(list(tickers), meta)
    total       = len(tickers)

    involves = network_df["ticker_i"].isin(tickers) | network_df["ticker_j"].isin(tickers)
    filtered = network_df[involves & (network_df["signal_strength"] >= min_signal_strength)].copy()
    if filtered.empty:
        return []

    # Check if centrality data is actually populated — if all zero, redistribute weight
    all_centrality = pd.concat([
        filtered["centrality_i"].fillna(0.0),
        filtered["centrality_j"].fillna(0.0),
    ])
    centrality_available = all_centrality.max() > 0.0
    if not centrality_available:
        # Redistribute centrality weight to signal_score
        signal_weight    = signal_weight + centrality_weight
        centrality_weight = 0.0

    candidates: dict = {}
    for _, row in filtered.iterrows():
        ti, tj   = row["ticker_i"], row["ticker_j"]
        ti_in    = ti in tickers
        tj_in    = tj in tickers
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
                "frequencies":      [],   # collected alongside signal_strengths
                "half_lives":       [],   # for durability score
                "related_holdings": [],
                "directions":       [],
                "centrality":       0.0,
                "sector":           "Unknown",
            }

        candidates[cand]["signal_strengths"].append(float(row["signal_strength"]))
        # frequency: fraction of 15-yr windows where this pair was significant
        candidates[cand]["frequencies"].append(float(row.get("frequency", 0.5)))
        # half_life: days until signal decays to half strength
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

        # Frequency-weighted signal score — pairs that appeared consistently
        # across more windows contribute more weight than sporadic ones
        total_freq = sum(freqs)
        if total_freq > 0:
            signal_score = float(np.average(sigs, weights=freqs))
        else:
            signal_score = float(np.mean(sigs))

        centrality_score       = (data["centrality"] / max_centrality) * 100.0
        coverage_score         = (n_conn / max_coverage) * 100.0
        sector_frac            = sector_dist.get(sector, 0) / total if total else 0
        sector_diversity_score = max(0.0, (1.0 - sector_frac) * 100.0)

        # Durability: use the best (longest) half_life among all connections
        # Best = most persistent relationship this candidate has with the portfolio
        valid_hls     = [h for h in hls if h > 0]
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
) -> list[IndependentRecommendation]:
    """
    Group B: stocks with ZERO detected lead-lag relationships to any user holding.

    If centrality data is unavailable (all zeros), full weight goes to sector_gap_score.
    """
    tickers = set(_normalize_tickers(user_tickers))
    if not tickers:
        return []

    meta           = get_ticker_metadata(network_df)
    universe       = set(meta.index.tolist())
    candidates     = universe - tickers
    sector_dist    = _sector_distribution(list(tickers), meta)
    total_holdings = len(tickers)

    # Check centrality availability
    all_centrality = pd.concat([
        network_df["centrality_i"].fillna(0.0),
        network_df["centrality_j"].fillna(0.0),
    ])
    centrality_available = all_centrality.max() > 0.0
    if not centrality_available:
        sector_gap_weight = 1.0
        centrality_weight = 0.0

    # Find all tickers connected to user holdings (signal-connected group)
    involves_user = (
        network_df["ticker_i"].isin(tickers) | network_df["ticker_j"].isin(tickers)
    )
    related_df = network_df[involves_user]
    connected  = set(related_df["ticker_i"].tolist() + related_df["ticker_j"].tolist())
    connected -= tickers

    truly_independent = candidates - connected
    if not truly_independent:
        return []

    raw = []
    for ticker in truly_independent:
        if ticker not in meta.index:
            continue
        row = meta.loc[ticker]
        raw.append({
            "ticker":     ticker,
            "sector":     row["sector"],
            "centrality": float(row["centrality"]),
        })

    if not raw:
        return []

    max_cent = max(r["centrality"] for r in raw) or 1.0

    results = []
    for r in raw:
        ticker = r["ticker"]
        sector = r["sector"]
        cent   = r["centrality"]

        sector_frac      = sector_dist.get(sector, 0) / total_holdings if total_holdings else 0
        sector_gap_score = max(0.0, (1.0 - sector_frac) * 100.0)
        centrality_score = (cent / max_cent) * 100.0

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
    """
    Return {ticker: sector} for all known user holdings.
    Used by frontend to populate the SectorDonut "current portfolio" view
    regardless of whether any overlaps exist.
    """
    tickers = _normalize_tickers(user_tickers)
    meta    = get_ticker_metadata(network_df)
    result  = {}
    for t in tickers:
        if t in meta.index:
            result[t] = str(meta.loc[t, "sector"])
    return result