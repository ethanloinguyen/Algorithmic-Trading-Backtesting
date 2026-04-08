# backend/app/services/portfolio_engine.py
"""
Core scoring logic for the portfolio diversification tool.
No FastAPI or BQ imports — pure pandas + dataclasses.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


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
    reasoning:                  str


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


def _rec_reasoning(ticker, related, direction, sector, signal_score,
                   sector_diversity_score, n) -> str:
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
    return (
        f"{ticker} ({sector}) statistically {dir_phrase} with a mean signal "
        f"strength of {round(signal_score, 1)}/100 across {n} of your holdings.{diversity_note}"
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
            ticker_leader=row["ticker_i"],    ticker_follower=row["ticker_j"],
            best_lag=int(row["best_lag"]),    signal_strength=round(float(row["signal_strength"]), 1),
            mean_dcor=round(float(row["mean_dcor"]), 4),
            oos_sharpe_net=round(float(row["oos_sharpe_net"]), 3),
            predicted_sharpe=round(float(row["predicted_sharpe"]), 3),
            sector_leader=row["sector_i"],    sector_follower=row["sector_j"],
            frequency=round(float(row["frequency"]), 3),
            half_life=round(float(row["half_life"]), 1),
            sharpness=round(float(row["sharpness"]), 3),
            interpretation=_interpretation(row),
        ))
    return results


# ── Function 2: Recommendation engine ────────────────────────────────────────

def get_recommendations(
    user_tickers: list,
    network_df: pd.DataFrame,
    top_n: int = 10,
    min_signal_strength: float = 55.0,
    signal_weight: float = 0.45,
    centrality_weight: float = 0.20,
    coverage_weight: float = 0.15,
    sector_diversity_weight: float = 0.20,
) -> list[Recommendation]:
    tickers = set(_normalize_tickers(user_tickers))
    if not tickers:
        return []

    meta = get_ticker_metadata(network_df)
    sector_dist = _sector_distribution(list(tickers), meta)
    total = len(tickers)

    involves = network_df["ticker_i"].isin(tickers) | network_df["ticker_j"].isin(tickers)
    filtered = network_df[involves & (network_df["signal_strength"] >= min_signal_strength)].copy()
    if filtered.empty:
        return []

    candidates: dict = {}
    for _, row in filtered.iterrows():
        ti, tj = row["ticker_i"], row["ticker_j"]
        ti_in, tj_in = ti in tickers, tj in tickers
        if ti_in and tj_in:
            continue
        if ti_in and not tj_in:
            cand, holding, direction = tj, ti, "follows_your_holdings"
        elif tj_in and not ti_in:
            cand, holding, direction = ti, tj, "leads_your_holdings"
        else:
            continue

        if cand not in candidates:
            candidates[cand] = {"signal_strengths":[],"related_holdings":[],"directions":[],"centrality":0.0,"sector":"Unknown"}

        candidates[cand]["signal_strengths"].append(float(row["signal_strength"]))
        candidates[cand]["related_holdings"].append(holding)
        candidates[cand]["directions"].append(direction)
        if cand == ti:
            candidates[cand]["centrality"] = max(candidates[cand]["centrality"], float(row["centrality_i"]))
            candidates[cand]["sector"] = row["sector_i"]
        else:
            candidates[cand]["centrality"] = max(candidates[cand]["centrality"], float(row["centrality_j"]))
            candidates[cand]["sector"] = row["sector_j"]

    if not candidates:
        return []

    max_centrality = max(d["centrality"] for d in candidates.values()) or 1.0
    max_coverage   = max(len(set(d["related_holdings"])) for d in candidates.values()) or 1.0

    recs = []
    for ticker, data in candidates.items():
        sigs    = data["signal_strengths"]
        related = list(dict.fromkeys(data["related_holdings"]))
        sector  = data["sector"]
        n_conn  = len(set(related))

        signal_score          = float(np.mean(sigs))
        centrality_score      = (data["centrality"] / max_centrality) * 100.0
        coverage_score        = (n_conn / max_coverage) * 100.0
        sector_frac           = sector_dist.get(sector, 0) / total if total else 0
        sector_diversity_score = max(0.0, (1.0 - sector_frac) * 100.0)

        composite = (
            signal_weight          * signal_score +
            centrality_weight      * centrality_score +
            coverage_weight        * coverage_score +
            sector_diversity_weight * sector_diversity_score
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
            reasoning=_rec_reasoning(ticker, related, direction, sector,
                                     signal_score, sector_diversity_score, n_conn),
        ))

    recs.sort(key=lambda r: r.composite_score, reverse=True)
    return recs[:top_n]