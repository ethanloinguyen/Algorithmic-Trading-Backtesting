"""
main.py — FastAPI Backend
--------------------------
Serves pre-computed results from BigQuery to the React frontend.
All endpoints are READ-ONLY. No live computation on request.

Endpoints:
    GET /pairs/top                  — Top N pairs by signal strength
    GET /pairs/search               — Search pairs by ticker
    GET /pairs/{ticker_i}/{ticker_j} — Pair detail
    GET /pairs/by-sector            — Filter by sector
    GET /network                    — Directed graph (nodes + edges)
    GET /network/centrality         — Top central nodes
    GET /features/importance        — β weights with CI bands
    GET /charts/cumulative-return   — Monte Carlo cone data
    GET /charts/decile-performance  — Stability decile vs OOS Sharpe
    GET /charts/lag-distribution    — Lag histogram
    GET /charts/centrality-rolling  — Centrality persistence over time
    GET /meta/config                — Current pipeline config (public subset)
    GET /meta/last-update           — Last pipeline run info
    GET /health                     — Health check
"""

import logging
from datetime import date, datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import bigquery

from src.bq_io import get_client, full_table, read_model_weights
from src.config_loader import load_config, get_config

# ── App Setup ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_config()
cfg = get_config()

app = FastAPI(
    title="Lead-Lag Signal API",
    description="Quantitative lead-lag relationship signals for equity pairs",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Response Models ───────────────────────────────────────────────────────────

class PairSummary(BaseModel):
    ticker_i: str
    ticker_j: str
    best_lag: Optional[int]
    predicted_sharpe: Optional[float]
    signal_strength: Optional[float]
    oos_sharpe_net: Optional[float]
    sector_i: Optional[str]
    sector_j: Optional[str]
    rank: Optional[int]
    mean_dcor: Optional[float]
    frequency: Optional[float]
    half_life: Optional[float]
    sharpness: Optional[float]
    centrality_i: Optional[float]
    centrality_j: Optional[float]


class NetworkNode(BaseModel):
    id: str
    label: str
    sector: str
    centrality: float


class NetworkEdge(BaseModel):
    source: str
    target: str
    weight: float
    lag: int
    predicted_sharpe: float


class NetworkResponse(BaseModel):
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]


class WeightFeature(BaseModel):
    feature: str
    weight: float
    ci_lower: float
    ci_upper: float


class ModelWeightsResponse(BaseModel):
    model_version: str
    refit_date: Optional[str]
    r2: Optional[float]
    n_pairs: Optional[int]
    features: List[WeightFeature]


# ── Helper ────────────────────────────────────────────────────────────────────

def get_latest_as_of_date() -> Optional[date]:
    """Return the most recent as_of_date in final_network."""
    client = get_client()
    query = f"""
        SELECT MAX(as_of_date) AS latest
        FROM `{full_table('final_network')}`
    """
    result = client.query(query).to_dataframe()
    if result.empty or result["latest"].iloc[0] is None:
        return None
    return result["latest"].iloc[0]


def query_final_network(
    min_score: float = 0,
    sector: Optional[str] = None,
    lag: Optional[int] = None,
    ticker: Optional[str] = None,
    intra_sector_only: bool = False,
    inter_sector_only: bool = False,
    min_market_cap: Optional[float] = None,
    top_n: int = 100,
    as_of_date: Optional[date] = None,
) -> pd.DataFrame:
    """Parameterized query for final_network with all filters."""
    if as_of_date is None:
        as_of_date = get_latest_as_of_date()
    if as_of_date is None:
        return pd.DataFrame()

    client = get_client()
    conditions = [f"as_of_date = '{as_of_date}'"]

    if min_score > 0:
        conditions.append(f"signal_strength >= {min_score}")
    if sector:
        conditions.append(f"(sector_i = '{sector}' OR sector_j = '{sector}')")
    if lag is not None:
        conditions.append(f"best_lag = {lag}")
    if ticker:
        conditions.append(f"(ticker_i = '{ticker}' OR ticker_j = '{ticker}')")
    if intra_sector_only:
        conditions.append("sector_i = sector_j")
    if inter_sector_only:
        conditions.append("sector_i != sector_j")

    where_clause = " AND ".join(conditions)
    query = f"""
        SELECT *
        FROM `{full_table('final_network')}`
        WHERE {where_clause}
        ORDER BY signal_strength DESC
        LIMIT {min(top_n, cfg['api']['max_top_n'])}
    """
    return client.query(query).to_dataframe()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Service health check."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/pairs/top", response_model=List[PairSummary])
def get_top_pairs(
    n: int = Query(default=100, le=1000, description="Number of pairs to return"),
    min_score: float = Query(default=0, ge=0, le=100, description="Minimum signal strength"),
    sector: Optional[str] = Query(default=None),
    lag: Optional[int] = Query(default=None, ge=1, le=5),
    intra_sector_only: bool = Query(default=False),
    inter_sector_only: bool = Query(default=False),
):
    """Return top N pairs ranked by signal strength with optional filters."""
    df = query_final_network(
        min_score=min_score,
        sector=sector,
        lag=lag,
        intra_sector_only=intra_sector_only,
        inter_sector_only=inter_sector_only,
        top_n=n,
    )
    if df.empty:
        return []
    return df.replace({float("nan"): None}).to_dict(orient="records")


@app.get("/pairs/search", response_model=List[PairSummary])
def search_pairs(
    ticker: str = Query(..., description="Search for pairs involving this ticker"),
    min_score: float = Query(default=0),
):
    """Search pairs by ticker (returns pairs where ticker is leader or follower)."""
    df = query_final_network(ticker=ticker.upper(), min_score=min_score, top_n=200)
    if df.empty:
        return []
    return df.replace({float("nan"): None}).to_dict(orient="records")


@app.get("/pairs/by-sector", response_model=List[PairSummary])
def get_pairs_by_sector(
    sector: str = Query(..., description="Sector name (e.g. Technology)"),
    min_score: float = Query(default=0),
    n: int = Query(default=50, le=500),
):
    """Return top pairs involving a specific sector."""
    df = query_final_network(sector=sector, min_score=min_score, top_n=n)
    if df.empty:
        return []
    return df.replace({float("nan"): None}).to_dict(orient="records")


@app.get("/pairs/{ticker_i}/{ticker_j}")
def get_pair_detail(ticker_i: str, ticker_j: str):
    """Return full detail for a specific pair."""
    client = get_client()
    ti = ticker_i.upper()
    tj = ticker_j.upper()

    query = f"""
        SELECT *
        FROM `{full_table('final_network')}`
        WHERE (ticker_i = '{ti}' AND ticker_j = '{tj}')
           OR (ticker_i = '{tj}' AND ticker_j = '{ti}')
        ORDER BY as_of_date DESC
        LIMIT 1
    """
    df = client.query(query).to_dataframe()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Pair {ti}/{tj} not found")

    return df.replace({float("nan"): None}).to_dict(orient="records")[0]


@app.get("/network", response_model=NetworkResponse)
def get_network(
    min_score: float = Query(default=50, description="Minimum signal strength for edges"),
    sector: Optional[str] = Query(default=None),
    max_edges: int = Query(default=500, le=2000),
):
    """Return directed network graph (nodes + edges) for frontend visualization."""
    df = query_final_network(min_score=min_score, sector=sector, top_n=max_edges)
    if df.empty:
        return {"nodes": [], "edges": []}

    # Build nodes
    tickers_i = df[["ticker_i", "sector_i", "centrality_i"]].rename(
        columns={"ticker_i": "ticker", "sector_i": "sector", "centrality_i": "centrality"}
    )
    tickers_j = df[["ticker_j", "sector_j", "centrality_j"]].rename(
        columns={"ticker_j": "ticker", "sector_j": "sector", "centrality_j": "centrality"}
    )
    all_tickers = pd.concat([tickers_i, tickers_j]).drop_duplicates("ticker")
    all_tickers = all_tickers.fillna({"sector": "Unknown", "centrality": 0.0})

    nodes = [
        NetworkNode(
            id=row["ticker"],
            label=row["ticker"],
            sector=row["sector"],
            centrality=round(float(row["centrality"]), 4),
        )
        for _, row in all_tickers.iterrows()
    ]

    edges = [
        NetworkEdge(
            source=row["ticker_i"],
            target=row["ticker_j"],
            weight=round(float(row["signal_strength"] or 0), 2),
            lag=int(row["best_lag"] or 1),
            predicted_sharpe=round(float(row["predicted_sharpe"] or 0), 4),
        )
        for _, row in df.iterrows()
    ]

    return NetworkResponse(nodes=nodes, edges=edges)


@app.get("/network/centrality")
def get_centrality(top_k: int = Query(default=20, le=100)):
    """Return top-K central nodes by eigenvector centrality."""
    client = get_client()
    as_of_date = get_latest_as_of_date()
    if not as_of_date:
        return []

    query = f"""
        WITH node_centrality AS (
            SELECT ticker_i AS ticker, MAX(centrality_i) AS centrality, MAX(sector_i) AS sector
            FROM `{full_table('final_network')}`
            WHERE as_of_date = '{as_of_date}'
            GROUP BY ticker_i
        )
        SELECT ticker, centrality, sector
        FROM node_centrality
        ORDER BY centrality DESC
        LIMIT {top_k}
    """
    df = client.query(query).to_dataframe()
    return df.replace({float("nan"): None}).to_dict(orient="records")


@app.get("/features/importance", response_model=ModelWeightsResponse)
def get_feature_importance():
    """Return current β weights with bootstrap CI bands."""
    weights_df = read_model_weights()
    if weights_df.empty:
        raise HTTPException(status_code=404, detail="No model weights found")

    features = []
    for _, row in weights_df.iterrows():
        if row["feature"] != "intercept":
            features.append(WeightFeature(
                feature=row["feature"],
                weight=round(float(row["weight"]), 6),
                ci_lower=round(float(row["ci_lower"]), 6),
                ci_upper=round(float(row["ci_upper"]), 6),
            ))

    first_row = weights_df.iloc[0]
    return ModelWeightsResponse(
        model_version=str(first_row.get("model_version", "unknown")),
        refit_date=str(first_row.get("refit_date", "")),
        r2=float(first_row.get("r2", 0)) if first_row.get("r2") else None,
        n_pairs=int(first_row.get("n_pairs", 0)) if first_row.get("n_pairs") else None,
        features=features,
    )


@app.get("/charts/cumulative-return")
def get_cumulative_return(
    ticker_i: str = Query(...),
    ticker_j: str = Query(...),
):
    """
    Return Monte Carlo cone data for a specific pair.
    Includes actual cumulative return + percentile bands.
    """
    client = get_client()
    ti = ticker_i.upper()
    tj = ticker_j.upper()

    # Try to get Monte Carlo data (if stored)
    # For now, compute from OOS returns directly
    query = f"""
        SELECT oos_date, strategy_return_net
        FROM `{full_table('oos_strategy_returns')}`
        WHERE ticker_i = '{ti}' AND ticker_j = '{tj}'
        ORDER BY oos_date
    """
    df = client.query(query).to_dataframe()
    if df.empty:
        raise HTTPException(status_code=404, detail="No OOS data for this pair")

    # Compute cumulative return
    returns = df["strategy_return_net"].values
    cum_returns = np.cumprod(1 + returns) - 1

    # Simple percentile bands from historical returns (no live MC)
    mc_cfg = get_config()["monte_carlo"]
    conf_levels = mc_cfg["confidence_levels"]

    from src.monte_carlo import block_bootstrap_returns, compute_cumulative_paths, compute_cone_percentiles
    import numpy as np
    rng = np.random.default_rng(42)
    sim_returns = block_bootstrap_returns(returns, 1000, mc_cfg["block_size"], len(returns), rng)
    cum_paths = compute_cumulative_paths(sim_returns)
    bands = compute_cone_percentiles(cum_paths, conf_levels)
    mean_path = cum_paths.mean(axis=0)
    prob_positive = (cum_paths > 0).mean(axis=0)

    result = []
    for t in range(len(returns)):
        row = {
            "day": t + 1,
            "date": str(df["oos_date"].iloc[t]),
            "actual_cumulative": round(float(cum_returns[t]), 6),
            "mean_path": round(float(mean_path[t]), 6),
            "prob_positive": round(float(prob_positive[t]), 4),
        }
        for pct in conf_levels:
            row[f"pct_{int(pct * 100)}"] = round(float(bands[pct][t]), 6)
        result.append(row)

    return result


@app.get("/charts/decile-performance")
def get_decile_performance():
    """
    Return stability decile vs OOS Sharpe data for the decile chart.
    Pairs are binned into 10 deciles by signal_strength.
    """
    client = get_client()
    as_of_date = get_latest_as_of_date()
    if not as_of_date:
        return []

    query = f"""
        SELECT signal_strength, oos_sharpe_net
        FROM `{full_table('final_network')}`
        WHERE as_of_date = '{as_of_date}'
          AND oos_sharpe_net IS NOT NULL
          AND signal_strength IS NOT NULL
    """
    df = client.query(query).to_dataframe()
    if df.empty:
        return []

    df["decile"] = pd.qcut(df["signal_strength"], q=10, labels=False, duplicates="drop") + 1

    decile_summary = df.groupby("decile").agg(
        mean_oos_sharpe=("oos_sharpe_net", "mean"),
        median_oos_sharpe=("oos_sharpe_net", "median"),
        n_pairs=("oos_sharpe_net", "count"),
        mean_signal_strength=("signal_strength", "mean"),
    ).reset_index()

    return decile_summary.replace({float("nan"): None}).to_dict(orient="records")


@app.get("/charts/lag-distribution")
def get_lag_distribution():
    """Return histogram of best lag distribution across significant pairs."""
    client = get_client()
    as_of_date = get_latest_as_of_date()
    if not as_of_date:
        return []

    query = f"""
        SELECT best_lag, COUNT(*) AS count,
               AVG(signal_strength) AS mean_signal_strength,
               AVG(oos_sharpe_net) AS mean_oos_sharpe
        FROM `{full_table('final_network')}`
        WHERE as_of_date = '{as_of_date}'
          AND best_lag IS NOT NULL
        GROUP BY best_lag
        ORDER BY best_lag
    """
    df = client.query(query).to_dataframe()
    return df.replace({float("nan"): None}).to_dict(orient="records")


@app.get("/charts/centrality-rolling")
def get_centrality_rolling(top_k: int = Query(default=10)):
    """
    Return rolling top-K central nodes across past windows.
    Used for centrality persistence visualization.
    """
    client = get_client()
    # Get all available as_of_dates
    query = f"""
        SELECT DISTINCT as_of_date
        FROM `{full_table('final_network')}`
        ORDER BY as_of_date DESC
        LIMIT 24
    """
    dates_df = client.query(query).to_dataframe()
    if dates_df.empty:
        return []

    result = []
    for dt in dates_df["as_of_date"]:
        query_cent = f"""
            WITH node_centrality AS (
                SELECT ticker_i AS ticker, MAX(centrality_i) AS centrality
                FROM `{full_table('final_network')}`
                WHERE as_of_date = '{dt}'
                GROUP BY ticker_i
            )
            SELECT ticker, centrality
            FROM node_centrality
            ORDER BY centrality DESC
            LIMIT {top_k}
        """
        cent_df = client.query(query_cent).to_dataframe()
        if not cent_df.empty:
            result.append({
                "date": str(dt),
                "top_nodes": cent_df.to_dict(orient="records"),
            })

    return result


@app.get("/meta/config")
def get_public_config():
    """Return public subset of pipeline configuration."""
    cfg = get_config()
    return {
        "universe": {
            "start_date": cfg["universe"]["start_date"],
            "end_date": cfg["universe"]["end_date"],
        },
        "windows": cfg["windows"],
        "lags": cfg["lags"],
        "fdr": {"alpha": cfg["fdr"]["alpha"]},
        "strategy": {
            "zscore_lookback_days": cfg["strategy"]["zscore_lookback_days"],
            "zscore_threshold": cfg["strategy"]["zscore_threshold"],
            "transaction_cost_bps": cfg["strategy"]["transaction_cost_bps"],
        },
        "model": {
            "refit_schedule": cfg["model"]["refit_schedule"],
        },
    }


@app.get("/meta/last-update")
def get_last_update():
    """Return information about the last pipeline run."""
    client = get_client()
    query = f"""
        SELECT run_id, run_date, window_start, window_end,
               n_significant_pairs, status, duration_seconds
        FROM `{full_table('pipeline_run_log')}`
        WHERE status = 'COMPLETE'
        ORDER BY run_date DESC
        LIMIT 1
    """
    df = client.query(query).to_dataframe()
    if df.empty:
        return {"status": "No completed runs found"}
    return df.replace({float("nan"): None}).to_dict(orient="records")[0]
