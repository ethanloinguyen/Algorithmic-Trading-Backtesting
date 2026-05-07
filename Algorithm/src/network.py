"""
network.py
----------
Builds directed graph from significant pairs.
Computes centrality metrics.
Tracks top-K node stability across windows.

Graph:
    Nodes = tickers
    Edges = significant pairs (directed: i leads j)
    Edge weights = signal_strength

Centrality:
    Eigenvector centrality (weighted by signal_strength)
    Out-degree centrality
"""

import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from Algorithm.src.bq_io import get_client, full_table, write_dataframe
from Algorithm.src.config_loader import get_config

logger = logging.getLogger(__name__)


def build_directed_graph(
    final_network_df: pd.DataFrame,
    min_signal_strength: float = 0.0,
) -> nx.DiGraph:
    """
    Build directed graph from final_network DataFrame.

    Edge direction: ticker_i → ticker_j (i leads j by best_lag days)
    Edge weight: signal_strength

    Parameters
    ----------
    final_network_df : DataFrame with ticker_i, ticker_j, signal_strength, best_lag
    min_signal_strength : filter threshold

    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()
    df = final_network_df[final_network_df["signal_strength"] >= min_signal_strength]

    for _, row in df.iterrows():
        ti = row["ticker_i"]
        tj = row["ticker_j"]
        G.add_node(ti, sector=row.get("sector_i", "Unknown"))
        G.add_node(tj, sector=row.get("sector_j", "Unknown"))
        G.add_edge(
            ti, tj,
            weight=float(row["signal_strength"]),
            lag=int(row["best_lag"]),
            predicted_sharpe=float(row.get("predicted_sharpe", 0.0)),
        )

    logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_centrality(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute node-level centrality metrics.

    Metrics:
    - out_degree_centrality: fraction of possible outgoing edges used
    - in_degree_centrality: fraction of possible incoming edges used
    - eigenvector_centrality: importance accounting for neighbor importance
      (uses weighted edges)

    Returns
    -------
    DataFrame with ticker, out_degree, in_degree, eigenvector_centrality
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    out_deg = nx.out_degree_centrality(G)
    in_deg = nx.in_degree_centrality(G)

    # Eigenvector centrality on undirected version (more stable)
    try:
        G_undirected = G.to_undirected()
        eigen_cent = nx.eigenvector_centrality_numpy(G_undirected, weight="weight")
    except Exception:
        logger.warning("Eigenvector centrality failed, using PageRank fallback.")
        eigen_cent = nx.pagerank(G, weight="weight")

    centrality_df = pd.DataFrame({
        "ticker": list(G.nodes()),
        "out_degree_centrality": [out_deg.get(n, 0.0) for n in G.nodes()],
        "in_degree_centrality": [in_deg.get(n, 0.0) for n in G.nodes()],
        "eigenvector_centrality": [eigen_cent.get(n, 0.0) for n in G.nodes()],
    })

    centrality_df = centrality_df.sort_values("eigenvector_centrality", ascending=False)
    return centrality_df


def get_top_k_central_nodes(centrality_df: pd.DataFrame, k: int = 20) -> List[str]:
    """Return top-K nodes by eigenvector centrality."""
    if centrality_df.empty:
        return []
    return centrality_df.head(k)["ticker"].tolist()


def compute_centrality_persistence(
    top_k_current: List[str],
    top_k_previous: List[str],
) -> float:
    """
    Measure stability of top-K central nodes across consecutive windows.
    Returns Jaccard similarity between the two top-K sets.

    Score of 1.0 = identical top-K.
    Score of 0.0 = completely different.
    """
    if not top_k_current or not top_k_previous:
        return 0.0
    set_curr = set(top_k_current)
    set_prev = set(top_k_previous)
    intersection = len(set_curr & set_prev)
    union = len(set_curr | set_prev)
    return intersection / union if union > 0 else 0.0


def build_network_json(
    final_network_df: pd.DataFrame,
    centrality_df: pd.DataFrame,
) -> dict:
    """
    Build JSON structure for frontend vis-network / react-force-graph.

    Returns
    -------
    {
        "nodes": [{"id": ticker, "label": ticker, "sector": ..., "centrality": ...}],
        "edges": [{"source": ti, "target": tj, "weight": ..., "lag": ...}]
    }
    """
    centrality_map = {}
    if not centrality_df.empty:
        centrality_map = dict(zip(
            centrality_df["ticker"],
            centrality_df["eigenvector_centrality"]
        ))

    # Collect all unique tickers
    tickers_i = final_network_df[["ticker_i", "sector_i"]].rename(
        columns={"ticker_i": "ticker", "sector_i": "sector"}
    )
    tickers_j = final_network_df[["ticker_j", "sector_j"]].rename(
        columns={"ticker_j": "ticker", "sector_j": "sector"}
    )
    all_tickers = pd.concat([tickers_i, tickers_j]).drop_duplicates("ticker")

    nodes = []
    for _, row in all_tickers.iterrows():
        ticker = row["ticker"]
        nodes.append({
            "id": ticker,
            "label": ticker,
            "sector": row.get("sector", "Unknown"),
            "centrality": round(centrality_map.get(ticker, 0.0), 4),
        })

    edges = []
    for _, row in final_network_df.iterrows():
        edges.append({
            "source": row["ticker_i"],
            "target": row["ticker_j"],
            "weight": round(float(row["signal_strength"]), 2),
            "lag": int(row["best_lag"]),
            "predicted_sharpe": round(float(row.get("predicted_sharpe", 0.0)), 4),
        })

    return {"nodes": nodes, "edges": edges}


def run_network_pipeline(as_of_date: date = None) -> Tuple[pd.DataFrame, dict]:
    """
    Full network computation pipeline:
    1. Load final_network for as_of_date
    2. Build graph
    3. Compute centrality
    4. Return centrality DataFrame and JSON for API

    Parameters
    ----------
    as_of_date : date, defaults to today

    Returns
    -------
    (centrality_df, network_json)
    """
    if as_of_date is None:
        as_of_date = date.today()

    client = get_client()
    query = f"""
        SELECT *
        FROM `{full_table('final_network')}`
        WHERE as_of_date = '{as_of_date}'
    """
    network_df = client.query(query).to_dataframe()

    if network_df.empty:
        logger.warning(f"No network data found for {as_of_date}")
        return pd.DataFrame(), {}

    G = build_directed_graph(network_df)
    centrality_df = compute_centrality(G)
    network_json = build_network_json(network_df, centrality_df)

    logger.info(
        f"Network pipeline complete: "
        f"{len(network_json['nodes'])} nodes, {len(network_json['edges'])} edges"
    )
    return centrality_df, network_json
