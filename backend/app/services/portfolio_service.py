# backend/app/services/portfolio_service.py
from __future__ import annotations
import os
import pandas as pd

_USE_MOCK = os.environ.get("USE_MOCK_DATA", "true").lower() == "true"


def _get_network_mock() -> pd.DataFrame:
    from app.services.mock_final_network import get_mock_final_network
    return get_mock_final_network()


def _get_network_bq() -> pd.DataFrame:
    from google.cloud import bigquery
    from app.core.bigquery import get_bq_client
    from app.core.config import get_settings
    settings = get_settings()
    client   = get_bq_client()
    table    = f"`{settings.gcp_project_id}.{settings.bq_dataset}.final_network`"
    # Explicit column list guards against missing centrality_i/j columns
    # in older pipeline runs — fills them with 0.0 if absent via COALESCE
    query = f"""
        SELECT
            as_of_date,
            ticker_i,
            ticker_j,
            best_lag,
            mean_dcor,
            variance_dcor,
            frequency,
            half_life,
            sharpness,
            predicted_sharpe,
            signal_strength,
            oos_sharpe_net,
            oos_dcor,
            sector_i,
            sector_j,
            rank,
            COALESCE(centrality_i, 0.0) AS centrality_i,
            COALESCE(centrality_j, 0.0) AS centrality_j
        FROM {table}
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM {table})
    """
    df = client.query(query).to_dataframe()
    # Ensure numeric columns that may come back as object are cast correctly
    for col in ["centrality_i", "centrality_j", "frequency", "half_life",
                "signal_strength", "mean_dcor", "oos_sharpe_net", "predicted_sharpe",
                "variance_dcor", "sharpness"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def get_final_network() -> pd.DataFrame:
    return _get_network_mock() if _USE_MOCK else _get_network_bq()


def run_portfolio_analysis(
    tickers: list[str],
    top_n: int = 10,
    min_signal: float = 55.0,
) -> dict:
    from app.services.portfolio_engine import (
        _normalize_tickers,
        get_ticker_metadata,
        analyze_portfolio_overlap,
        get_signal_recommendations,
        get_independent_recommendations,
        get_holdings_sectors,
    )
    from dataclasses import asdict

    normalized = _normalize_tickers(tickers)
    if not normalized:
        return {
            "tickers_analyzed": [], "unknown_tickers": [],
            "overlaps": [], "signal_recommendations": [],
            "independent_recommendations": [],
            "holdings_sectors": {},
        }

    df   = get_final_network()
    meta = get_ticker_metadata(df)

    known   = [t for t in normalized if t in meta.index]
    unknown = [t for t in normalized if t not in meta.index]

    return {
        "tickers_analyzed":            known,
        "unknown_tickers":             unknown,
        "overlaps":                    [asdict(o) for o in analyze_portfolio_overlap(normalized, df)],
        "signal_recommendations":      [asdict(r) for r in get_signal_recommendations(normalized, df, top_n=top_n, min_signal_strength=min_signal)],
        "independent_recommendations": [asdict(r) for r in get_independent_recommendations(normalized, df, top_n=top_n)],
        # Sector map for every known holding — used by frontend SectorDonut
        # regardless of whether any overlaps exist
        "holdings_sectors":            get_holdings_sectors(normalized, df),
    }