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
    query    = f"SELECT * FROM {table} WHERE as_of_date = (SELECT MAX(as_of_date) FROM {table})"
    return client.query(query).to_dataframe()


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
    )
    from dataclasses import asdict

    normalized = _normalize_tickers(tickers)
    if not normalized:
        return {
            "tickers_analyzed": [], "unknown_tickers": [],
            "overlaps": [], "signal_recommendations": [],
            "independent_recommendations": [],
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
    }