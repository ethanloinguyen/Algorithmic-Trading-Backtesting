# backend/app/models/model_inference.py
from pydantic import BaseModel


class LeaderStock(BaseModel):
    rank:   int    # 1-based rank
    symbol: str    # ticker
    sector: str    # from ticker_metadata
    lag:    int    # lag in trading days (1–10)
    signal: float  # model score (0–1, higher = stronger leader)


class SectorCount(BaseModel):
    sector: str
    count:  int


class ModelPeriodResult(BaseModel):
    period:        str               # "3d" | "6d" | "10d"
    leaders:       list[LeaderStock]
    sector_counts: list[SectorCount]


class ModelResponse(BaseModel):
    target:  str
    results: list[ModelPeriodResult]


class ModelInfoResponse(BaseModel):
    tickers:   list[str]
    n_tickers: int
    sectors:   list[str]
    lookback:  int
    l_max:     int