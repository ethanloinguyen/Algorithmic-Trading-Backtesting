# backend/app/services/model_service.py
"""
DeltaLag inference service — deltalag_epoch0020.pt

Architecture (confirmed from state_dict inspection):
  • LSTM encoder: input [L, 6] → hidden [N=128]  (single layer, no sector emb)
  • W_Q / W_K: [128, 128] linear projections, no bias
  • lag_bias:  [l_max=10] learned per-lag scalar
  • MLP head:  [12 → 64 → 32 → 1]
      input = concat(last_hidden_target, last_hidden_leader)  dim=12
      This scores (target, leader) pairs jointly

Feature vector (F=6) per day, same order as training:
  [close/open,  high/open,  low/open,  log_return,  log(volume),  adj_close/open]

Normalisation: sklearn RobustScaler stored in checkpoint['scaler']
  X_norm = (X - center_) / scale_

Lookback L=40 trading days per stock.

Model file expected at:  backend/models/deltalag_epoch0020.pt
"""
from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from google.cloud import bigquery

from app.core.bigquery import get_bq_client
from app.core.config import get_settings
from app.models.model_inference import LeaderStock, ModelPeriodResult, SectorCount

# ── Paths ─────────────────────────────────────────────────────────────────────

_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
_PT_PATH   = _MODEL_DIR / "deltalag_epoch0020.pt"

# ── Architecture ──────────────────────────────────────────────────────────────

class DeltaLagEncoder(nn.Module):
    """LSTM encoder: maps [batch, L, F] → [batch, N]"""
    def __init__(self, f: int = 6, n: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(f, n, batch_first=True)
        self.norm = nn.LayerNorm(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)          # h: [1, batch, N]
        return self.norm(h.squeeze(0))     # [batch, N]


class DeltaLagModel(nn.Module):
    """
    Full DeltaLag model matching deltalag_epoch0020.pt state_dict:
      log_temp   scalar
      lag_bias   [l_max]
      encoder.*  LSTM + LayerNorm
      W_Q        [N, N]
      W_K        [N, N]
      mlp        64 → 32 → 1  (input dim 12 = 6+6, for concat approach)
    """
    def __init__(self, f: int = 6, n: int = 128, l_max: int = 10):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(0.0))
        self.lag_bias = nn.Parameter(torch.zeros(l_max))
        self.encoder  = DeltaLagEncoder(f, n)
        self.W_Q      = nn.Linear(n, n, bias=False)
        self.W_K      = nn.Linear(n, n, bias=False)
        # MLP: input is [h_target; h_leader] projected through attention, dim=12
        # State dict has mlp.0.weight [64,12], mlp.3.weight [32,64], mlp.6.weight [1,32]
        self.mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, L, F] → [batch, N]"""
        return self.encoder(x)

    def score(
        self,
        h_target: torch.Tensor,  # [1, N]
        h_leader: torch.Tensor,  # [1, N]
        lag_idx:  int,
    ) -> float:
        """
        Compute relevance score for (target, leader) at a given lag.

        Attention logit: (W_Q h_target) · (W_K h_leader) / sqrt(N)
                         + lag_bias[lag_idx]
        Final score:     sigmoid of logit (bounded 0–1)
        """
        q     = self.W_Q(h_target)    # [1, N]
        k     = self.W_K(h_leader)    # [1, N]
        temp  = self.log_temp.exp()
        logit = (q * k).sum(-1) / (math.sqrt(q.size(-1)) * temp)
        logit = logit + self.lag_bias[lag_idx]
        return torch.sigmoid(logit).item()


# ── Loader ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_checkpoint():
    if not _PT_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {_PT_PATH}\n"
            "Place deltalag_epoch0020.pt in backend/models/"
        )

    ck = torch.load(_PT_PATH, map_location="cpu", weights_only=False)

    hparams = ck["hparams"]
    f       = hparams["F"]        # 6
    n       = hparams["N"]        # 128
    l_max   = hparams["l_max"]    # 10
    lookback = hparams["L"]       # 40

    model = DeltaLagModel(f=f, n=n, l_max=l_max)
    model.load_state_dict(ck["model_state"])
    model.eval()

    scaler  = ck["scaler"]        # sklearn RobustScaler
    tickers = ck["tickers"]       # list of 1690 symbols

    return model, scaler, tickers, lookback, l_max


def get_model_info() -> dict:
    _, _, tickers, lookback, l_max = _load_checkpoint()
    # Sectors come from BigQuery ticker_metadata
    return {
        "tickers":   tickers,
        "n_tickers": len(tickers),
        "sectors":   [
            "Communication", "Consumer Discretionary", "Consumer Staples",
            "Energy", "Financials", "Health Care", "Industrials",
            "Information Technology", "Materials", "Real Estate", "Utilities",
        ],
        "lookback": lookback,
        "l_max":    l_max,
    }


# ── BigQuery data fetch ───────────────────────────────────────────────────────

def _fetch_features(
    symbols:  list[str],
    n_rows:   int,
) -> dict[str, np.ndarray]:
    """
    Fetch the most recent `n_rows` trading days for each symbol.
    Returns dict: symbol → float32 array [n_rows, 6].

    Feature order (matches RobustScaler training):
      0: close / open
      1: high  / open
      2: low   / open
      3: log_return
      4: log(volume)
      5: adj_close / open
    """
    client   = get_bq_client()
    settings = get_settings()

    query = f"""
        SELECT ticker, date, open, high, low, close, adj_close, volume, log_return
        FROM (
            SELECT
                ticker, date, open, high, low, close, adj_close, volume, log_return,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM {settings.fq_market_data}
            WHERE ticker IN UNNEST(@symbols)
              AND open > 0
        )
        WHERE rn <= @n_rows
        ORDER BY ticker, date ASC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("symbols", "STRING", [s.upper() for s in symbols]),
            bigquery.ScalarQueryParameter("n_rows",  "INT64",  n_rows),
        ]
    )

    rows = list(client.query(query, job_config=job_config).result())

    # Group rows by ticker
    by_ticker: dict[str, list] = defaultdict(list)
    for row in rows:
        by_ticker[row.ticker].append(row)

    result: dict[str, np.ndarray] = {}
    for sym, sym_rows in by_ticker.items():
        sym_rows = sym_rows[-n_rows:]       # safety trim
        arr = np.zeros((len(sym_rows), 6), dtype=np.float32)
        for i, row in enumerate(sym_rows):
            o = float(row.open) or 1.0
            arr[i, 0] = float(row.close)      / o
            arr[i, 1] = float(row.high)       / o
            arr[i, 2] = float(row.low)        / o
            arr[i, 3] = float(row.log_return) if row.log_return else 0.0
            arr[i, 4] = math.log(max(float(row.volume), 1))
            arr[i, 5] = float(row.adj_close)  / o
        result[sym] = arr

    return result


def _fetch_sectors(symbols: list[str]) -> dict[str, str]:
    """Fetch sector for each symbol from ticker_metadata."""
    client   = get_bq_client()
    settings = get_settings()

    query = f"""
        SELECT ticker, sector
        FROM {settings.fq_ticker_metadata}
        WHERE ticker IN UNNEST(@symbols)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("symbols", "STRING", [s.upper() for s in symbols])
        ]
    )
    rows = client.query(query, job_config=job_config).result()
    return {row.ticker: (row.sector or "Unknown") for row in rows}


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalise(arr: np.ndarray, scaler) -> np.ndarray:
    """Apply RobustScaler: (X - center_) / scale_"""
    center = np.array(scaler.center_, dtype=np.float32)
    scale  = np.array(scaler.scale_,  dtype=np.float32)
    scale  = np.where(scale < 1e-8, 1.0, scale)
    return (arr - center) / scale


# ── Period → lag mapping ──────────────────────────────────────────────────────

# For each user-facing period, which lags (1-indexed) to consider.
# l_max = 10 so we can't go beyond lag 10.
PERIOD_LAGS: dict[str, list[int]] = {
    "3d":  [1, 2, 3],
    "6d":  [1, 2, 3, 4, 5, 6],
    "10d": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

TOP_N = 10   # leaders to return per period


# ── Main inference ────────────────────────────────────────────────────────────

def run_analysis(target_symbol: str) -> list[ModelPeriodResult]:
    """
    Run DeltaLag inference for a target symbol.

    Steps:
      1. Load model + checkpoint metadata.
      2. Fetch raw features from BigQuery for target + all 1690 tickers.
      3. Normalise with the checkpoint's RobustScaler.
      4. Encode target with LSTM.
      5. For each candidate and each lag, encode the lagged window and
         compute the (target, candidate, lag) attention score.
      6. Pick the best lag per candidate, sort by score, return top N.
      7. Build sector breakdown.
    """
    model, scaler, tickers, lookback, l_max = _load_checkpoint()

    target = target_symbol.upper()
    if target not in tickers:
        raise ValueError(
            f"'{target}' is not in the model's ticker universe ({len(tickers)} tickers). "
            "Use GET /api/model/info to see available tickers."
        )

    # Need lookback + l_max rows per ticker so we can shift windows
    n_fetch = lookback + l_max

    print(f"[model] Fetching {n_fetch} rows × {len(tickers)} tickers from BigQuery…")
    raw = _fetch_features([target] + tickers, n_fetch)

    if target not in raw:
        raise ValueError(f"No BigQuery data found for '{target}'.")

    # Fetch sectors for all candidates that have data
    candidates_with_data = [s for s in tickers if s != target and s in raw]
    print(f"[model] {len(candidates_with_data)} candidates have data. Fetching sectors…")
    sector_map = _fetch_sectors(candidates_with_data + [target])

    # Encode target using the most recent `lookback` rows
    target_raw    = raw[target][-lookback:]
    if len(target_raw) < lookback:
        raise ValueError(
            f"Target '{target}' only has {len(target_raw)} rows; need {lookback}."
        )
    target_norm   = _normalise(target_raw, scaler)
    target_tensor = torch.tensor(target_norm).unsqueeze(0)   # [1, L, F]

    with torch.no_grad():
        h_target = model.encode(target_tensor)   # [1, N]

    results: list[ModelPeriodResult] = []

    for period_label, lags in PERIOD_LAGS.items():
        scores: list[tuple[str, int, float]] = []   # (symbol, best_lag, score)

        for candidate in candidates_with_data:
            cand_raw = raw[candidate]
            if len(cand_raw) < lookback + 1:
                continue

            best_score = -1.0
            best_lag   = lags[0]

            for lag in lags:
                # Shift window back by `lag` days relative to target's window
                end   = len(cand_raw) - lag
                start = end - lookback
                if start < 0:
                    continue

                cand_norm   = _normalise(cand_raw[start:end], scaler)
                cand_tensor = torch.tensor(cand_norm).unsqueeze(0)   # [1, L, F]

                with torch.no_grad():
                    h_cand = model.encode(cand_tensor)   # [1, N]
                    s      = model.score(h_target, h_cand, lag - 1)  # lag_bias is 0-indexed

                if s > best_score:
                    best_score = s
                    best_lag   = lag

            scores.append((candidate, best_lag, round(best_score, 4)))

        # Sort by score descending, take top N
        scores.sort(key=lambda x: x[2], reverse=True)
        top = scores[:TOP_N]

        leaders = [
            LeaderStock(
                rank   = rank,
                symbol = sym,
                sector = sector_map.get(sym, "Unknown"),
                lag    = lag,
                signal = signal,
            )
            for rank, (sym, lag, signal) in enumerate(top, start=1)
        ]

        # Sector breakdown
        sc_map: dict[str, int] = {}
        for l in leaders:
            sc_map[l.sector] = sc_map.get(l.sector, 0) + 1
        sector_counts = [
            SectorCount(sector=s, count=c)
            for s, c in sorted(sc_map.items(), key=lambda x: -x[1])
        ]

        results.append(ModelPeriodResult(
            period        = period_label,
            leaders       = leaders,
            sector_counts = sector_counts,
        ))

    return results