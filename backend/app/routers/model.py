# backend/app/routers/model.py
"""
DeltaLag analysis endpoint — serves pre-computed results from
deltalag_webapp_data.json (1164 targets) and enriches with live prices.

The model was trained on the Russell 1000 + top-1000 Russell 2000 by weight.
All inference is already done; this router is a fast lookup + price enrichment.

Endpoints
---------
GET  /api/model/tickers          → list of valid target tickers
POST /api/model/analyze          → top-10 leaders + sector breakdown for a target
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# ── Load pre-computed data at startup ────────────────────────────────────────

_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

def _load_json(name: str) -> Any:
    path = _DATA_DIR / name
    if not path.exists():
        raise RuntimeError(f"Required data file not found: {path}")
    with open(path) as f:
        return json.load(f)


# Loaded once at import time (fast dict lookups at request time)
_webapp_data: dict[str, Any] = {}
_ticker_sector: dict[str, str] = {}   # symbol → GICS sector
_leader_tickers: list[str] = []

def _init():
    global _webapp_data, _ticker_sector, _leader_tickers
    _webapp_data    = _load_json("deltalag_webapp_data.json")
    meta            = _load_json("deltalag_russell2000_dual_meta.json")
    _ticker_sector  = meta["ticker_sector"]
    _leader_tickers = meta["leader_tickers"]

try:
    _init()
except RuntimeError as e:
    # Deferred — will fail at request time if data is missing
    import warnings
    warnings.warn(str(e))

# ── API helpers ───────────────────────────────────────────────────────────────

API_BASE = os.getenv("INTERNAL_API_URL", "http://localhost:4000")


async def _fetch_summaries(symbols: list[str]) -> dict[str, dict]:
    """Fetch live price/change/volume for a list of symbols in one call."""
    if not symbols:
        return {}
    q = ",".join(s.upper() for s in symbols)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{API_BASE}/api/stocks/summaries?symbols={q}")
            r.raise_for_status()
            data = r.json().get("data", [])
        return {item["symbol"]: item for item in data}
    except Exception:
        return {}


# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/model", tags=["model"])


class AnalyzeRequest(BaseModel):
    symbol: str


@router.get("/tickers")
async def list_tickers():
    """
    Return all valid target tickers with name and sector.
    The frontend uses this to populate the search autocomplete.
    """
    if not _webapp_data:
        raise HTTPException(status_code=503, detail="Model data not loaded")

    tickers = [
        {
            "symbol": sym,
            "name":   _ticker_sector_name(sym),
            "sector": entry["target_sector"],
        }
        for sym, entry in sorted(_webapp_data.items())
    ]
    return {"tickers": tickers}


@router.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Return pre-computed top-10 leaders for a target ticker,
    enriched with live prices from BigQuery.

    Response shape
    --------------
    {
      "target":        "AAPL",
      "target_sector": "Information Technology",
      "target_ic":     0.0139,
      "as_of_date":    "2024-12-30",
      "leaders": [
        {
          "rank":          1,
          "symbol":        "WTM",
          "name":          "White Mountains Insurance",
          "sector":        "Financials",
          "lag_days":      2,
          "attn_weight":   0.1173,
          "leader_ic":     -0.0294,
          "price":         "$145.20",
          "change":        "+0.53%",
          "volume":        "45.2K",
          "positive":      true
        }, ...
      ],
      "sector_breakdown": {
        "Financials":  { "count": 5, "total_weight": 0.5198 },
        "Materials":   { "count": 3, "total_weight": 0.3115 },
        "Industrials": { "count": 2, "total_weight": 0.1688 }
      }
    }
    """
    if not _webapp_data:
        raise HTTPException(status_code=503, detail="Model data not loaded")

    sym = req.symbol.upper().strip()
    if sym not in _webapp_data:
        raise HTTPException(
            status_code=422,
            detail=f"{sym} is not in the model's universe of {len(_webapp_data)} tickers.",
        )

    entry = _webapp_data[sym]
    raw_leaders = entry["top_leaders"]  # already ranked by attn_weight desc

    # De-duplicate: keep highest-weight entry per (leader_ticker, lag_days) pair.
    # The webapp data can have the same stock at multiple lag values; we show
    # the top-10 as-is (the model already handles deduplication).

    # Fetch live prices for all unique leader tickers
    unique_syms = list({l["leader_ticker"] for l in raw_leaders})
    price_map = await _fetch_summaries(unique_syms)

    # Build leader list
    leaders = []
    for l in raw_leaders:
        lsym = l["leader_ticker"]
        pdata = price_map.get(lsym, {})
        leaders.append({
            "rank":        l["rank"],
            "symbol":      lsym,
            "name":        _ticker_sector_name(lsym),
            "sector":      l["leader_sector"],
            "lag_days":    l["lag_days"],
            "attn_weight": round(l["attn_weight"], 4),
            "leader_ic":   round(l["leader_ic"], 4),
            "price":       pdata.get("price",  "—"),
            "change":      pdata.get("change", "—"),
            "volume":      pdata.get("volume", "—"),
            "positive":    pdata.get("positive", True),
        })

    # Sector breakdown — count + total attention weight per sector
    sector_agg: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_weight": 0.0})
    for l in raw_leaders:
        sec = l["leader_sector"]
        sector_agg[sec]["count"]        += 1
        sector_agg[sec]["total_weight"] = round(
            sector_agg[sec]["total_weight"] + l["attn_weight"], 4
        )

    # Sort sectors by total attention weight descending
    sector_breakdown = dict(
        sorted(sector_agg.items(), key=lambda x: x[1]["total_weight"], reverse=True)
    )

    return {
        "target":           sym,
        "target_sector":    entry["target_sector"],
        "target_ic":        entry["target_ic"],
        "as_of_date":       entry["as_of_date"],
        "leaders":          leaders,
        "sector_breakdown": sector_breakdown,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

# Minimal name lookup from the sector map (symbol → "TICKER (Sector)")
# A richer name DB would replace this, but it works for display.
_KNOWN_NAMES: dict[str, str] = {
    # Major names not in ticker_sector as a name field
    # (ticker_sector only maps symbol → sector, not full name)
    # We keep a small override table; otherwise we display the symbol itself.
}

def _ticker_sector_name(sym: str) -> str:
    """Return a human-readable name for a ticker. Falls back to symbol."""
    return _KNOWN_NAMES.get(sym, sym)