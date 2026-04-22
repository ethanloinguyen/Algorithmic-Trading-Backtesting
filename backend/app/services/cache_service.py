# backend/app/services/cache_service.py
"""
Server-side Firestore caching layer.

All cache documents live in the top-level `cache` collection of the
`capstone-firestore` database so every user benefits from the same
warmed data.

Cache keys:
  cache/stock_summaries            — featured watchlist  (TTL 5 min)
  cache/index_summaries            — SPX / IXIC / DJI    (TTL 5 min)
  cache/ohlcv_{SYMBOL}_{RANGE}     — OHLCV candles       (TTL 1 min for 1D,
                                                           10 min otherwise)
Document shape:
  { "data": [...], "updated_at": <Firestore SERVER_TIMESTAMP> }
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from google.cloud import firestore

from app.core.firestore import get_fs_client
from app.models.stock import OHLCVCandle, StockSummary, IndexSummary, TimeRange

# ── TTLs ──────────────────────────────────────────────────────────────────────

TTL_SUMMARIES      = timedelta(minutes=5)
TTL_OHLCV_INTRADAY = timedelta(minutes=1)
TTL_OHLCV_OTHER    = timedelta(minutes=10)


def _ohlcv_ttl(range_: TimeRange) -> timedelta:
    return TTL_OHLCV_INTRADAY if range_ == TimeRange.ONE_DAY else TTL_OHLCV_OTHER


def _is_stale(doc_data: dict[str, Any], ttl: timedelta) -> bool:
    updated_at = doc_data.get("updated_at")
    if updated_at is None:
        return True
    # google-cloud-firestore returns DatetimeWithNanoseconds (always tz-aware)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - updated_at > ttl


# ── Stock summaries ───────────────────────────────────────────────────────────

def get_cached_summaries(doc_id: str = "stock_summaries") -> list[StockSummary] | None:
    """Return cached summaries if fresh, else None (triggers BigQuery fallback)."""
    try:
        fs   = get_fs_client()
        snap = fs.collection("cache").document(doc_id).get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        if _is_stale(data, TTL_SUMMARIES):
            return None
        return [StockSummary(**s) for s in data["data"]]
    except Exception as e:
        # Log but never crash the request — BigQuery will serve as fallback
        print(f"[cache] get_cached_summaries({doc_id}) failed: {e}")
        return None


def set_cached_summaries(
    summaries: list[StockSummary],
    doc_id: str = "stock_summaries",
) -> None:
    try:
        fs = get_fs_client()
        fs.collection("cache").document(doc_id).set({
            "data":       [s.model_dump() for s in summaries],
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
    except Exception as e:
        print(f"[cache] set_cached_summaries({doc_id}) failed: {e}")


# ── Index summaries ───────────────────────────────────────────────────────────

def get_cached_index_summaries() -> list[IndexSummary] | None:
    try:
        fs   = get_fs_client()
        snap = fs.collection("cache").document("index_summaries").get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        if _is_stale(data, TTL_SUMMARIES):
            return None
        return [IndexSummary(**s) for s in data["data"]]
    except Exception as e:
        print(f"[cache] get_cached_index_summaries failed: {e}")
        return None


def set_cached_index_summaries(summaries: list[IndexSummary]) -> None:
    try:
        fs = get_fs_client()
        fs.collection("cache").document("index_summaries").set({
            "data":       [s.model_dump() for s in summaries],
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
    except Exception as e:
        print(f"[cache] set_cached_index_summaries failed: {e}")


# ── OHLCV candles ─────────────────────────────────────────────────────────────

def _ohlcv_doc_id(symbol: str, range_: TimeRange) -> str:
    return f"ohlcv_{symbol.upper()}_{range_.value}"


def get_cached_ohlcv(symbol: str, range_: TimeRange) -> list[OHLCVCandle] | None:
    try:
        fs   = get_fs_client()
        snap = fs.collection("cache").document(_ohlcv_doc_id(symbol, range_)).get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        if _is_stale(data, _ohlcv_ttl(range_)):
            return None
        return [OHLCVCandle(**c) for c in data["data"]]
    except Exception as e:
        print(f"[cache] get_cached_ohlcv({symbol},{range_}) failed: {e}")
        return None


def set_cached_ohlcv(
    symbol:  str,
    range_:  TimeRange,
    candles: list[OHLCVCandle],
) -> None:
    try:
        fs = get_fs_client()
        fs.collection("cache").document(_ohlcv_doc_id(symbol, range_)).set({
            "data":       [c.model_dump() for c in candles],
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
    except Exception as e:
        print(f"[cache] set_cached_ohlcv({symbol},{range_}) failed: {e}")