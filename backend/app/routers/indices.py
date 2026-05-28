# backend/app/routers/indices.py
from fastapi import APIRouter
from app.models.stock import IndexListResponse
from app.services.bigquery_services import get_index_summaries
from app.services.cache_service import get_cached_index_summaries, set_cached_index_summaries
from app.services.dashboard_cache import get_cached_indices, dashboard_cache_ready, trigger_refresh_if_stale

router = APIRouter(prefix="/api/indices", tags=["indices"])


# ── GET /api/indices ──────────────────────────────────────────────────────────
@router.get("", response_model=IndexListResponse)
def list_indices():
    """
    Returns the latest summary for SPX, IXIC, DJI.

    Priority:
      1. In-memory dashboard cache (loaded at startup, refreshed nightly at
         8:30 PM ET) — zero BigQuery / Firestore cost on the hot path.
      2. Firestore cache — covers the brief startup window before the in-memory
         cache has completed its first BigQuery load (~3-5 s).
      3. BigQuery direct query — last resort, only on very first cold start.
    """
    # 1. In-memory cache (always warm after startup)
    if dashboard_cache_ready():
        trigger_refresh_if_stale()   # no-op if fresh; fires background refresh if >23 h old
        return IndexListResponse(data=get_cached_indices())

    # 2. Firestore cache (startup fallback)
    cached = get_cached_index_summaries()
    if cached is not None:
        return IndexListResponse(data=cached)

    # 3. BigQuery fallback — write to Firestore for subsequent startup requests
    data = get_index_summaries()
    set_cached_index_summaries(data)
    return IndexListResponse(data=data)