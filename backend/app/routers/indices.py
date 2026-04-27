# backend/app/routers/indices.py
from fastapi import APIRouter
from app.models.stock import IndexListResponse
from app.services.bigquery_services import get_index_summaries
from app.services.cache_service import get_cached_index_summaries, set_cached_index_summaries

router = APIRouter(prefix="/api/indices", tags=["indices"])


# ── GET /api/indices ──────────────────────────────────────────────────────────
@router.get("", response_model=IndexListResponse)
def list_indices():
    """
    Returns the latest summary for SPX, IXIC, DJI.
    Checks Firestore cache first (TTL 5 min) before querying BigQuery.
    """
    # 1. Try Firestore cache
    cached = get_cached_index_summaries()
    if cached is not None:
        return IndexListResponse(data=cached)

    # 2. Cache miss — query BigQuery
    data = get_index_summaries()

    # 3. Write to Firestore cache
    set_cached_index_summaries(data)

    return IndexListResponse(data=data)