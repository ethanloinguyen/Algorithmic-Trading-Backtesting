# backend/app/routers/indices.py
from fastapi import APIRouter
from app.models.stock import IndexListResponse
from app.services.bigquery_services import get_index_summaries

router = APIRouter(prefix="/api/indices", tags=["indices"])


# ── GET /api/indices ──────────────────────────────────────────────────────────
@router.get("", response_model=IndexListResponse)
def list_indices():
    """
    Returns the latest summary data for the three main market indices:
    SPX (S&P 500), IXIC (NASDAQ), DJI (Dow Jones).
    Used to populate the index cards at the top of the dashboard.
    """
    data = get_index_summaries()
    return IndexListResponse(data=data)