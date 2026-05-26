# backend/app/services/ticker_cache.py
# -------------------------------------
# In-memory ticker cache for autocomplete search.
#
# Loads the full ticker list from BigQuery once at backend startup, then
# refreshes it every 24 hours in the background. All search requests are
# served from this in-memory list — zero BigQuery queries per keystroke.
#
# Why in-memory instead of Firestore/BQ per keystroke:
#   - ticker_metadata has ~2,500 rows (~50 KB total)
#   - The symbol universe only changes during the nightly update_data run
#   - In-memory lookup is <1ms vs ~200–500ms for a BQ round trip
#   - During refresh the stale cache keeps serving — no downtime

import asyncio
import logging

from app.core.bigquery import get_bq_client
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Single shared list — all requests from all users hit this same object.
# Mutations (clear + extend during refresh) are fast enough to be safe
# without a lock at this scale.
_TICKER_CACHE: list[dict] = []

REFRESH_INTERVAL_SECONDS = 24 * 60 * 60  # 24 hours


def _load_tickers() -> list[dict]:
    """
    Query ticker_metadata once and return a list of
    {"symbol": str, "name": str} dicts ordered by ticker.

    BigQuery cost: ~50 KB per call (ticker_metadata is tiny).
    At one call per day this is effectively free.
    """
    client   = get_bq_client()
    settings = get_settings()

    sql = f"""
        SELECT
            ticker,
            COALESCE(company_name, ticker) AS name
        FROM {settings.fq_ticker_metadata}
        ORDER BY ticker
    """
    rows = client.query(sql).result()
    return [{"symbol": r.ticker, "name": r.name} for r in rows]


async def refresh_ticker_cache() -> None:
    """
    Background task: load tickers immediately on startup, then reload
    every 24 hours. Runs for the lifetime of the backend process.

    On refresh failure the stale cache is kept — searches keep working.
    Start this with asyncio.create_task() inside the FastAPI startup event.
    """
    while True:
        try:
            fresh = await asyncio.get_event_loop().run_in_executor(
                None, _load_tickers
            )
            _TICKER_CACHE.clear()
            _TICKER_CACHE.extend(fresh)
            logger.info(f"Ticker cache loaded: {len(_TICKER_CACHE):,} tickers")
        except Exception as e:
            logger.error(f"Ticker cache refresh failed — keeping stale cache: {e}")

        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)


def search_tickers(query: str, limit: int = 20) -> list[dict]:
    """
    Search the in-memory ticker cache. No BigQuery call involved.

    Matches:
      1. Ticker symbol starts with query  (e.g. "AP" → AAPL, APD, APH ...)
      2. Company name contains query      (e.g. "apple" → AAPL, APLD ...)

    Prefix matches are returned first, then name-contains matches,
    both groups sorted alphabetically. Combined results capped at limit.

    Returns list of {"symbol": str, "name": str}.
    """
    q = query.strip().upper()
    if not q:
        return []

    limit = max(1, min(int(limit), 50))

    prefix_hits = []
    name_hits   = []

    for entry in _TICKER_CACHE:
        symbol = entry["symbol"]
        name   = entry["name"].upper()

        if symbol.startswith(q):
            prefix_hits.append(entry)
        elif q in name:
            name_hits.append(entry)

    return (prefix_hits + name_hits)[:limit]
