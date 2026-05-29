# backend/app/services/dashboard_cache.py
"""
In-memory dashboard cache for the stock watchlist and market indices.

Loads FEATURED_TICKERS + index summaries from BigQuery once at backend startup,
then uses two complementary mechanisms to stay fresh:

  1. Scheduled sleep: after each load, sleeps until the next 8:30 PM ET
     (30 min after the nightly market_data update at ~8 PM ET).
     Works well when the server stays warm with regular traffic.

  2. On-demand staleness check (request-triggered refresh): the routers call
     trigger_refresh_if_stale() on every request. If the wall-clock age of the
     cache exceeds CACHE_MAX_AGE_SECONDS (23 hours), a background refresh is
     fired immediately — without blocking the response. This is the safety net
     for Cloud Run scale-to-zero: even if the container was dormant for a week
     and the sleep timer never fired, the very first user request that day
     triggers a fresh BigQuery load in the background. The user gets the stale
     data instantly (still correct last-close prices), and the cache is live
     again within ~5 seconds.

BigQuery cost: 2 queries/day × ~400 MB scanned = ~24 GB/month, well within
the 1 TB/month free tier regardless of table clustering.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.models.stock import IndexSummary, StockSummary
from app.services.bigquery_services import (
    FEATURED_TICKERS,
    get_index_summaries,
    get_stock_summaries,
)

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# ── Shared in-memory caches ───────────────────────────────────────────────────

_STOCK_SUMMARY_CACHE: list[StockSummary] = []
_INDEX_SUMMARY_CACHE: list[IndexSummary] = []

# Wall-clock timestamp of the last successful BigQuery load.
# Uses time.time() (real wall-clock), NOT time.monotonic(), so it advances
# correctly even when the Cloud Run container is frozen between requests.
_last_refresh_wall: float = 0.0

# A background refresh is already in flight — don't fire a second one.
_refresh_in_progress: bool = False

# Trigger a fresh load if cached data is older than this.
CACHE_MAX_AGE_SECONDS = 23 * 60 * 60  # 23 hours

# ── Refresh schedule (scheduled sleep path) ───────────────────────────────────

_REFRESH_HOUR   = 20  # 8 PM ET
_REFRESH_MINUTE = 30  # :30


def _seconds_until_next_refresh() -> float:
    """Return wall-clock seconds until the next 8:30 PM ET."""
    now    = datetime.now(ET)
    target = now.replace(
        hour=_REFRESH_HOUR, minute=_REFRESH_MINUTE, second=0, microsecond=0
    )
    if now >= target:
        target += timedelta(days=1)
    return (target - now).total_seconds()


# ── Core load function ────────────────────────────────────────────────────────

async def _do_refresh() -> None:
    """Run the BigQuery load and update the in-memory caches."""
    global _last_refresh_wall, _refresh_in_progress
    loop = asyncio.get_event_loop()
    try:
        stocks  = await loop.run_in_executor(
            None, lambda: get_stock_summaries(FEATURED_TICKERS)
        )
        indices = await loop.run_in_executor(None, get_index_summaries)

        _STOCK_SUMMARY_CACHE.clear()
        _STOCK_SUMMARY_CACHE.extend(stocks)

        _INDEX_SUMMARY_CACHE.clear()
        _INDEX_SUMMARY_CACHE.extend(indices)

        _last_refresh_wall = time.time()

        logger.info(
            "Dashboard cache refreshed: %d stocks, %d indices",
            len(_STOCK_SUMMARY_CACHE),
            len(_INDEX_SUMMARY_CACHE),
        )
    except Exception as e:
        logger.error("Dashboard cache refresh failed — keeping stale data: %s", e)
    finally:
        _refresh_in_progress = False


# ── Background task (startup + scheduled path) ────────────────────────────────

async def refresh_dashboard_cache() -> None:
    """
    Background task started at FastAPI startup.

    Loads data immediately, then sleeps until 8:30 PM ET each day.
    Works perfectly when the container stays warm. If the container was
    dormant and the sleep timer drifted, trigger_refresh_if_stale() on
    the first incoming request acts as the safety net.

    Start with asyncio.create_task() inside the FastAPI startup event.
    """
    while True:
        await _do_refresh()

        sleep_secs = _seconds_until_next_refresh()
        logger.info(
            "Dashboard cache: next scheduled refresh in %.1f hours (8:30 PM ET)",
            sleep_secs / 3600,
        )
        await asyncio.sleep(sleep_secs)


# ── On-demand staleness check (request-triggered path) ────────────────────────

def trigger_refresh_if_stale() -> None:
    """
    Call this at the top of any router that serves cached dashboard data.

    If the cache is older than CACHE_MAX_AGE_SECONDS (23 h) AND no refresh
    is already running, fires a non-blocking background refresh. The current
    request is served immediately from the existing (stale) cache — users
    never wait. The refresh completes in the background (~5 s) and all
    subsequent requests will hit the updated data.

    This is the safety net for Cloud Run scale-to-zero: even if the scheduled
    sleep timer never advanced because the container was dormant, the first
    user request on any given day ensures the cache self-heals.
    """
    global _refresh_in_progress

    if _refresh_in_progress:
        return

    age = time.time() - _last_refresh_wall
    if age > CACHE_MAX_AGE_SECONDS:
        logger.info(
            "Dashboard cache is %.1f hours old — triggering background refresh",
            age / 3600,
        )
        _refresh_in_progress = True
        asyncio.create_task(_do_refresh())


# ── Public getters ────────────────────────────────────────────────────────────

def get_cached_stocks() -> list[StockSummary]:
    return list(_STOCK_SUMMARY_CACHE)


def get_cached_indices() -> list[IndexSummary]:
    return list(_INDEX_SUMMARY_CACHE)


def dashboard_cache_ready() -> bool:
    """True once at least one successful load has completed."""
    return bool(_STOCK_SUMMARY_CACHE) and bool(_INDEX_SUMMARY_CACHE)
