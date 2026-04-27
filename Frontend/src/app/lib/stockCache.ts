// Frontend/src/app/lib/stockCache.ts
//
// Frontend Firestore cache reader.
//
// The backend (FastAPI) is responsible for ALL cache writes to Firestore.
// Every time the backend serves data from BigQuery it writes it to:
//
//   cache/stock_summaries        — featured watchlist  (TTL 5 min)
//   cache/index_summaries        — SPX / IXIC / DJI    (TTL 5 min)
//   cache/ohlcv_{SYMBOL}_{RANGE} — OHLCV candles       (TTL 1 min / 10 min)
//
// The frontend reads these documents directly from Firestore for instant
// display, then falls back to the backend API when the cache is cold or stale.
//
// Click tracking (users/{uid}/clickedStocks) is written by the frontend only
// since it is per-user data the backend does not need to know about.

import {
  doc,
  getDoc,
  updateDoc,
  Timestamp,
  increment,
} from "firebase/firestore";
import { db } from "@/src/app/lib/firebase";
import type { StockSummary, OHLCVCandle, IndexSummary, TimeRange } from "@/src/app/lib/api";

// ── TTL constants (ms) — must match backend cache_service.py ─────────────────

const TTL_SUMMARIES      = 5  * 60 * 1000;   // 5 min
const TTL_OHLCV_INTRADAY = 1  * 60 * 1000;   // 1 min  (1D range)
const TTL_OHLCV_OTHER    = 10 * 60 * 1000;   // 10 min (all other ranges)

function ohlcvTTL(range: TimeRange): number {
  return range === "1D" ? TTL_OHLCV_INTRADAY : TTL_OHLCV_OTHER;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function isStale(updatedAt: Timestamp, ttl: number): boolean {
  return Date.now() - updatedAt.toMillis() > ttl;
}

// ── Firestore document shapes (mirror backend cache_service.py) ───────────────

interface SummariesDoc {
  data:       StockSummary[];
  updated_at: Timestamp;
}

interface IndicesDoc {
  data:       IndexSummary[];   // ← correctly typed, not Record<string, unknown>
  updated_at: Timestamp;
}

interface OHLCVDoc {
  data:       OHLCVCandle[];
  updated_at: Timestamp;
}

// ── Stock summaries ───────────────────────────────────────────────────────────

/**
 * Read cached watchlist summaries written by the backend.
 * Returns { data, stale } if a cache doc exists, null if cold.
 */
export async function getCachedSummaries(): Promise<{
  data:  StockSummary[];
  stale: boolean;
} | null> {
  try {
    const snap = await getDoc(doc(db, "cache", "stock_summaries"));
    if (!snap.exists()) return null;
    const d = snap.data() as SummariesDoc;
    return {
      data:  d.data,
      stale: isStale(d.updated_at, TTL_SUMMARIES),
    };
  } catch {
    return null;
  }
}

// ── Index summaries ───────────────────────────────────────────────────────────

/**
 * Read cached index summaries (SPX / IXIC / DJI) written by the backend.
 * Returns { data, stale } if a cache doc exists, null if cold.
 */
export async function getCachedIndices(): Promise<{
  data:  IndexSummary[];   // ← correct return type, no cast needed at call site
  stale: boolean;
} | null> {
  try {
    const snap = await getDoc(doc(db, "cache", "index_summaries"));
    if (!snap.exists()) return null;
    const d = snap.data() as IndicesDoc;
    return {
      data:  d.data,
      stale: isStale(d.updated_at, TTL_SUMMARIES),
    };
  } catch {
    return null;
  }
}

// ── OHLCV candles ─────────────────────────────────────────────────────────────

function ohlcvDocId(symbol: string, range: TimeRange): string {
  return `ohlcv_${symbol.toUpperCase()}_${range}`;
}

/**
 * Read cached OHLCV candles written by the backend.
 * Returns { candles, stale } if a cache doc exists, null if cold.
 */
export async function getCachedOHLCV(
  symbol: string,
  range:  TimeRange,
): Promise<{ candles: OHLCVCandle[]; stale: boolean } | null> {
  try {
    const snap = await getDoc(doc(db, "cache", ohlcvDocId(symbol, range)));
    if (!snap.exists()) return null;
    const d = snap.data() as OHLCVDoc;
    return {
      candles: d.data,
      stale:   isStale(d.updated_at, ohlcvTTL(range)),
    };
  } catch {
    return null;
  }
}

// ── Click frequency tracking (frontend-only, per-user) ────────────────────────

/**
 * Increment click count for a symbol in users/{uid}/clickedStocks.
 * Used to decide which stocks to pre-warm after login.
 */
export async function recordStockClick(uid: string, symbol: string): Promise<void> {
  try {
    await updateDoc(doc(db, "users", uid), {
      [`clickedStocks.${symbol.toUpperCase()}`]: increment(1),
    });
  } catch {
    // Non-critical — never block the UI
  }
}

/**
 * Return the top N most-clicked symbols for a user.
 * Used by AuthContext to pre-warm OHLCV cache on login.
 */
export async function getTopClickedSymbols(uid: string, n = 5): Promise<string[]> {
  try {
    const snap = await getDoc(doc(db, "users", uid));
    if (!snap.exists()) return [];
    const clicks = (snap.data().clickedStocks ?? {}) as Record<string, number>;
    return Object.entries(clicks)
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .map(([sym]) => sym);
  } catch {
    return [];
  }
}