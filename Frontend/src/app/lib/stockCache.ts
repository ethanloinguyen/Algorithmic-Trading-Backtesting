// Frontend/src/app/lib/stockCache.ts
//
// Firestore caching layer for stock data.
//
// Strategy:
//   • Watchlist summaries  → cached in Firestore at  cache/stock_summaries
//     TTL: 5 minutes. On cache hit, data is returned immediately while a
//     background refresh runs if the cache is >2 minutes old.
//
//   • OHLCV candles        → cached per symbol+range at  cache/ohlcv_{SYMBOL}_{RANGE}
//     TTL: 1 minute for 1D (intraday), 10 minutes for all other ranges.
//
//   • Click frequency      → tracked per user at  users/{uid}/clickedStocks
//     The top-5 most-clicked stocks get their OHLCV pre-warmed in Firestore
//     so the modal opens instantly.
//
// All cache docs live in the top-level `cache` collection (not per-user) so
// all users benefit from the same warmed data.

import {
  doc,
  getDoc,
  setDoc,
  Timestamp,
  updateDoc,
  increment,
} from "firebase/firestore";
import { db } from "@/src/app/lib/firebase";
import type { StockSummary, OHLCVCandle, TimeRange } from "@/src/app/lib/api";

// ── TTL constants (ms) ────────────────────────────────────────────────────────

const TTL_SUMMARIES      = 5  * 60 * 1000;   // 5 min
const TTL_OHLCV_INTRADAY = 1  * 60 * 1000;   // 1 min  (1D range)
const TTL_OHLCV_OTHER    = 10 * 60 * 1000;   // 10 min (all other ranges)

function ohlcvTTL(range: TimeRange): number {
  return range === "1D" ? TTL_OHLCV_INTRADAY : TTL_OHLCV_OTHER;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function ageMs(updatedAt: Timestamp): number {
  return Date.now() - updatedAt.toMillis();
}

function isStale(updatedAt: Timestamp, ttl: number): boolean {
  return ageMs(updatedAt) > ttl;
}

// ── Stock summaries cache ─────────────────────────────────────────────────────

interface SummariesCacheDoc {
  data:      StockSummary[];
  updatedAt: Timestamp;
}

/**
 * Read cached summaries from Firestore.
 * Returns { data, stale } where stale=true means a refresh should run.
 * Returns null if no cache entry exists yet.
 */
export async function getCachedSummaries(): Promise<{
  data:  StockSummary[];
  stale: boolean;
} | null> {
  try {
    const snap = await getDoc(doc(db, "cache", "stock_summaries"));
    if (!snap.exists()) return null;
    const cached = snap.data() as SummariesCacheDoc;
    return {
      data:  cached.data,
      stale: isStale(cached.updatedAt, TTL_SUMMARIES),
    };
  } catch {
    return null;
  }
}

/** Write fresh summaries into the Firestore cache. */
export async function setCachedSummaries(data: StockSummary[]): Promise<void> {
  try {
    await setDoc(doc(db, "cache", "stock_summaries"), {
      data,
      updatedAt: Timestamp.now(),
    });
  } catch {
    // Non-critical — cache write failure is silent
  }
}

// ── OHLCV cache ───────────────────────────────────────────────────────────────

interface OHLCVCacheDoc {
  candles:   OHLCVCandle[];
  updatedAt: Timestamp;
}

function ohlcvDocId(symbol: string, range: TimeRange): string {
  return `ohlcv_${symbol.toUpperCase()}_${range}`;
}

/**
 * Read cached OHLCV candles from Firestore.
 * Returns { candles, stale } or null if no cache entry exists.
 */
export async function getCachedOHLCV(
  symbol: string,
  range:  TimeRange,
): Promise<{ candles: OHLCVCandle[]; stale: boolean } | null> {
  try {
    const snap = await getDoc(doc(db, "cache", ohlcvDocId(symbol, range)));
    if (!snap.exists()) return null;
    const cached = snap.data() as OHLCVCacheDoc;
    return {
      candles: cached.candles,
      stale:   isStale(cached.updatedAt, ohlcvTTL(range)),
    };
  } catch {
    return null;
  }
}

/** Write fresh OHLCV candles into the Firestore cache. */
export async function setCachedOHLCV(
  symbol:  string,
  range:   TimeRange,
  candles: OHLCVCandle[],
): Promise<void> {
  try {
    await setDoc(doc(db, "cache", ohlcvDocId(symbol, range)), {
      candles,
      updatedAt: Timestamp.now(),
    });
  } catch {
    // Non-critical
  }
}

// ── Click frequency tracking ──────────────────────────────────────────────────

/**
 * Increment the click count for a stock symbol for the given user.
 * Stored at users/{uid}/clickedStocks as a map { AAPL: 5, MSFT: 3, ... }
 * This is used to pre-warm OHLCV cache for the user's most-viewed stocks.
 */
export async function recordStockClick(uid: string, symbol: string): Promise<void> {
  try {
    const ref = doc(db, "users", uid);
    await updateDoc(ref, {
      [`clickedStocks.${symbol.toUpperCase()}`]: increment(1),
    });
  } catch {
    // Non-critical — don't block the UI
  }
}

/**
 * Return the top N most-clicked symbols for a user.
 * Used to decide which stocks to pre-warm in the OHLCV cache.
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