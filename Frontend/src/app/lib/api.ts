// Frontend/src/app/lib/api.ts
// Central API client — all fetch calls to the FastAPI/BigQuery backend.
//
// Required .env.local:
//   NEXT_PUBLIC_API_URL=http://localhost:4000

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:4000";

// ── Types — mirror backend Pydantic models exactly ────────────────────────────

export interface StockSummary {
  symbol:   string;   // "AAPL"
  name:     string;   // "Apple Inc."
  price:    string;   // "$189.84"
  change:   string;   // "+1.25%"
  volume:   string;   // "52.3M"
  positive: boolean;
}

export interface OHLCVCandle {
  date:   string;   // "Mar 01" | "9:30 AM" | "Mar '24"
  open:   string;   // "188.42"
  high:   string;
  low:    string;
  close:  string;
  volume: string;   // "52.3M"
}

export interface IndexSummary {
  symbol:   string;   // "SPX"
  name:     string;   // "S&P 500 Index"
  value:    string;   // "5248.49"
  change:   string;   // "+0.63%"
  pct:      string;   // "+0.63%"
  price:    string;   // "$5248.49"
  positive: boolean;
}

// Must exactly match backend TimeRange enum in app/models/stock.py
export type TimeRange = "1D" | "1W" | "1M" | "3M" | "1Y" | "5Y";

// ── Internal helper ───────────────────────────────────────────────────────────

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    cache:   "no-store",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    throw new Error(`API ${path} → HTTP ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ── Endpoints ─────────────────────────────────────────────────────────────────

/** GET /api/stocks — summaries for FEATURED_TICKERS list */
export async function fetchAllStocks(): Promise<StockSummary[]> {
  const data = await apiFetch<{ data: StockSummary[] }>("/api/stocks");
  return data.data;
}

/** GET /api/stocks/summaries?symbols=AAPL,MSFT */
export async function fetchStockSummaries(symbols: string[]): Promise<StockSummary[]> {
  if (!symbols.length) return [];
  const q    = symbols.map(s => s.toUpperCase()).join(",");
  const data = await apiFetch<{ data: StockSummary[] }>(`/api/stocks/summaries?symbols=${q}`);
  return data.data;
}

/** GET /api/stocks/{symbol}/ohlcv?range=1M */
export async function fetchOHLCV(symbol: string, range: TimeRange): Promise<OHLCVCandle[]> {
  const data = await apiFetch<{ symbol: string; range: string; candles: OHLCVCandle[] }>(
    `/api/stocks/${encodeURIComponent(symbol.toUpperCase())}/ohlcv?range=${range}`
  );
  return data.candles;
}

/** GET /api/indices — SPX, IXIC, DJI */
export async function fetchIndices(): Promise<IndexSummary[]> {
  const data = await apiFetch<{ data: IndexSummary[] }>("/api/indices");
  return data.data;
}