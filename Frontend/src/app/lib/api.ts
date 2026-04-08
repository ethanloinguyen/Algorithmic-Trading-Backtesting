// Frontend/src/app/lib/api.ts
// Central API client — all fetch calls to the FastAPI/BigQuery backend.
//
// Required .env.local:
//   NEXT_PUBLIC_API_URL=http://localhost:4000

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:4000";

// ── Existing types ────────────────────────────────────────────────────────────

export interface StockSummary {
  symbol:   string;
  name:     string;
  price:    string;
  change:   string;
  volume:   string;
  positive: boolean;
}

export interface OHLCVCandle {
  date:   string;
  open:   string;
  high:   string;
  low:    string;
  close:  string;
  volume: string;
}

export interface IndexSummary {
  symbol:   string;
  name:     string;
  value:    string;
  change:   string;
  pct:      string;
  price:    string;
  positive: boolean;
}

export type TimeRange = "1D" | "1W" | "1M" | "3M" | "1Y" | "5Y";

// ── Portfolio diversification types ──────────────────────────────────────────

export interface OverlapResult {
  ticker_leader:    string;
  ticker_follower:  string;
  best_lag:         number;
  signal_strength:  number;
  mean_dcor:        number;
  oos_sharpe_net:   number;
  predicted_sharpe: number;
  sector_leader:    string;
  sector_follower:  string;
  frequency:        number;
  half_life:        number;
  sharpness:        number;
  interpretation:   string;
}

export interface Recommendation {
  ticker:                     string;
  sector:                     string;
  centrality:                 number;
  composite_score:            number;
  n_portfolio_relationships:  number;
  best_relationship_strength: number;
  mean_relationship_strength: number;
  related_holdings:           string[];
  direction:                  "leads_your_holdings" | "follows_your_holdings" | "both";
  signal_score:               number;
  centrality_score:           number;
  sector_diversity_score:     number;
  coverage_score:             number;
  reasoning:                  string;
}

export interface PortfolioAnalysisResponse {
  tickers_analyzed: string[];
  unknown_tickers:  string[];
  overlaps:         OverlapResult[];
  recommendations:  Recommendation[];
}

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

// ── Existing endpoints (unchanged) ────────────────────────────────────────────

export async function fetchAllStocks(): Promise<StockSummary[]> {
  const data = await apiFetch<{ data: StockSummary[] }>("/api/stocks");
  return data.data;
}

export async function fetchStockSummaries(symbols: string[]): Promise<StockSummary[]> {
  if (!symbols.length) return [];
  const q    = symbols.map(s => s.toUpperCase()).join(",");
  const data = await apiFetch<{ data: StockSummary[] }>(`/api/stocks/summaries?symbols=${q}`);
  return data.data;
}

export async function fetchOHLCV(symbol: string, range: TimeRange): Promise<OHLCVCandle[]> {
  const data = await apiFetch<{ symbol: string; range: string; candles: OHLCVCandle[] }>(
    `/api/stocks/${encodeURIComponent(symbol.toUpperCase())}/ohlcv?range=${range}`
  );
  return data.candles;
}

export async function fetchIndices(): Promise<IndexSummary[]> {
  const data = await apiFetch<{ data: IndexSummary[] }>("/api/indices");
  return data.data;
}

// ── Portfolio endpoint ────────────────────────────────────────────────────────

/** POST /api/portfolio/analyze */
export async function analyzePortfolio(tickers: string[]): Promise<PortfolioAnalysisResponse> {
  const res = await fetch(`${BASE}/api/portfolio/analyze`, {
    method:  "POST",
    cache:   "no-store",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ tickers: tickers.map(t => t.toUpperCase()) }),
  });
  if (!res.ok) {
    throw new Error(`/api/portfolio/analyze → HTTP ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<PortfolioAnalysisResponse>;
}