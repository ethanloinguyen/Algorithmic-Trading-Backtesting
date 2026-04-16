// Frontend/src/app/lib/api.ts
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:4000";

export interface StockSummary {
  symbol: string; name: string; price: string;
  change: string; volume: string; positive: boolean;
}
export interface OHLCVCandle {
  date: string; open: string; high: string;
  low: string; close: string; volume: string;
}
export interface IndexSummary {
  symbol: string; name: string; value: string;
  change: string; pct: string; price: string; positive: boolean;
}
export type TimeRange = "1D" | "1W" | "1M" | "3M" | "1Y" | "5Y";

// ── Portfolio types ───────────────────────────────────────────────────────────

export interface OverlapResult {
  ticker_leader: string; ticker_follower: string; best_lag: number;
  signal_strength: number; mean_dcor: number; oos_sharpe_net: number;
  predicted_sharpe: number; sector_leader: string; sector_follower: string;
  frequency: number; half_life: number; sharpness: number; interpretation: string;
}

/** Group A — stock with a detected lead-lag connection to your holdings */
export interface Recommendation {
  ticker: string; sector: string; centrality: number; composite_score: number;
  n_portfolio_relationships: number; best_relationship_strength: number;
  mean_relationship_strength: number; related_holdings: string[];
  direction: "leads_your_holdings" | "follows_your_holdings" | "both";
  signal_score: number;
  centrality_score: number;
  sector_diversity_score: number;
  coverage_score: number;
  durability_score: number;   // normalized half-life 0-100
  reasoning: string;
}

/** Group B — stock with ZERO detected relationship to any of your holdings */
export interface IndependentRecommendation {
  ticker:           string;
  sector:           string;
  centrality:       number;
  composite_score:  number;
  sector_gap_score: number;
  centrality_score: number;
  reasoning:        string;
}

export interface PortfolioAnalysisResponse {
  tickers_analyzed:            string[];
  unknown_tickers:             string[];
  overlaps:                    OverlapResult[];
  signal_recommendations:      Recommendation[];
  independent_recommendations: IndependentRecommendation[];
  /** Sector for each known holding — always populated regardless of overlaps */
  holdings_sectors:            Record<string, string>;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    cache: "no-store", headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) throw new Error(`API ${path} → HTTP ${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

// ── Endpoints ─────────────────────────────────────────────────────────────────

export async function fetchAllStocks(): Promise<StockSummary[]> {
  const data = await apiFetch<{ data: StockSummary[] }>("/api/stocks");
  return data.data;
}

export async function fetchStockSummaries(symbols: string[]): Promise<StockSummary[]> {
  if (!symbols.length) return [];
  const q = symbols.map(s => s.toUpperCase()).join(",");
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

export async function analyzePortfolio(tickers: string[]): Promise<PortfolioAnalysisResponse> {
  const res = await fetch(`${BASE}/api/portfolio/analyze`, {
    method: "POST", cache: "no-store",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tickers: tickers.map(t => t.toUpperCase()) }),
  });
  if (!res.ok) throw new Error(`/api/portfolio/analyze → HTTP ${res.status} ${res.statusText}`);
  return res.json() as Promise<PortfolioAnalysisResponse>;
}