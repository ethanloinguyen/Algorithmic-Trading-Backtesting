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
export type AnalysisMode = "broad_market" | "in_sector";

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

/** Quality Picks — stock scored on five portfolio-aware quality dimensions */
export interface QualityRecommendation {
  ticker:                          string;
  sector:                          string;
  centrality:                      number;
  composite_score:                 number;
  momentum_score:                  number;
  fundamental_quality_score:       number;
  sector_diversity_score:          number;
  volatility_compatibility_score:  number;
  centrality_score:                number;
  reasoning:                       string;
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

/** Extended stock info for the Analysis page fundamentals panel */
export interface StockDetail {
  symbol:     string;
  name:       string;
  sector:     string | null;
  industry:   string | null;
  market_cap: number | null;
  pe_ratio:   number | null;
  high_52w:   number | null;
  low_52w:    number | null;
}

/** Lead-lag pair details from final_network / sector_final_network */
export interface PairDetail {
  ticker_i:        string;
  ticker_j:        string;
  best_lag:        number;
  mean_dcor:       number;
  signal_strength: number;
  frequency:       number;
  half_life:       number;
  oos_sharpe_net:  number;
  sector_i:        string;
  sector_j:        string;
  found:           boolean;
}

export interface PortfolioAnalysisResponse {
  tickers_analyzed:            string[];
  unknown_tickers:             string[];
  overlaps:                    OverlapResult[];
  signal_recommendations:      Recommendation[];
  independent_recommendations: IndependentRecommendation[];
  quality_picks:               QualityRecommendation[];
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

export async function fetchStockDetail(symbol: string): Promise<StockDetail> {
  return apiFetch<StockDetail>(`/api/stocks/${encodeURIComponent(symbol.toUpperCase())}/detail`);
}

export async function fetchPairData(
  tickerI: string,
  tickerJ: string,
  analysis_mode: AnalysisMode = "broad_market",
): Promise<PairDetail> {
  const ti = encodeURIComponent(tickerI.toUpperCase());
  const tj = encodeURIComponent(tickerJ.toUpperCase());
  return apiFetch<PairDetail>(`/api/pairs/${ti}/${tj}?analysis_mode=${analysis_mode}`);
}

// ── Network graph types + endpoint ────────────────────────────────────────────

export interface NetworkNode {
  id:         string;
  sector:     string;
  centrality: number;
  out_degree: number;
}

export interface NetworkEdge {
  source:          string;
  target:          string;
  signal_strength: number;
  best_lag:        number;
  mean_dcor:       number;
}

export interface NetworkResponse {
  nodes:         NetworkNode[];
  edges:         NetworkEdge[];
  analysis_mode: string;
  min_signal:    number;
}

export async function fetchNetwork(
  analysis_mode: AnalysisMode = "broad_market",
  min_signal:    number       = 55,
  limit:         number       = 50,
): Promise<NetworkResponse> {
  return apiFetch<NetworkResponse>(
    `/api/network?analysis_mode=${analysis_mode}&min_signal=${min_signal}&limit=${limit}`
  );
}

export async function analyzePortfolio(
  tickers: string[],
  analysis_mode: AnalysisMode = "broad_market",
): Promise<PortfolioAnalysisResponse> {
  const res = await fetch(`${BASE}/api/portfolio/analyze`, {
    method: "POST", cache: "no-store",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tickers: tickers.map(t => t.toUpperCase()), analysis_mode }),
  });
  if (!res.ok) throw new Error(`/api/portfolio/analyze → HTTP ${res.status} ${res.statusText}`);
  return res.json() as Promise<PortfolioAnalysisResponse>;
}

// ── Risk pipeline types + endpoint ────────────────────────────────────────────

export interface StockRiskMetrics {
  var_95: number;
  cvar_95: number;
  tail_risk_ratio_95: number;
  var_99: number;
  cvar_99: number;
  tail_risk_ratio_99: number;
  expected_max_drawdown: number;
  worst_case_max_drawdown_p95: number;
  prob_loss: number;
  prob_return_above_10pct: number;
  skewness: number;
  avg_recovery_days: number | null;
  sortino_ratio_historical_252d: number | null;
}

export interface PortfolioRiskMetrics extends StockRiskMetrics {
  diversification_benefit_95: number;
  diversification_benefit_99: number;
  risk_contribution_per_stock_95: Record<string, number>;
  risk_contribution_per_stock_99: Record<string, number>;
}

export interface ClusteringPick {
  sector: string;
  stock: string;
  cluster: number;
  is_medoid: boolean;
  avg_dcor_to_portfolio: number;
  mean_intra_dist: number;
  n_sector_candidates: number;
  cluster_size: number;
}

export interface RiskPipelineResult {
  user_portfolio: string[];
  recommendations: ClusteringPick[];
  risk: {
    tickers: string[];
    missing: string[];
    weights: Record<string, number>;
    horizon_days: number;
    n_simulations: number;
    per_stock: Record<string, StockRiskMetrics>;
    portfolio: PortfolioRiskMetrics;
  };
}

export async function runRiskPipeline(tickers: string[]): Promise<RiskPipelineResult> {
  const res = await fetch(`${BASE}/api/portfolio/risk-pipeline`, {
    method: "POST", cache: "no-store",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tickers: tickers.map(t => t.toUpperCase()) }),
  });
  if (!res.ok) throw new Error(`/api/portfolio/risk-pipeline → HTTP ${res.status} ${res.statusText}`);
  return res.json() as Promise<RiskPipelineResult>;
}