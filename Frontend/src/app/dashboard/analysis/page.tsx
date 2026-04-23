// frontend/src/app/dashboard/analysis/page.tsx
"use client";
import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import Sidebar from "@/components/ui/Sidebar";
import { ChevronDown, Loader2, AlertTriangle, Search } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import {
  fetchOHLCV,
  fetchStockDetail,
  fetchPairData,
  fetchAllStocks,
  type OHLCVCandle,
  type StockDetail,
  type PairDetail,
  type StockSummary,
  type AnalysisMode,
} from "@/src/app/lib/api";

// ─── Design tokens ────────────────────────────────────────────────────────────
const BG       = "hsl(213, 27%, 7%)";
const CARD     = { background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" };
const CARD_H   = "hsl(215, 25%, 14%)";
const TEXT_PRI = "hsl(210, 40%, 92%)";
const TEXT_SEC = "hsl(215, 15%, 55%)";
const TEXT_MUT = "hsl(215, 15%, 40%)";
const BLUE     = "hsl(217, 91%, 60%)";
const BLUE_DIM = "hsla(217, 91%, 60%, 0.15)";
const GREEN    = "hsl(142, 71%, 45%)";
const AMBER    = "hsl(38, 92%, 50%)";
const RED      = "hsl(0, 84%, 60%)";
const BORDER   = "hsl(215, 20%, 18%)";
const BORDER_D = "hsl(215, 20%, 16%)";

const labelStyle = { fill: TEXT_SEC, fontSize: 11 } as const;

// Three distinct colors for the year-over-year overlay
const YEAR_COLORS = [
  "hsl(240, 80%, 65%)",
  "hsl(10,  80%, 60%)",
  "hsl(170, 70%, 50%)",
];

const LAG_RANGES = ["1M", "3M", "6M", "1Y"] as const;
type LagRange = typeof LAG_RANGES[number];
const SLICE: Record<LagRange, number> = { "1M": 21, "3M": 63, "6M": 126, "1Y": 252 };

// ─── Network mock data (unchanged — deferred to Task 2 integration) ──────────
type NetworkNode = {
  id: string; sector: string; centrality: number; outDegree: number;
  x?: number; y?: number;
};
type NetworkEdge = { source: string; target: string; weight: number };

const MOCK_NODES: NetworkNode[] = [
  { id: "MSFT", sector: "Tech",    centrality: 98, outDegree: 24 },
  { id: "AAPL", sector: "Tech",    centrality: 85, outDegree: 18 },
  { id: "NVDA", sector: "Tech",    centrality: 92, outDegree: 20 },
  { id: "INTC", sector: "Tech",    centrality: 45, outDegree: 5  },
  { id: "CSCO", sector: "Tech",    centrality: 50, outDegree: 7  },
  { id: "JPM",  sector: "Finance", centrality: 88, outDegree: 16 },
  { id: "GS",   sector: "Finance", centrality: 75, outDegree: 12 },
  { id: "V",    sector: "Finance", centrality: 82, outDegree: 15 },
  { id: "MA",   sector: "Finance", centrality: 80, outDegree: 14 },
];
const MOCK_EDGES: NetworkEdge[] = [
  { source: "MSFT", target: "AAPL", weight: 0.85 },
  { source: "MSFT", target: "NVDA", weight: 0.70 },
  { source: "MSFT", target: "INTC", weight: 0.95 },
  { source: "MSFT", target: "CSCO", weight: 0.76 },
  { source: "NVDA", target: "AAPL", weight: 0.65 },
  { source: "JPM",  target: "GS",   weight: 0.90 },
  { source: "V",    target: "JPM",  weight: 0.55 },
  { source: "V",    target: "MA",   weight: 0.88 },
  { source: "MSFT", target: "V",    weight: 0.45 },
  { source: "AAPL", target: "GS",   weight: 0.20 },
];
const NODE_POSITIONS: Record<string, { x: number; y: number }> = {
  MSFT: { x: 0.30, y: 0.25 }, NVDA: { x: 0.70, y: 0.25 },
  AAPL: { x: 0.50, y: 0.50 }, INTC: { x: 0.15, y: 0.55 },
  CSCO: { x: 0.20, y: 0.78 }, JPM:  { x: 0.50, y: 0.82 },
  GS:   { x: 0.75, y: 0.68 }, V:    { x: 0.82, y: 0.45 },
  MA:   { x: 0.85, y: 0.72 },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatMarketCap(cap: number | null): string {
  if (cap === null) return "—";
  if (cap >= 1_000_000_000_000) return `$${(cap / 1_000_000_000_000).toFixed(2)}T`;
  if (cap >= 1_000_000_000)     return `$${(cap / 1_000_000_000).toFixed(1)}B`;
  if (cap >= 1_000_000)         return `$${(cap / 1_000_000).toFixed(1)}M`;
  return `$${cap.toLocaleString()}`;
}

/** Normalize candles so the first close = 100. */
function normalizeCandles(candles: OHLCVCandle[]): number[] {
  if (!candles.length) return [];
  const base = parseFloat(candles[0].close);
  if (!base) return candles.map(() => 100);
  return candles.map(c => parseFloat(((parseFloat(c.close) / base) * 100).toFixed(2)));
}

/** Compute daily % returns from candles. */
function dailyReturns(candles: OHLCVCandle[]): number[] {
  return candles.map((c, i) => {
    if (i === 0) return 0;
    const prev = parseFloat(candles[i - 1].close);
    const curr = parseFloat(c.close);
    return prev ? parseFloat(((curr - prev) / prev * 100).toFixed(3)) : 0;
  });
}

/**
 * Parse the 4-digit year from a 5Y-formatted date label ("Mar '24" → 2024).
 * Returns null if the label doesn't match the expected format.
 */
function parseYear5Y(dateLabel: string): number | null {
  const m = dateLabel.match(/'(\d{2})$/);
  return m ? 2000 + parseInt(m[1]) : null;
}

/** Approximate month label from trading-day index within a year. */
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
function monthFromDayIdx(i: number) {
  const idx = Math.floor((i / 252) * 12);
  return MONTHS[Math.min(idx, 11)];
}

// ─── Stock Dropdown ───────────────────────────────────────────────────────────

function StockDropdown({
  value, onChange, stocks, loading,
}: {
  value: string; onChange: (s: string) => void;
  stocks: StockSummary[]; loading: boolean;
}) {
  const [open,  setOpen]  = useState(false);
  const [query, setQuery] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const h = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  const filtered = stocks.filter(s =>
    s.symbol.includes(query.toUpperCase()) || s.name.toLowerCase().includes(query.toLowerCase())
  ).slice(0, 30);
  const current = stocks.find(s => s.symbol === value);

  return (
    <div className="relative" ref={ref}>
      <button onClick={() => setOpen(o => !o)}
        className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium"
        style={{ background: CARD_H, border: `1px solid ${BORDER}`, color: TEXT_PRI }}>
        {loading && <Loader2 className="w-3.5 h-3.5 animate-spin" style={{ color: TEXT_SEC }} />}
        <span>{value}{current ? ` — ${current.name.substring(0, 12)}` : ""}</span>
        <ChevronDown className="w-3.5 h-3.5" style={{ color: TEXT_SEC }} />
      </button>
      {open && (
        <div className="absolute top-full left-0 mt-1 w-64 rounded-lg overflow-hidden z-20"
          style={{ background: "hsl(215,25%,13%)", border: `1px solid ${BORDER}`, boxShadow: "0 8px 24px rgba(0,0,0,0.5)" }}>
          <div className="p-2 border-b" style={{ borderColor: BORDER_D }}>
            <div className="flex items-center gap-2 px-2 py-1.5 rounded-md" style={{ background: "hsl(215,25%,9%)" }}>
              <Search className="w-3.5 h-3.5 flex-shrink-0" style={{ color: TEXT_MUT }} />
              <input autoFocus value={query} onChange={e => setQuery(e.target.value)}
                placeholder="Search ticker or name…"
                className="flex-1 bg-transparent outline-none text-xs" style={{ color: TEXT_PRI }} />
            </div>
          </div>
          <div className="max-h-56 overflow-y-auto">
            {filtered.map(s => (
              <button key={s.symbol}
                onClick={() => { onChange(s.symbol); setOpen(false); setQuery(""); }}
                className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm"
                style={{ background: s.symbol === value ? BLUE_DIM : "transparent", color: s.symbol === value ? BLUE : TEXT_PRI }}
                onMouseEnter={e => { if (s.symbol !== value) e.currentTarget.style.background = "hsl(215,25%,17%)"; }}
                onMouseLeave={e => { if (s.symbol !== value) e.currentTarget.style.background = "transparent"; }}>
                <span className="font-bold w-14 flex-shrink-0">{s.symbol}</span>
                <span className="text-xs truncate" style={{ color: TEXT_SEC }}>{s.name}</span>
              </button>
            ))}
            {!filtered.length && (
              <p className="px-4 py-3 text-xs text-center" style={{ color: TEXT_MUT }}>No results</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Year-over-Year Price Chart ───────────────────────────────────────────────
// Fetches 5Y OHLCV and overlays the last 3 complete calendar years,
// each normalized to 100 at the start of that year — shows seasonal patterns.

function PriceChart({ symbol }: { symbol: string }) {
  const [candles, setCandles] = useState<OHLCVCandle[]>([]);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  useEffect(() => {
    if (!symbol) return;
    setLoading(true); setError(null);
    fetchOHLCV(symbol, "5Y")
      .then(setCandles)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [symbol]);

  // Split candles into calendar years using the "Mar '24" date format
  const { years, chartData } = useMemo(() => {
    if (!candles.length) return { years: [], chartData: [] };

    const byYear = new Map<number, OHLCVCandle[]>();
    for (const c of candles) {
      const yr = parseYear5Y(c.date);
      if (yr === null) continue;
      if (!byYear.has(yr)) byYear.set(yr, []);
      byYear.get(yr)!.push(c);
    }

    // Take the last 3 complete-ish years
    const sortedYears = [...byYear.keys()].sort().slice(-3);
    const maxLen = Math.max(...sortedYears.map(y => byYear.get(y)!.length));

    const data = Array.from({ length: maxLen }, (_, i) => {
      const point: Record<string, number | string> = { day: i, month: monthFromDayIdx(i) };
      for (const yr of sortedYears) {
        const yc = byYear.get(yr)!;
        if (i < yc.length) {
          const base = parseFloat(yc[0].close);
          point[String(yr)] = parseFloat(((parseFloat(yc[i].close) / base) * 100).toFixed(2));
        }
      }
      return point;
    });

    return { years: sortedYears, chartData: data };
  }, [candles]);

  if (loading) return (
    <div className="flex-1 rounded-xl flex items-center justify-center" style={CARD}>
      <Loader2 className="w-6 h-6 animate-spin" style={{ color: BLUE }} />
    </div>
  );
  if (error) return (
    <div className="flex-1 rounded-xl flex items-center justify-center gap-2 px-6" style={CARD}>
      <AlertTriangle className="w-4 h-4" style={{ color: RED }} />
      <p className="text-sm" style={{ color: RED }}>{error}</p>
    </div>
  );

  return (
    <div className="flex-1 rounded-xl p-5" style={CARD}>
      <p className="text-xs font-semibold mb-4 uppercase tracking-widest" style={{ color: TEXT_MUT }}>
        {symbol} — Year-over-Year Price Performance (normalized)
      </p>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={BORDER} />
            <XAxis dataKey="month" axisLine={false} tickLine={false} tick={labelStyle}
              interval={20} />
            <YAxis domain={["auto", "auto"]} axisLine={false} tickLine={false} tick={labelStyle}
              width={48} tickFormatter={v => `${v.toFixed(0)}`} />
            <Tooltip
              contentStyle={{ background: "hsl(215,25%,13%)", border: `1px solid ${BORDER}`, borderRadius: 8 }}
              labelStyle={{ color: TEXT_SEC, fontSize: 11 }}
              itemStyle={{ fontSize: 12 }}
              formatter={(v: number, name: string) => [`${v.toFixed(1)}`, name]}
            />
            <ReferenceLine y={100} stroke={BORDER} strokeDasharray="4 4" />
            {years.map((yr, i) => (
              <Line key={yr} type="monotone" dataKey={String(yr)}
                stroke={YEAR_COLORS[i % YEAR_COLORS.length]} strokeWidth={2}
                dot={false} isAnimationActive={false} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {/* Legend */}
      <div className="flex gap-5 mt-3">
        {years.map((yr, i) => (
          <div key={yr} className="flex items-center gap-2">
            <div className="w-5 h-0.5 rounded" style={{ background: YEAR_COLORS[i % YEAR_COLORS.length] }} />
            <span className="text-xs" style={{ color: TEXT_SEC }}>{yr}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Fundamentals Panel ───────────────────────────────────────────────────────

function FundamentalsPanel({ symbol }: { symbol: string }) {
  const [detail, setDetail]   = useState<StockDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!symbol) return;
    setLoading(true);
    fetchStockDetail(symbol)
      .then(setDetail)
      .catch(() => setDetail(null))
      .finally(() => setLoading(false));
  }, [symbol]);

  const rows = detail ? [
    { label: "Sector",     value: detail.sector     ?? "—" },
    { label: "Industry",   value: detail.industry   ?? "—" },
    { label: "Market Cap", value: formatMarketCap(detail.market_cap) },
    { label: "P/E Ratio",  value: detail.pe_ratio != null ? detail.pe_ratio.toFixed(1) : "—" },
    { label: "52W High",   value: detail.high_52w != null ? `$${detail.high_52w.toFixed(2)}` : "—" },
    { label: "52W Low",    value: detail.low_52w  != null ? `$${detail.low_52w.toFixed(2)}`  : "—" },
  ] : [];

  return (
    <div className="w-64 rounded-xl p-5 flex-shrink-0" style={CARD}>
      <p className="text-lg font-bold mb-0.5" style={{ color: TEXT_PRI }}>{symbol}</p>
      <p className="text-sm mb-5" style={{ color: TEXT_SEC }}>{detail?.name ?? ""}</p>
      {loading ? (
        <div className="flex justify-center py-8">
          <Loader2 className="w-5 h-5 animate-spin" style={{ color: BLUE }} />
        </div>
      ) : rows.map(({ label, value }) => (
        <div key={label} className="flex items-center justify-between py-3"
          style={{ borderBottom: `1px solid ${BORDER_D}` }}>
          <span className="text-sm" style={{ color: TEXT_SEC }}>{label}</span>
          <span className="text-sm font-semibold" style={{ color: TEXT_PRI }}>{value}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Ticker Input ─────────────────────────────────────────────────────────────

function TickerInput({
  value, onChange, placeholder, label, color,
}: {
  value: string; onChange: (v: string) => void;
  placeholder: string; label: string; color: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color }}>{label}</span>
      <input
        value={value}
        onChange={e => onChange(e.target.value.toUpperCase())}
        placeholder={placeholder}
        maxLength={10}
        className="w-28 px-3 py-2 rounded-lg text-sm font-bold outline-none"
        style={{ background: "hsl(215,25%,9%)", border: `1px solid ${color}40`, color: TEXT_PRI }}
      />
    </div>
  );
}

// ─── Lag Alignment Lab ────────────────────────────────────────────────────────

function LagAlignmentLab({ analysisMode }: { analysisMode: AnalysisMode }) {
  const [stockA, setStockA] = useState("MSFT");
  const [stockB, setStockB] = useState("AAPL");
  const [lagRange, setLagRange] = useState<LagRange>("3M");
  const [showResiduals, setShowResiduals] = useState(false);

  const [pairData,      setPairData]      = useState<PairDetail | null>(null);
  const [leaderOHLCV,   setLeaderOHLCV]   = useState<OHLCVCandle[]>([]);
  const [followerOHLCV, setFollowerOHLCV] = useState<OHLCVCandle[]>([]);
  // Resolved tickers after pair lookup (may differ from inputs if ordering was flipped)
  const [leaderTicker,   setLeaderTicker]   = useState("MSFT");
  const [followerTicker, setFollowerTicker] = useState("AAPL");
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    const sa = stockA.trim().toUpperCase();
    const sb = stockB.trim().toUpperCase();
    if (!sa || !sb || sa === sb) return;
    setLoading(true); setError(null);
    try {
      // Pair lookup determines which is leader (ticker_i) and which is follower (ticker_j)
      const pair = await fetchPairData(sa, sb, analysisMode);
      const leader   = pair.found ? pair.ticker_i : sa;
      const follower = pair.found ? pair.ticker_j : sb;

      // Always fetch 1Y so time range buttons can slice client-side instantly
      const [lOHLCV, fOHLCV] = await Promise.all([
        fetchOHLCV(leader,   "1Y"),
        fetchOHLCV(follower, "1Y"),
      ]);
      setPairData(pair);
      setLeaderOHLCV(lOHLCV);
      setFollowerOHLCV(fOHLCV);
      setLeaderTicker(leader);
      setFollowerTicker(follower);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load pair data.");
    } finally {
      setLoading(false);
    }
  }, [stockA, stockB, analysisMode]);

  // Auto-fetch on mount
  useEffect(() => { fetchData(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-fetch when analysis mode switches (only if we have prior results)
  useEffect(() => {
    if (pairData) fetchData();
  }, [analysisMode]); // eslint-disable-line react-hooks/exhaustive-deps

  // Client-side time range slice — no extra API call
  const count  = SLICE[lagRange];
  const visL   = leaderOHLCV.slice(-count);
  const visF   = followerOHLCV.slice(-count);

  const chartData = useMemo(() => {
    const n = Math.min(visL.length, visF.length);
    if (!n) return [];
    if (showResiduals) {
      const lr = dailyReturns(visL.slice(0, n));
      const fr = dailyReturns(visF.slice(0, n));
      return visL.slice(0, n).map((c, i) => ({ date: c.date, leader: lr[i], follower: fr[i] }));
    }
    const ln = normalizeCandles(visL.slice(0, n));
    const fn = normalizeCandles(visF.slice(0, n));
    return visL.slice(0, n).map((c, i) => ({ date: c.date, leader: ln[i], follower: fn[i] }));
  }, [visL, visF, showResiduals]);

  const LEADER_COLOR   = "hsl(217, 91%, 60%)";
  const FOLLOWER_COLOR = "hsl(142, 71%, 45%)";

  return (
    <div className="rounded-xl p-5" style={{ background: "hsl(215,25%,11%)", border: `1px solid ${BORDER}` }}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-5 pb-4"
        style={{ borderBottom: `1px solid ${BORDER}` }}>
        <div>
          <h2 className="text-base font-bold" style={{ color: TEXT_PRI }}>Lead-Lag Hypothesis Lab</h2>
          <p className="text-xs mt-0.5" style={{ color: TEXT_MUT }}>
            Enter two stocks — analysis determines which leads and which follows
          </p>
        </div>

        <div className="flex items-end gap-4 p-3 rounded-lg"
          style={{ background: CARD_H, border: `1px solid ${BORDER}` }}>
          {/* Stock inputs — no leader/follower label on the inputs themselves */}
          <TickerInput value={stockA} onChange={setStockA} label="Stock A" placeholder="e.g. MSFT" color={LEADER_COLOR} />
          <TickerInput value={stockB} onChange={setStockB} label="Stock B" placeholder="e.g. AAPL" color={FOLLOWER_COLOR} />

          <div className="flex flex-col gap-1.5">
            {/* Time range */}
            <div className="flex gap-1">
              {LAG_RANGES.map(r => (
                <button key={r} onClick={() => setLagRange(r)}
                  className="px-2 py-1 rounded text-[10px] font-bold"
                  style={{
                    background: r === lagRange ? BLUE : "hsl(215,25%,17%)",
                    color:      r === lagRange ? "white" : TEXT_SEC,
                    border: `1px solid ${r === lagRange ? "transparent" : BORDER}`,
                  }}>
                  {r}
                </button>
              ))}
            </div>
            {/* Action buttons */}
            <div className="flex gap-1.5">
              <button onClick={fetchData}
                disabled={loading || !stockA.trim() || !stockB.trim()}
                className="flex-1 h-8 px-3 text-xs font-bold rounded-lg disabled:opacity-40"
                style={{ background: BLUE, color: "white" }}>
                {loading
                  ? <Loader2 className="w-3.5 h-3.5 animate-spin inline" />
                  : "Analyze"}
              </button>
              <button onClick={() => setShowResiduals(r => !r)}
                className="h-8 px-2 text-[10px] font-bold uppercase rounded-lg"
                style={{ background: "hsl(215,25%,17%)", color: TEXT_SEC, border: `1px solid ${BORDER}` }}>
                {showResiduals ? "Raw" : "Residuals"}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 mb-4 px-4 py-3 rounded-lg"
          style={{ background: "hsla(0,84%,60%,0.1)", border: "1px solid hsla(0,84%,60%,0.3)" }}>
          <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: RED }} />
          <p className="text-xs" style={{ color: RED }}>{error}</p>
        </div>
      )}

      {/* Pair stats strip — shows data-determined leader → follower direction */}
      {pairData && (
        <div className="flex items-center gap-6 mb-4 px-4 py-3 rounded-lg flex-wrap"
          style={{ background: CARD_H, border: `1px solid ${BORDER_D}` }}>
          {pairData.found ? (
            <>
              <div>
                <p className="text-[10px] font-bold uppercase tracking-widest mb-0.5" style={{ color: TEXT_MUT }}>Detected Direction</p>
                <p className="text-sm font-bold">
                  <span style={{ color: LEADER_COLOR }}>{pairData.ticker_i}</span>
                  <span style={{ color: TEXT_MUT }}> leads </span>
                  <span style={{ color: FOLLOWER_COLOR }}>{pairData.ticker_j}</span>
                </p>
              </div>
              {[
                { label: "Optimal Lag",     value: `${pairData.best_lag}d`,                          color: BLUE },
                { label: "dCor",            value: pairData.mean_dcor.toFixed(3),                    color: TEXT_PRI },
                { label: "Signal",          value: `${pairData.signal_strength.toFixed(0)}/100`,     color: pairData.signal_strength >= 65 ? GREEN : AMBER },
                { label: "Frequency",       value: `${(pairData.frequency * 100).toFixed(0)}%`,      color: TEXT_PRI },
                { label: "Half-life",       value: `${Math.round(pairData.half_life)}d`,              color: TEXT_PRI },
              ].map(({ label, value, color }) => (
                <div key={label}>
                  <p className="text-[10px] font-bold uppercase tracking-widest mb-0.5" style={{ color: TEXT_MUT }}>{label}</p>
                  <p className="text-sm font-bold" style={{ color }}>{value}</p>
                </div>
              ))}
            </>
          ) : (
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" style={{ color: AMBER }} />
              <p className="text-xs" style={{ color: AMBER }}>
                No lead-lag relationship detected between{" "}
                <strong>{pairData.ticker_i}</strong> and <strong>{pairData.ticker_j}</strong>{" "}
                in the {analysisMode === "in_sector" ? "in-sector" : "broad market"} analysis.
                Showing price comparison only.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Chart */}
      {loading ? (
        <div className="h-64 flex items-center justify-center">
          <Loader2 className="w-6 h-6 animate-spin" style={{ color: BLUE }} />
        </div>
      ) : chartData.length > 0 ? (
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={BORDER} />
              <XAxis dataKey="date" axisLine={false} tickLine={false} tick={labelStyle}
                interval="preserveStartEnd" minTickGap={40} />
              <YAxis domain={["auto", "auto"]} axisLine={false} tickLine={false} tick={labelStyle}
                width={showResiduals ? 52 : 52}
                tickFormatter={v => showResiduals ? `${v > 0 ? "+" : ""}${v.toFixed(1)}%` : `${v.toFixed(0)}`}
              />
              <Tooltip
                contentStyle={{ background: "hsl(215,25%,13%)", border: `1px solid ${BORDER}`, borderRadius: 8 }}
                labelStyle={{ color: TEXT_SEC, fontSize: 11 }}
                formatter={(v: number, name: string) => [
                  showResiduals ? `${v > 0 ? "+" : ""}${v.toFixed(2)}%` : v.toFixed(2),
                  name === "leader" ? leaderTicker : followerTicker,
                ]}
              />
              {showResiduals && <ReferenceLine y={0} stroke={BORDER} strokeDasharray="4 4" />}
              <Line type="monotone" dataKey="leader"   stroke={LEADER_COLOR}   strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="follower" stroke={FOLLOWER_COLOR} strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="h-64 flex items-center justify-center rounded-lg" style={{ background: "hsl(215,25%,9%)" }}>
          <p className="text-sm" style={{ color: TEXT_MUT }}>Enter two tickers and click Analyze</p>
        </div>
      )}

      {/* Legend — keyed by role, not ticker value, to avoid duplicate-key error */}
      {chartData.length > 0 && (
        <div className="flex gap-5 mt-3">
          {([
            { role: "leader",   ticker: leaderTicker,   color: LEADER_COLOR   },
            { role: "follower", ticker: followerTicker, color: FOLLOWER_COLOR },
          ] as const).map(({ role, ticker, color }) => (
            <div key={role} className="flex items-center gap-2">
              <div className="w-5 h-0.5 rounded" style={{ background: color }} />
              <span className="text-xs font-semibold" style={{ color: TEXT_SEC }}>
                {ticker}
                {pairData?.found && (
                  <span className="ml-1 font-normal" style={{ color: TEXT_MUT }}>
                    ({role === "leader" ? "leads" : "follows"})
                  </span>
                )}
              </span>
            </div>
          ))}
          <span className="text-xs ml-2" style={{ color: TEXT_MUT }}>
            {showResiduals ? "Daily % return" : "Normalized (base 100)"}
          </span>
        </div>
      )}
    </div>
  );
}

// ─── Lead-Lag Network (mock data — real data integration deferred) ─────────────

function LeadLagNetwork() {
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const animRef    = useRef<number>(0);
  const [minSignal, setMinSignal]       = useState(0.4);
  const [selectedSector, setSelectedSector] = useState("All");
  const [activeNode, setActiveNode]     = useState<NetworkNode>(MOCK_NODES[0]);
  const [hoverNodeId, setHoverNodeId]   = useState<string | null>(null);

  const filteredNodes = useMemo(
    () => MOCK_NODES.filter(n => selectedSector === "All" || n.sector === selectedSector),
    [selectedSector]
  );
  const filteredNodeIds = useMemo(() => new Set(filteredNodes.map(n => n.id)), [filteredNodes]);
  const filteredEdges = useMemo(
    () => MOCK_EDGES.filter(e => e.weight >= minSignal && filteredNodeIds.has(e.source) && filteredNodeIds.has(e.target)),
    [minSignal, filteredNodeIds]
  );
  const activeFollowers = useMemo(
    () => MOCK_EDGES.filter(e => e.source === activeNode.id && e.weight >= minSignal)
           .sort((a, b) => b.weight - a.weight),
    [activeNode, minSignal]
  );

  const particleRef = useRef<{ edge: NetworkEdge; t: number }[]>([]);
  useEffect(() => {
    particleRef.current = filteredEdges.map(e => ({ edge: e, t: Math.random() }));
  }, [filteredEdges]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "hsl(215,30%,8%)";
    ctx.fillRect(0, 0, W, H);

    const px = (n: NetworkNode) => NODE_POSITIONS[n.id].x * W;
    const py = (n: NetworkNode) => NODE_POSITIONS[n.id].y * H;

    filteredEdges.forEach(e => {
      const sNode = filteredNodes.find(n => n.id === e.source);
      const tNode = filteredNodes.find(n => n.id === e.target);
      if (!sNode || !tNode) return;
      const x1 = px(sNode), y1 = py(sNode), x2 = px(tNode), y2 = py(tNode);
      let edgeColor = `rgba(148,163,184,${e.weight * 0.5})`;
      if (hoverNodeId) {
        if (e.source === hoverNodeId)      edgeColor = "rgba(34,197,94,0.85)";
        else if (e.target === hoverNodeId) edgeColor = "rgba(239,68,68,0.85)";
        else                                edgeColor = "rgba(100,116,139,0.08)";
      }
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
      ctx.strokeStyle = edgeColor; ctx.lineWidth = e.weight * 2.5; ctx.stroke();
      const angle = Math.atan2(y2 - y1, x2 - x1), arr = 7;
      const ex = x2 - Math.cos(angle) * 14, ey = y2 - Math.sin(angle) * 14;
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - arr * Math.cos(angle - 0.4), ey - arr * Math.sin(angle - 0.4));
      ctx.lineTo(ex - arr * Math.cos(angle + 0.4), ey - arr * Math.sin(angle + 0.4));
      ctx.closePath(); ctx.fillStyle = edgeColor; ctx.fill();
    });

    particleRef.current.forEach(p => {
      p.t = (p.t + p.edge.weight * 0.004) % 1;
      const sNode = filteredNodes.find(n => n.id === p.edge.source);
      const tNode = filteredNodes.find(n => n.id === p.edge.target);
      if (!sNode || !tNode) return;
      const x = px(sNode) + (px(tNode) - px(sNode)) * p.t;
      const y = py(sNode) + (py(tNode) - py(sNode)) * p.t;
      ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.7)"; ctx.fill();
    });

    filteredNodes.forEach(n => {
      const x = px(n), y = py(n);
      const r = Math.max(8, n.centrality / 12);
      const isHovered = n.id === hoverNodeId;
      const isActive  = n.id === activeNode.id;
      const dimmed    = hoverNodeId && !isHovered &&
        !filteredEdges.some(e => (e.source === hoverNodeId && e.target === n.id) || (e.target === hoverNodeId && e.source === n.id));
      const base = n.sector === "Tech" ? "99,102,241" : "16,185,129";
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = dimmed ? "rgba(100,116,139,0.2)" : `rgba(${base},${isActive ? 1 : 0.85})`;
      ctx.fill();
      if (isActive || isHovered) {
        ctx.beginPath(); ctx.arc(x, y, r + 4, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${base},0.4)`; ctx.lineWidth = 2; ctx.stroke();
      }
      ctx.font = "bold 11px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillStyle = dimmed ? "rgba(255,255,255,0.15)" : "white";
      ctx.fillText(n.id, x, y);
    });
    animRef.current = requestAnimationFrame(draw);
  }, [filteredNodes, filteredEdges, hoverNodeId, activeNode]);

  useEffect(() => {
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [draw]);

  const getNodeAt = useCallback((mx: number, my: number, canvas: HTMLCanvasElement) => {
    const rect = canvas.getBoundingClientRect();
    const cx = (mx - rect.left) * (canvas.width / rect.width);
    const cy = (my - rect.top) * (canvas.height / rect.height);
    return filteredNodes.find(n => {
      const nx = NODE_POSITIONS[n.id].x * canvas.width;
      const ny = NODE_POSITIONS[n.id].y * canvas.height;
      return Math.hypot(cx - nx, cy - ny) <= Math.max(8, n.centrality / 12) + 4;
    }) ?? null;
  }, [filteredNodes]);

  return (
    <div className="rounded-xl p-5" style={{ background: "hsl(215,25%,11%)", border: `1px solid ${BORDER}` }}>
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-5 pb-4"
        style={{ borderBottom: `1px solid ${BORDER}` }}>
        <div>
          <h2 className="text-base font-bold" style={{ color: TEXT_PRI }}>Lead-Lag Network Analytics Lab</h2>
          <p className="text-xs mt-0.5" style={{ color: TEXT_MUT }}>
            Mock data — real network integration coming in Task 2
          </p>
        </div>
        <div className="flex gap-4 text-xs font-bold" style={{ color: TEXT_SEC }}>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-500" /> Tech
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Finance
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Sidebar */}
        <div className="space-y-4">
          <div className="rounded-lg p-4" style={{ background: CARD_H, border: `1px solid ${BORDER_D}` }}>
            <p className="text-[10px] font-bold uppercase tracking-widest mb-3" style={{ color: TEXT_MUT }}>
              Network Filters
            </p>
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-semibold mb-1.5" style={{ color: TEXT_PRI }}>Sector</label>
                <select value={selectedSector} onChange={e => setSelectedSector(e.target.value)}
                  className="w-full text-xs rounded-md px-3 py-2"
                  style={{ background: "hsl(215,25%,11%)", border: `1px solid ${BORDER_D}`, color: TEXT_PRI }}>
                  {["All","Tech","Finance"].map(s => <option key={s}>{s}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold mb-1" style={{ color: TEXT_PRI }}>
                  Min Signal: <span style={{ color: BLUE }}>{minSignal.toFixed(2)}</span>
                </label>
                <input type="range" min={0} max={0.95} step={0.05} value={minSignal}
                  onChange={e => setMinSignal(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500" />
              </div>
            </div>
          </div>
          <div className="rounded-lg p-4" style={{ background: CARD_H, border: `1px solid ${BORDER_D}` }}>
            <p className="text-[10px] font-bold uppercase tracking-widest mb-3" style={{ color: TEXT_MUT }}>Selected Node</p>
            <p className="text-base font-bold mb-0.5" style={{ color: TEXT_PRI }}>{activeNode.id}</p>
            <p className="text-xs mb-3" style={{ color: TEXT_SEC }}>{activeNode.sector}</p>
            {[
              { label: "Centrality", value: activeNode.centrality },
              { label: "Out-Degree", value: activeNode.outDegree },
            ].map(({ label, value }) => (
              <div key={label} className="flex justify-between py-2" style={{ borderBottom: `1px solid ${BORDER}` }}>
                <span className="text-xs" style={{ color: TEXT_SEC }}>{label}</span>
                <span className="text-xs font-semibold" style={{ color: TEXT_PRI }}>{value}</span>
              </div>
            ))}
            <p className="text-[10px] font-bold uppercase tracking-widest mt-3 mb-2" style={{ color: TEXT_MUT }}>Leads</p>
            {activeFollowers.length === 0 ? (
              <p className="text-xs italic" style={{ color: TEXT_MUT }}>None above threshold</p>
            ) : (
              <ul className="space-y-1">
                {activeFollowers.map(e => (
                  <li key={e.target} className="flex justify-between text-xs">
                    <span style={{ color: TEXT_PRI }}>{e.target}</span>
                    <span className="font-medium px-1.5 py-0.5 rounded"
                      style={{ background: "hsla(142,71%,45%,0.15)", color: GREEN }}>
                      {e.weight.toFixed(2)} sig
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Canvas */}
        <div className="lg:col-span-3 rounded-lg overflow-hidden" style={{ background: "hsl(215,30%,8%)", minHeight: 380 }}>
          <canvas ref={canvasRef} width={800} height={420} className="w-full h-full"
            style={{ cursor: hoverNodeId ? "pointer" : "default" }}
            onMouseMove={e => { const n = getNodeAt(e.clientX, e.clientY, canvasRef.current!); setHoverNodeId(n ? n.id : null); }}
            onMouseLeave={() => setHoverNodeId(null)}
            onClick={e => { const n = getNodeAt(e.clientX, e.clientY, canvasRef.current!); if (n) setActiveNode(n); }}
          />
        </div>
      </div>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function AnalysisPage() {
  const [selectedStock, setSelectedStock] = useState("AAPL");
  const [analysisMode,  setAnalysisMode]  = useState<AnalysisMode>("broad_market");
  const [stocks,        setStocks]        = useState<StockSummary[]>([]);
  const [stocksLoading, setStocksLoading] = useState(true);

  useEffect(() => {
    fetchAllStocks()
      .then(setStocks)
      .catch(() => setStocks([]))
      .finally(() => setStocksLoading(false));
  }, []);

  return (
    <div className="min-h-screen" style={{ background: BG }}>
      <Sidebar />
      <main className="pt-14">
        <div className="max-w-7xl mx-auto px-6 py-6 space-y-6">

          {/* ── Top bar ── */}
          <div className="flex items-center gap-4 flex-wrap">
            <StockDropdown value={selectedStock} onChange={setSelectedStock}
              stocks={stocks} loading={stocksLoading} />

            {/* Analysis mode toggle */}
            <div className="ml-auto flex items-center gap-2">
              <span className="text-xs" style={{ color: TEXT_MUT }}>Analysis scope:</span>
              <div className="flex items-center p-0.5 rounded-lg"
                style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER}` }}>
                {(["broad_market", "in_sector"] as const).map(mode => (
                  <button key={mode} onClick={() => setAnalysisMode(mode)}
                    className="px-3 py-1.5 rounded-md text-xs font-semibold"
                    style={analysisMode === mode
                      ? { background: BLUE, color: "white" }
                      : { color: TEXT_SEC, background: "transparent" }}>
                    {mode === "broad_market" ? "Broad Market" : "In-Sector"}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* ── Price Chart + Fundamentals ── */}
          <div className="flex gap-4">
            <PriceChart symbol={selectedStock} />
            <FundamentalsPanel symbol={selectedStock} />
          </div>

          {/* ── Lag Alignment Lab ── */}
          <LagAlignmentLab analysisMode={analysisMode} />

          {/* ── Network Graph ── */}
          <LeadLagNetwork />

        </div>
      </main>
    </div>
  );
}
