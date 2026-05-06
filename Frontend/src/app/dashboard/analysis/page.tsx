// frontend/src/app/dashboard/analysis/page.tsx
"use client";
import MonteCarlo from "@/components/ui/MonteCarlo";
import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import Sidebar from "@/components/ui/Sidebar";
import { ChevronDown, Loader2, AlertTriangle, Search, Info } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Brush,
} from "recharts";
import * as d3 from "d3";
import {
  fetchOHLCV,
  fetchStockDetail,
  fetchPairData,
  fetchNetwork,
  fetchAllStocks,
  type OHLCVCandle,
  type StockDetail,
  type PairDetail,
  type StockSummary,
  type AnalysisMode,
  type NetworkNode,
  type NetworkEdge,
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

// ─── Network: sector colors + D3 force helpers ────────────────────────────────

// Sector → hue for cluster ring positioning and UI accents.
// Includes both GICS names and Morningstar/Yahoo Finance variants.
const SECTOR_COLORS: Record<string, string> = {
  "Technology":              "hsl(217, 91%, 60%)",
  "Information Technology":  "hsl(217, 91%, 60%)",
  "Financials":              "hsl(152, 65%, 45%)",
  "Financial Services":      "hsl(152, 65%, 45%)",
  "Health Care":             "hsl(270, 70%, 65%)",
  "Healthcare":              "hsl(270, 70%, 65%)",
  "Consumer Discretionary":  "hsl(28,  90%, 55%)",
  "Consumer Cyclical":       "hsl(28,  90%, 55%)",
  "Consumer Staples":        "hsl(355, 80%, 60%)",
  "Industrials":             "hsl(195, 80%, 50%)",
  "Energy":                  "hsl(50,  80%, 48%)",
  "Materials":               "hsl(90,  55%, 48%)",
  "Real Estate":             "hsl(330, 70%, 60%)",
  "Communication Services":  "hsl(260, 60%, 65%)",
  "Utilities":               "hsl(175, 60%, 48%)",
};
const sectorColor = (s: string) => SECTOR_COLORS[s] ?? "hsl(215, 15%, 55%)";

/**
 * Node radius based on total degree (in + out edges in the visible graph).
 * More connections → larger node.  Range: 7–20 px.
 */
function nodeRadiusByDegree(degree: number): number {
  return Math.max(7, Math.min(20, 7 + Math.sqrt(degree) * 2.8));
}

/**
 * Node fill color based on net leadership score.
 *  score = (outDeg − inDeg) / (outDeg + inDeg)  ∈ [−1, 1]
 *  +1 → pure leader   → teal   hsl(197, 88%, 55%)
 *   0 → balanced      → slate  hsl(240, 18%, 58%)
 *  −1 → pure follower → violet hsl(280, 58%, 65%)
 *
 * Teal echoes the app's blue accent; violet is complementary and reads
 * clearly on dark backgrounds — neither color clashes with UI chrome.
 */
function netLeadershipColor(outDeg: number, inDeg: number): string {
  const total = outDeg + inDeg;
  if (total === 0) return "hsl(215,15%,55%)";
  const score = (outDeg - inDeg) / total;   // [−1, 1]
  if (score >= 0) {
    // balanced slate → leader teal
    const h = Math.round(240 - score * 43);  // 240 → 197
    const s = Math.round(18  + score * 70);  // 18  → 88
    const l = Math.round(58  - score * 3);   // 58  → 55
    return `hsl(${h},${s}%,${l}%)`;
  } else {
    // balanced slate → follower violet
    const t = -score;
    const h = Math.round(240 + t * 40);      // 240 → 280
    const s = Math.round(18  + t * 40);      // 18  → 58
    const l = Math.round(58  + t * 7);       // 58  → 65
    return `hsl(${h},${s}%,${l}%)`;
  }
}

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

function SectorDropdown({
  value, onChange, sectors,
}: {
  value: string; onChange: (s: string) => void; sectors: string[];
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const h = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between gap-2 px-3 py-2 rounded-lg text-xs font-medium"
        style={{ background: "hsl(215,25%,11%)", border: `1px solid ${BORDER_D}`, color: TEXT_PRI }}>
        <span className="truncate">{value}</span>
        <ChevronDown className="w-3 h-3 flex-shrink-0" style={{ color: TEXT_SEC }} />
      </button>
      {open && (
        <div className="absolute top-full left-0 right-0 mt-1 rounded-lg overflow-hidden z-20"
          style={{ background: "hsl(215,25%,13%)", border: `1px solid ${BORDER}`, boxShadow: "0 8px 24px rgba(0,0,0,0.5)" }}>
          <div className="max-h-48 overflow-y-auto">
            {sectors.map(s => (
              <button key={s}
                onClick={() => { onChange(s); setOpen(false); }}
                className="w-full flex items-center px-3 py-2 text-left text-xs"
                style={{ background: s === value ? BLUE_DIM : "transparent", color: s === value ? BLUE : TEXT_PRI }}
                onMouseEnter={e => { if (s !== value) (e.currentTarget as HTMLButtonElement).style.background = "hsl(215,25%,17%)"; }}
                onMouseLeave={e => { if (s !== value) (e.currentTarget as HTMLButtonElement).style.background = "transparent"; }}>
                {s}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

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
              formatter={(v, name) => [v != null ? `${(v as number).toFixed(1)}` : "", String(name)]}
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
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState<string | null>(null);
  // Lag indicator: index of the currently-hovered data point in chartData
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

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
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 4, right: 8, bottom: 4, left: 0 }}
              onMouseMove={(data: any) => {
                if (data?.activeLabel !== undefined) {
                  const idx = chartData.findIndex((d: any) => d.date === data.activeLabel);
                  setHoveredIndex(idx >= 0 ? idx : null);
                }
              }}
              onMouseLeave={() => setHoveredIndex(null)}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={BORDER} />
              <XAxis dataKey="date" axisLine={false} tickLine={false} tick={labelStyle}
                interval="preserveStartEnd" minTickGap={40} />
              <YAxis domain={["auto", "auto"]} axisLine={false} tickLine={false} tick={labelStyle}
                width={52}
                tickFormatter={v => showResiduals ? `${v > 0 ? "+" : ""}${v.toFixed(1)}%` : `${v.toFixed(0)}`}
              />
              <Tooltip
                cursor={{ stroke: BORDER, strokeWidth: 1 }}
                content={(props: any) => {
                  const { payload, label, active } = props;
                  if (!active || !payload?.length || !label) return null;

                  const fmt = (v: number) =>
                    showResiduals ? `${v > 0 ? "+" : ""}${v.toFixed(2)}%` : v.toFixed(2);

                  const leaderVal   = payload.find((p: any) => p.dataKey === "leader")?.value as number | undefined;
                  const followerVal = payload.find((p: any) => p.dataKey === "follower")?.value as number | undefined;

                  // Use label to look up index so tooltip is always in sync with active point
                  const currentIdx  = chartData.findIndex((d: any) => d.date === label);
                  const hasLag      = pairData?.found && (pairData.best_lag ?? 0) > 0 && currentIdx >= 0;
                  const rawLagIdx   = hasLag ? currentIdx + pairData!.best_lag : null;
                  const lagIdx      = rawLagIdx !== null ? Math.min(rawLagIdx, chartData.length - 1) : null;
                  const lagPoint    = lagIdx !== null ? chartData[lagIdx] : null;
                  const lagFollower = lagPoint?.follower as number | undefined;
                  const lagDate     = lagPoint?.date as string | undefined;
                  // flag if the projected date falls outside the loaded chart range
                  const lagOutOfRange = rawLagIdx !== null && rawLagIdx >= chartData.length;

                  const row = (label: string, value: number, color: string, dimmed = false) => (
                    <div style={{ display: "flex", justifyContent: "space-between", gap: 20, marginBottom: 3 }}>
                      <span style={{ color, opacity: dimmed ? 0.65 : 1, fontSize: 11 }}>{label}</span>
                      <span style={{ color: "rgba(226,232,240,0.9)", fontWeight: 600, fontSize: 11 }}>{fmt(value)}</span>
                    </div>
                  );

                  return (
                    <div style={{
                      background: "hsl(215,25%,13%)",
                      border: `1px solid ${BORDER}`,
                      borderRadius: 8,
                      padding: "10px 12px",
                      minWidth: 190,
                    }}>
                      {/* ── Same-day section ── */}
                      <p style={{ color: TEXT_MUT, fontSize: 10, fontWeight: 600,
                        textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 6 }}>
                        Same day · {label}
                      </p>
                      {leaderVal   !== undefined && row(leaderTicker,   leaderVal,   LEADER_COLOR)}
                      {followerVal !== undefined && row(followerTicker, followerVal, FOLLOWER_COLOR)}

                      {/* ── Lag projection section ── */}
                      {hasLag && lagFollower !== undefined && (
                        <>
                          <div style={{ borderTop: `1px solid ${BORDER}`, margin: "8px 0 6px" }} />
                          <p style={{ color: TEXT_MUT, fontSize: 10, fontWeight: 600,
                            textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 6 }}>
                            +{pairData!.best_lag}d lag projection
                            {lagDate && (
                              <span style={{ fontWeight: 400, textTransform: "none", marginLeft: 4 }}>
                                · {lagDate}{lagOutOfRange ? " (est.)" : ""}
                              </span>
                            )}
                          </p>
                          {row(followerTicker, lagFollower, FOLLOWER_COLOR, true)}
                          {lagOutOfRange && (
                            <p style={{ color: TEXT_MUT, fontSize: 9, marginTop: 2 }}>
                              Projected date beyond chart range
                            </p>
                          )}
                        </>
                      )}
                    </div>
                  );
                }}
              />
              {showResiduals && <ReferenceLine y={0} stroke={BORDER} strokeDasharray="4 4" />}

              {/* ── Lag indicator: hover date + projected follower reaction date ── */}
              {hoveredIndex !== null && pairData?.found && pairData.best_lag > 0 && (() => {
                const hoverDate = chartData[hoveredIndex]?.date;
                const lagIdx    = Math.min(hoveredIndex + pairData.best_lag, chartData.length - 1);
                const lagDate   = chartData[lagIdx]?.date;
                return (
                  <>
                    {hoverDate && (
                      <ReferenceLine x={hoverDate} stroke={LEADER_COLOR}
                        strokeDasharray="4 3" strokeOpacity={0.7} strokeWidth={1.5} />
                    )}
                    {lagDate && lagDate !== hoverDate && (
                      <ReferenceLine x={lagDate} stroke={FOLLOWER_COLOR}
                        strokeDasharray="4 3" strokeOpacity={0.7} strokeWidth={1.5}
                        label={{ value: `+${pairData.best_lag}d`, fill: FOLLOWER_COLOR, fontSize: 10, position: "insideTopRight" }}
                      />
                    )}
                  </>
                );
              })()}

              <Line type="monotone" dataKey="leader"   stroke={LEADER_COLOR}   strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="follower" stroke={FOLLOWER_COLOR} strokeWidth={2} dot={false} isAnimationActive={false} />

              {/* ── Brush for fine-grained range selection ── */}
              <Brush dataKey="date" height={22} stroke={BORDER}
                fill="hsl(215,25%,9%)" travellerWidth={6}
                startIndex={0} endIndex={chartData.length - 1} />
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
        <div className="flex flex-col gap-2 mt-3">
          <div className="flex gap-5 flex-wrap">
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
          {/* Lag indicator disclaimer — only shown when relationship is detected */}
          {pairData?.found && pairData.best_lag > 0 && (
            <div className="flex items-start gap-1.5">
              <Info className="w-3 h-3 flex-shrink-0 mt-0.5" style={{ color: TEXT_MUT }} />
              <p className="text-[10px] leading-relaxed" style={{ color: TEXT_MUT }}>
                Hover the chart to see the projected {pairData.best_lag}-day lag offset (
                <span style={{ color: LEADER_COLOR }}>leader dashed line</span> →{" "}
                <span style={{ color: FOLLOWER_COLOR }}>follower reaction line</span>
                ). This lag is a statistical average measured over rolling windows — individual price echoes will vary.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Lead-Lag Network (real data, D3 force layout) ───────────────────────────

const SIM_W      = 1600;
const SIM_H      = 900;
const ZOOM_STEP  = 0.25;
const ZOOM_MIN   = 0.25;
const ZOOM_MAX   = 3.0;
const ZOOM_INIT  = 0.5;

function LeadLagNetwork({ analysisMode }: { analysisMode: AnalysisMode }) {
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const animRef    = useRef<number>(0);
  const isDragging = useRef(false);
  const dragStart  = useRef({ mx: 0, my: 0, px: 0, py: 0 });

  // Sim dimensions change based on node count — stored in a ref so the draw
  // loop always reads the latest values without being in the dependency array.
  const simDimsRef = useRef({ w: SIM_W, h: SIM_H });

  // Device pixel ratio — kept in a ref so it's readable every frame without
  // causing re-renders. Set/updated by the resize observer below.
  const dprRef = useRef(typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1);

  const [allNodes,          setAllNodes]          = useState<NetworkNode[]>([]);
  const [allEdges,          setAllEdges]           = useState<NetworkEdge[]>([]);
  const [positions,         setPositions]          = useState<Record<string, { x: number; y: number }>>({});
  const [loading,           setLoading]            = useState(false);
  const [error,             setError]              = useState<string | null>(null);
  const [nodeLimit,         setNodeLimit]          = useState(25);
  const [minSignal,         setMinSignal]          = useState(70);
  const [maxEdgesPerNode,   setMaxEdgesPerNode]    = useState(6);
  const [selectedSector,    setSelectedSector]     = useState("All");
  const [activeNode,        setActiveNode]         = useState<NetworkNode | null>(null);
  const [hoverNodeId,       setHoverNodeId]        = useState<string | null>(null);
  const [zoom,              setZoom]               = useState(ZOOM_INIT);
  const [pan,               setPan]                = useState({ x: 0, y: 0 });

  // ── HiDPI canvas sizing ─────────────────────────────────────────────────────
  // Sets the canvas pixel buffer to CSS size × devicePixelRatio so rendering
  // is never blurry on Retina / high-DPI displays.
  // The canvas element is always in the DOM (never conditionally unmounted),
  // so this effect fires exactly once on mount and the ResizeObserver handles
  // any subsequent container size changes (e.g. window resize).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const applyDpr = () => {
      const dpr  = window.devicePixelRatio || 1;
      dprRef.current = dpr;
      const cssW = canvas.offsetWidth;
      const cssH = canvas.offsetHeight;
      if (!cssW || !cssH) return;
      canvas.width  = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    };
    applyDpr();
    const ro = new ResizeObserver(applyDpr);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  // ── Fetch ───────────────────────────────────────────────────────────────────
  useEffect(() => {
    setLoading(true); setError(null); setPositions({});
    setZoom(ZOOM_INIT); setPan({ x: 0, y: 0 });
    fetchNetwork(analysisMode, 55, nodeLimit)
      .then(data => {
        setAllNodes(data.nodes);
        setAllEdges(data.edges);
        setActiveNode(data.nodes[0] ?? null);
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [analysisMode, nodeLimit]);

  // ── D3 force layout — sector-clustered, dynamically scaled by node count ─────
  useEffect(() => {
    if (!allNodes.length) return;

    const N = allNodes.length;

    // Scale sim space, repulsion, and link distance proportionally to node count
    // so sparse graphs are tighter and dense graphs have more breathing room.
    const SW        = Math.round(700  + N * 30);
    const SH        = Math.round(450  + N * 18);
    const charge    = -(220 + N * 14);
    const linkDist  = 110 + N * 2.5;
    const ringRadX  = SW * 0.30;
    const ringRadY  = SH * 0.30;
    simDimsRef.current = { w: SW, h: SH };

    // Total degree per node across ALL edges (used for collide radius)
    const allDeg: Record<string, number> = {};
    allNodes.forEach(n => { allDeg[n.id] = 0; });
    allEdges.forEach(e => {
      allDeg[e.source] = (allDeg[e.source] ?? 0) + 1;
      allDeg[e.target] = (allDeg[e.target] ?? 0) + 1;
    });

    // Assign each unique sector a position around a ring
    const sectors = [...new Set(allNodes.map(n => n.sector))].filter(Boolean);
    const sectorCenter: Record<string, { x: number; y: number }> = {};
    sectors.forEach((s, i) => {
      const angle = (i / sectors.length) * Math.PI * 2 - Math.PI / 2;
      sectorCenter[s] = {
        x: SW / 2 + Math.cos(angle) * ringRadX,
        y: SH / 2 + Math.sin(angle) * ringRadY,
      };
    });

    const simNodes: any[] = allNodes.map(n => {
      const base = sectorCenter[n.sector] ?? { x: SW / 2, y: SH / 2 };
      return {
        ...n,
        degree: allDeg[n.id] ?? 0,
        x: base.x + (Math.random() - 0.5) * 200,
        y: base.y + (Math.random() - 0.5) * 200,
      };
    });
    const idSet    = new Set(simNodes.map((n: any) => n.id));
    const simLinks: any[] = allEdges
      .filter(e => idSet.has(e.source) && idSet.has(e.target))
      .map(e => ({ source: e.source, target: e.target }));

    const sim = (d3 as any)
      .forceSimulation(simNodes)
      .force("link",    (d3 as any).forceLink(simLinks).id((d: any) => d.id).distance(linkDist).strength(0.12))
      .force("charge",  (d3 as any).forceManyBody().strength(charge))
      .force("x",       (d3 as any).forceX((d: any) => sectorCenter[d.sector]?.x ?? SW / 2).strength(0.07))
      .force("y",       (d3 as any).forceY((d: any) => sectorCenter[d.sector]?.y ?? SH / 2).strength(0.07))
      .force("collide", (d3 as any).forceCollide().radius((d: any) => nodeRadiusByDegree(d.degree) + 32));

    sim.tick(600);
    sim.stop();

    const pos: Record<string, { x: number; y: number }> = {};
    simNodes.forEach((n: any) => {
      pos[n.id] = {
        x: Math.max(40, Math.min(SW - 40, n.x ?? SW / 2)),
        y: Math.max(40, Math.min(SH - 40, n.y ?? SH / 2)),
      };
    });
    setPositions(pos);
  }, [allNodes, allEdges]);

  // ── Derived sets ─────────────────────────────────────────────────────────────
  const uniqueSectors = useMemo(
    () => ["All", ...[...new Set(allNodes.map(n => n.sector))].sort()],
    [allNodes]
  );
  const filteredNodes = useMemo(
    () => allNodes.filter(n => selectedSector === "All" || n.sector === selectedSector),
    [allNodes, selectedSector]
  );
  const filteredNodeIds = useMemo(() => new Set(filteredNodes.map(n => n.id)), [filteredNodes]);
  const filteredEdges   = useMemo(() => {
    // Step 1: basic signal + sector filter
    const passing = allEdges.filter(e =>
      e.signal_strength >= minSignal &&
      filteredNodeIds.has(e.source) &&
      filteredNodeIds.has(e.target)
    );
    // Step 2: cap outgoing edges per source node to the top N by signal strength,
    // so high-degree hubs don't generate overwhelming visual clutter.
    const countBySource: Record<string, number> = {};
    const capped: typeof passing = [];
    // Edges are already ordered by signal_strength DESC from the backend
    for (const e of passing) {
      const c = countBySource[e.source] ?? 0;
      if (c < maxEdgesPerNode) {
        capped.push(e);
        countBySource[e.source] = c + 1;
      }
    }
    return capped;
  }, [allEdges, minSignal, filteredNodeIds, maxEdgesPerNode]);
  const activeFollowers = useMemo(
    () => filteredEdges
      .filter(e => e.source === activeNode?.id)
      .sort((a, b) => b.signal_strength - a.signal_strength)
      .slice(0, 8),
    [activeNode, filteredEdges]
  );

  // ── Per-node in/out degree in the visible (filtered) graph ────────────────────
  const degreeMap = useMemo(() => {
    const map: Record<string, { out: number; in: number }> = {};
    filteredNodes.forEach(n => { map[n.id] = { out: 0, in: 0 }; });
    filteredEdges.forEach(e => {
      if (map[e.source]) map[e.source].out++;
      if (map[e.target]) map[e.target].in++;
    });
    return map;
  }, [filteredNodes, filteredEdges]);

  // ── Particles ────────────────────────────────────────────────────────────────
  const particleRef = useRef<{ edge: NetworkEdge; t: number }[]>([]);
  useEffect(() => {
    particleRef.current = filteredEdges.map(e => ({ edge: e, t: Math.random() }));
  }, [filteredEdges]);

  // ── Draw ─────────────────────────────────────────────────────────────────────
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !Object.keys(positions).length) {
      animRef.current = requestAnimationFrame(draw);
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Work in CSS pixels throughout — setTransform handles the DPR scaling so
    // every draw call below uses the same units as the layout / hit-test code.
    const dpr = dprRef.current;
    const CW  = canvas.width  / dpr;
    const CH  = canvas.height / dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);  // replaces any previous transform (no compounding)

    ctx.clearRect(0, 0, CW, CH);
    ctx.fillStyle = "hsl(215,30%,8%)";
    ctx.fillRect(0, 0, CW, CH);

    // Sim → screen coordinate helpers (use dynamic dims from latest layout run)
    const { w: SW, h: SH } = simDimsRef.current;
    const toSx = (wx: number) => (wx - SW / 2) * zoom + CW / 2 + pan.x;
    const toSy = (wy: number) => (wy - SH / 2) * zoom + CH / 2 + pan.y;
    const sx    = (id: string) => toSx(positions[id]?.x ?? SIM_W / 2);
    const sy    = (id: string) => toSy(positions[id]?.y ?? SIM_H / 2);

    // Edges + arrowheads
    filteredEdges.forEach(e => {
      if (!positions[e.source] || !positions[e.target]) return;
      const x1 = sx(e.source), y1 = sy(e.source);
      const x2 = sx(e.target), y2 = sy(e.target);
      const alpha = Math.min(1, (e.signal_strength - minSignal) / (100 - minSignal) + 0.15);

      let edgeColor = `rgba(148,163,184,${alpha * 0.45})`;
      if (hoverNodeId) {
        if (e.source === hoverNodeId)      edgeColor = "rgba(34,197,94,0.85)";
        else if (e.target === hoverNodeId) edgeColor = "rgba(239,68,68,0.75)";
        else                               edgeColor = "rgba(100,116,139,0.05)";
      }
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
      ctx.strokeStyle = edgeColor; ctx.lineWidth = Math.max(0.5, alpha * 1.8); ctx.stroke();

      // Arrowhead at target
      const tgtDeg = (degreeMap[e.target]?.out ?? 0) + (degreeMap[e.target]?.in ?? 0);
      const tgtR   = Math.max(5, nodeRadiusByDegree(tgtDeg) * zoom);
      const angle = Math.atan2(y2 - y1, x2 - x1);
      const arr   = 6;
      const ex    = x2 - Math.cos(angle) * (tgtR + 3);
      const ey    = y2 - Math.sin(angle) * (tgtR + 3);
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - arr * Math.cos(angle - 0.4), ey - arr * Math.sin(angle - 0.4));
      ctx.lineTo(ex - arr * Math.cos(angle + 0.4), ey - arr * Math.sin(angle + 0.4));
      ctx.closePath(); ctx.fillStyle = edgeColor; ctx.fill();
    });

    // Particles along edges — slow speed to reduce visual clutter
    particleRef.current.forEach(p => {
      p.t = (p.t + (p.edge.signal_strength / 100) * 0.001) % 1;
      if (!positions[p.edge.source] || !positions[p.edge.target]) return;
      const x = sx(p.edge.source) + (sx(p.edge.target) - sx(p.edge.source)) * p.t;
      const y = sy(p.edge.source) + (sy(p.edge.target) - sy(p.edge.source)) * p.t;
      ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.55)"; ctx.fill();
    });

    // Nodes + always-visible labels
    filteredNodes.forEach(n => {
      if (!positions[n.id]) return;
      const x  = sx(n.id), y = sy(n.id);
      const deg = (degreeMap[n.id]?.out ?? 0) + (degreeMap[n.id]?.in ?? 0);
      const r  = Math.max(5, nodeRadiusByDegree(deg) * zoom);
      const isHov    = n.id === hoverNodeId;
      const isActive = n.id === activeNode?.id;
      const connected = hoverNodeId
        ? filteredEdges.some(e =>
            (e.source === hoverNodeId && e.target === n.id) ||
            (e.target === hoverNodeId && e.source === n.id))
        : false;
      const dimmed = !!(hoverNodeId && !isHov && !connected);
      const col    = netLeadershipColor(degreeMap[n.id]?.out ?? 0, degreeMap[n.id]?.in ?? 0);

      ctx.globalAlpha = dimmed ? 0.15 : 1.0;

      // Glow ring for active / hovered
      if (isActive || isHov) {
        ctx.beginPath(); ctx.arc(x, y, r + 5, 0, Math.PI * 2);
        ctx.strokeStyle = col; ctx.lineWidth = 2;
        ctx.globalAlpha = dimmed ? 0.05 : 0.35; ctx.stroke();
        ctx.globalAlpha = dimmed ? 0.15 : 1.0;
      }

      // Node circle
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = col; ctx.fill();

      // Always-visible ticker label below node (drawn in screen space — fixed font size)
      const fontSize = Math.max(9, Math.min(13, 10 * zoom));
      ctx.font = `bold ${fontSize}px sans-serif`;
      ctx.textAlign    = "center";
      ctx.textBaseline = "top";
      const textW  = ctx.measureText(n.id).width;
      const labelX = x;
      const labelY = y + r + 3;
      // dark pill background for contrast
      ctx.fillStyle = "rgba(10,15,25,0.72)";
      ctx.fillRect(labelX - textW / 2 - 2, labelY, textW + 4, fontSize + 3);
      ctx.fillStyle = dimmed ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.92)";
      ctx.fillText(n.id, labelX, labelY + 1);

      ctx.globalAlpha = 1.0;
    });

    animRef.current = requestAnimationFrame(draw);
  }, [filteredNodes, filteredEdges, degreeMap, positions, hoverNodeId, activeNode, minSignal, zoom, pan]);

  useEffect(() => {
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [draw]);

  // ── Hit test (inverse transform: screen → sim) ────────────────────────────
  const getNodeAt = useCallback((mx: number, my: number, canvas: HTMLCanvasElement) => {
    const rect = canvas.getBoundingClientRect();
    // Mouse position in CSS pixels — matches the coordinate space used in draw()
    const cx   = mx - rect.left;
    const cy   = my - rect.top;
    const dpr  = dprRef.current;
    const CW   = canvas.width  / dpr;
    const CH   = canvas.height / dpr;
    const { w: SW, h: SH } = simDimsRef.current;
    const wx = (cx - CW / 2 - pan.x) / zoom + SW / 2;
    const wy = (cy - CH / 2 - pan.y) / zoom + SH / 2;
    return filteredNodes.find(n => {
      if (!positions[n.id]) return false;
      const deg = (degreeMap[n.id]?.out ?? 0) + (degreeMap[n.id]?.in ?? 0);
      return Math.hypot(wx - positions[n.id].x, wy - positions[n.id].y) <= nodeRadiusByDegree(deg) + 6;
    }) ?? null;
  }, [filteredNodes, degreeMap, positions, zoom, pan]);

  // ── Mouse / drag handlers ─────────────────────────────────────────────────
  const onCanvasMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    isDragging.current = true;
    dragStart.current = { mx: e.clientX, my: e.clientY, px: pan.x, py: pan.y };
    if (canvasRef.current) canvasRef.current.style.cursor = "grabbing";
  }, [pan]);

  const onCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging.current) {
      setPan({
        x: dragStart.current.px + e.clientX - dragStart.current.mx,
        y: dragStart.current.py + e.clientY - dragStart.current.my,
      });
      setHoverNodeId(null);
      if (canvasRef.current) canvasRef.current.style.cursor = "grabbing";
    } else {
      const n = getNodeAt(e.clientX, e.clientY, canvasRef.current!);
      setHoverNodeId(n ? n.id : null);
      if (canvasRef.current) canvasRef.current.style.cursor = n ? "pointer" : "grab";
    }
  }, [getNodeAt]);

  const onCanvasMouseUp = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging.current) return;
    const dx = e.clientX - dragStart.current.mx;
    const dy = e.clientY - dragStart.current.my;
    isDragging.current = false;
    // Treat as click only if the mouse barely moved (not a pan)
    if (Math.hypot(dx, dy) < 4) {
      const n = getNodeAt(e.clientX, e.clientY, canvasRef.current!);
      if (n) setActiveNode(n);
    }
    if (canvasRef.current) {
      const n = canvasRef.current ? getNodeAt(e.clientX, e.clientY, canvasRef.current) : null;
      canvasRef.current.style.cursor = n ? "pointer" : "grab";
    }
  }, [getNodeAt]);

  const onCanvasMouseLeave = useCallback(() => {
    isDragging.current = false;
    setHoverNodeId(null);
  }, []);

  // ── Zoom controls ─────────────────────────────────────────────────────────
  const zoomIn    = () => setZoom(z => parseFloat(Math.min(ZOOM_MAX, z + ZOOM_STEP).toFixed(2)));
  const zoomOut   = () => setZoom(z => parseFloat(Math.max(ZOOM_MIN, z - ZOOM_STEP).toFixed(2)));
  const zoomReset = () => { setZoom(ZOOM_INIT); setPan({ x: 0, y: 0 }); };

  const ZoomBtn = ({ onClick, label }: { onClick: () => void; label: string }) => (
    <button onClick={onClick}
      className="w-7 h-7 flex items-center justify-center rounded text-sm font-bold select-none"
      style={{ background: "hsl(215,25%,15%)", border: `1px solid ${BORDER_D}`, color: TEXT_PRI }}>
      {label}
    </button>
  );

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="rounded-xl p-5" style={{ background: "hsl(215,25%,11%)", border: `1px solid ${BORDER}` }}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-5 pb-4"
        style={{ borderBottom: `1px solid ${BORDER}` }}>
        <div>
          <h2 className="text-base font-bold" style={{ color: TEXT_PRI }}>Lead-Lag Network Analytics Lab</h2>
          <p className="text-xs mt-0.5" style={{ color: TEXT_MUT }}>
            Top 50 stocks · node color = net leadership · node size = connections · nodes clustered by sector · drag to pan · +/− to zoom
          </p>
        </div>
        {/* Net leadership color scale */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-[10px] font-semibold" style={{ color: "hsl(280,58%,65%)" }}>Follower</span>
          <div className="w-24 h-2.5 rounded-full" style={{
            background: "linear-gradient(to right, hsl(280,58%,65%), hsl(240,18%,58%), hsl(197,88%,55%))"
          }} />
          <span className="text-[10px] font-semibold" style={{ color: "hsl(197,88%,55%)" }}>Leader</span>
        </div>
      </div>

      {/* Grid is always rendered — filters stay visible during node-count refetches.
           Loading and error states are overlaid on the canvas area only.
           This also keeps the canvas element permanently in the DOM so the DPR
           resize effect never needs to re-fire after a canvas unmount/remount. */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">

        {/* ── Sidebar ────────────────────────────────────────────────────────── */}
        <div className="space-y-4">
          <div className="rounded-lg p-4" style={{ background: CARD_H, border: `1px solid ${BORDER_D}` }}>
            <p className="text-[10px] font-bold uppercase tracking-widest mb-3" style={{ color: TEXT_MUT }}>Filters</p>
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-semibold mb-1" style={{ color: TEXT_PRI }}>
                  Nodes: <span style={{ color: BLUE }}>{nodeLimit}</span>
                </label>
                <input type="range" min={10} max={50} step={5} value={nodeLimit}
                  onChange={e => setNodeLimit(Number(e.target.value))}
                  className="w-full accent-blue-500" />
              </div>
              <div>
                <label className="block text-xs font-semibold mb-1.5" style={{ color: TEXT_PRI }}>Sector</label>
                <SectorDropdown
                  value={selectedSector}
                  onChange={setSelectedSector}
                  sectors={uniqueSectors}
                />
              </div>
              <div>
                <label className="block text-xs font-semibold mb-1" style={{ color: TEXT_PRI }}>
                  Min Signal: <span style={{ color: BLUE }}>{minSignal}</span>
                </label>
                <input type="range" min={55} max={95} step={5} value={minSignal}
                  onChange={e => setMinSignal(Number(e.target.value))}
                  className="w-full accent-blue-500" />
              </div>
              <div>
                <label className="block text-xs font-semibold mb-1" style={{ color: TEXT_PRI }}>
                  Max edges/node: <span style={{ color: BLUE }}>{maxEdgesPerNode}</span>
                </label>
                <input type="range" min={2} max={12} step={1} value={maxEdgesPerNode}
                  onChange={e => setMaxEdgesPerNode(Number(e.target.value))}
                  className="w-full accent-blue-500" />
              </div>
            </div>
          </div>

          {/* Selected node — hidden while loading so stale degree stats aren't shown */}
          {activeNode && !loading && (
            <div className="rounded-lg p-4" style={{ background: CARD_H, border: `1px solid ${BORDER_D}` }}>
              <p className="text-[10px] font-bold uppercase tracking-widest mb-3" style={{ color: TEXT_MUT }}>Selected Node</p>
              <p className="text-base font-bold mb-0.5" style={{ color: TEXT_PRI }}>{activeNode.id}</p>
              <p className="text-xs mb-3" style={{ color: sectorColor(activeNode.sector) }}>{activeNode.sector}</p>
              {(() => {
                const outD = degreeMap[activeNode.id]?.out ?? 0;
                const inD  = degreeMap[activeNode.id]?.in  ?? 0;
                const total = outD + inD;
                const score = total > 0 ? ((outD - inD) / total * 100).toFixed(0) : "—";
                const scoreNum = total > 0 ? (outD - inD) / total : 0;
                const scoreColor = netLeadershipColor(outD, inD);
                return [
                  { label: "Leads",       value: outD },
                  { label: "Follows",     value: inD },
                  { label: "Connections", value: total },
                  { label: "Net score",   value: `${scoreNum >= 0 ? "+" : ""}${score}%`, color: scoreColor },
                  { label: "Centrality",  value: activeNode.centrality.toFixed(5) },
                ].map(({ label, value, color }) => (
                  <div key={label} className="flex justify-between py-2" style={{ borderBottom: `1px solid ${BORDER}` }}>
                    <span className="text-xs" style={{ color: TEXT_SEC }}>{label}</span>
                    <span className="text-xs font-semibold" style={{ color: color ?? TEXT_PRI }}>{value}</span>
                  </div>
                ));
              })()}
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
                        {e.signal_strength.toFixed(0)}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>

        {/* ── Canvas area ────────────────────────────────────────────────────── */}
        {/* Fixed height — prevents sidebar growing the container and stretching
             the canvas buffer. Canvas element is always in the DOM so the DPR
             resize effect fires exactly once on mount and never re-runs. */}
        <div className="lg:col-span-3 rounded-lg overflow-hidden relative"
          style={{ background: "hsl(215,30%,8%)", height: 520 }}>

          {/* Loading overlay — covers canvas while fetching new data */}
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center z-10"
              style={{ background: "hsl(215,30%,8%)" }}>
              <Loader2 className="w-7 h-7 animate-spin" style={{ color: BLUE }} />
            </div>
          )}

          {/* Error overlay */}
          {!loading && error && (
            <div className="absolute inset-0 flex items-center justify-center gap-2 z-10"
              style={{ background: "hsl(215,30%,8%)" }}>
              <AlertTriangle className="w-5 h-5" style={{ color: RED }} />
              <p className="text-sm" style={{ color: RED }}>{error}</p>
            </div>
          )}

          {/* Zoom controls */}
          <div className="absolute top-2 right-2 flex flex-col gap-1 z-10">
            <ZoomBtn onClick={zoomIn}    label="+" />
            <ZoomBtn onClick={zoomOut}   label="−" />
            <ZoomBtn onClick={zoomReset} label="⟳" />
          </div>
          {/* Zoom level indicator */}
          <div className="absolute bottom-2 left-2 z-10 text-[10px] px-1.5 py-0.5 rounded"
            style={{ background: "rgba(10,15,25,0.6)", color: TEXT_MUT }}>
            {Math.round(zoom * 100)}%
          </div>

          {/* Canvas — always mounted, initial size matches DPR-aware defaults */}
          <canvas ref={canvasRef} width={900} height={520} className="w-full h-full"
            style={{ cursor: "grab" }}
            onMouseDown={onCanvasMouseDown}
            onMouseMove={onCanvasMouseMove}
            onMouseUp={onCanvasMouseUp}
            onMouseLeave={onCanvasMouseLeave}
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
          <LeadLagNetwork analysisMode={analysisMode} />

          {/* ── Monte Carlo Lab ── */}
          <MonteCarlo stocks={stocks} stocksLoading={stocksLoading} />

        </div>
      </main>
    </div>
  );
}
