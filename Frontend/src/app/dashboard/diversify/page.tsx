// Frontend/src/app/dashboard/diversify/page.tsx
"use client";
import { useState, useRef, useEffect, useMemo } from "react";
import Sidebar from "@/components/ui/Sidebar";
import SpiderChart from "@/components/ui/SpiderChart";
import SectorDonut from "@/components/ui/SectorDonut";
import StockModal, { type Stock } from "@/components/ui/StockModal";
import {
  LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import {
  analyzePortfolio,
  fetchStockSummaries,
  fetchOHLCV,
  fetchStockDetail,
  type OHLCVCandle,
  type StockDetail,
  type OverlapResult,
  type Recommendation,
  type DcorCandidate,
  type AnalysisMode,
} from "@/src/app/lib/api";
import { useAuth } from "@/src/app/context/AuthContext";
import {
  AlertTriangle, TrendingUp, Plus, X,
  ChevronDown, ChevronUp, Loader2, Sparkles, ArrowRight,
  ShieldAlert, BarChart3, Unlink, Info, Zap, Globe, Link2, Layers, Timer,
  Briefcase,
} from "lucide-react";

// ── Design tokens ─────────────────────────────────────────────────────────────
const BG         = "hsl(213, 27%, 7%)";
const CARD       = { background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" };
const CARD_H     = "hsl(215, 25%, 13%)";
const TEXT_PRI   = "hsl(210, 40%, 92%)";
const TEXT_SEC   = "hsl(215, 15%, 55%)";
const TEXT_MUT   = "hsl(215, 15%, 40%)";
const BLUE       = "hsl(217, 91%, 60%)";
const BLUE_DIM   = "hsla(217, 91%, 60%, 0.15)";
const GREEN      = "hsl(142, 71%, 45%)";
const GREEN_DIM  = "hsla(142, 71%, 45%, 0.15)";
const AMBER      = "hsl(38, 92%, 50%)";
const RED        = "hsl(0, 84%, 60%)";
const BORDER     = "hsl(215, 20%, 18%)";
const BORDER_D   = "hsl(215, 20%, 16%)";
const PURPLE     = "hsl(270, 70%, 65%)";
const PURPLE_DIM = "hsla(270, 70%, 65%, 0.15)";

const RANK_COLORS = [
  BLUE, GREEN, AMBER, PURPLE, "hsl(195,80%,50%)",
  "hsl(15,90%,58%)", "hsl(330,70%,60%)", "hsl(90,60%,50%)",
  "hsl(260,60%,65%)", "hsl(50,80%,55%)",
];

const SECTOR_COLORS: Record<string, string> = {
  Technology: BLUE, Financials: GREEN, Healthcare: PURPLE,
  Energy: AMBER, Consumer: "hsl(15, 90%, 58%)",
  Industrials: "hsl(195, 80%, 50%)", Utilities: "hsl(60, 70%, 50%)",
};
const sc = (s: string) => SECTOR_COLORS[s] ?? TEXT_SEC;

// ── Built-in preset portfolios ────────────────────────────────────────────────
const BUILTIN_PRESETS = [
  { label: "Big Tech",   tickers: ["NVDA","AMD","AAPL","MSFT","GOOGL"], saved: false },
  { label: "Financials", tickers: ["JPM","GS","MS","BAC","V"],          saved: false },
  { label: "Energy",     tickers: ["XOM","CVX","COP","SLB"],            saved: false },
  { label: "Healthcare", tickers: ["JNJ","UNH","MRK","PFE","ABBV"],     saved: false },
];

// ── Factor explanations — 5 factors ──────────────────────────────────────────
const FACTOR_EXPLANATIONS = [
  {
    label: "Signal Strength", weight: "40%", color: BLUE, Icon: Zap,
    what: "How statistically robust and economically meaningful the lead-lag relationship is between the candidate stock and your holdings.",
    how: "Frequency-weighted mean signal strength across all connections — pairs that appeared consistently across more 15-year rolling windows contribute proportionally more weight.",
    why: "A high signal score means the relationship is not a statistical fluke — it appeared reliably over time and was historically exploitable.",
  },
  {
    label: "Signal Durability", weight: "15%", color: "hsl(195,80%,50%)", Icon: Timer,
    what: "How long the lead-lag signal persists before decaying — the half-life of the relationship.",
    how: "The best (longest) half-life among all connections this candidate has with your holdings, normalized 0–100 with a 252-day cap.",
    why: "A signal that decays in 20 days gives very little time to act. One persisting 150+ days is structurally more reliable for portfolio construction.",
  },
  {
    label: "Portfolio Coverage", weight: "15%", color: AMBER, Icon: Link2,
    what: "How many of your existing holdings this stock has a detected relationship with.",
    how: "Count of your holdings that appear in a significant lead-lag pair with this candidate, divided by the maximum coverage across all candidates.",
    why: "A stock that connects to four of your holdings is more efficient than one connecting to one — you gain informational coverage across more positions.",
  },
  {
    label: "Sector Diversity", weight: "20%", color: GREEN, Icon: Layers,
    what: "How much new sector exposure this stock adds relative to what you already own.",
    how: "A full bonus (100) if your portfolio has zero stocks in this sector. The bonus scales down proportionally as your sector concentration increases.",
    why: "Different sectors respond to different economic drivers. Adding an underrepresented sector reduces the risk that a single macro event affects your entire portfolio.",
  },
  {
    label: "Market Centrality", weight: "10%", color: PURPLE, Icon: Globe,
    what: "How connected this stock is across the entire 2000-stock market network, not just to your holdings.",
    how: "Normalized eigenvector centrality — stocks that are highly connected to other highly connected stocks score highest. Gracefully zeroed if centrality data is unavailable.",
    why: "Central stocks tend to transmit information efficiently and are more likely to be reliable signal sources.",
  },
];

// ── Shared helpers ────────────────────────────────────────────────────────────

/**
 * Normalize a candle array so the first close equals 100.
 * Used by MiniPairChart to put two price series on the same scale
 * regardless of their absolute prices.
 */
function normalizeCandles(candles: OHLCVCandle[]): number[] {
  if (!candles.length) return [];
  const base = parseFloat(candles[0].close);
  if (!base) return candles.map(() => 100);
  return candles.map(c => parseFloat(((parseFloat(c.close) / base) * 100).toFixed(2)));
}

/**
 * Format a raw market cap number into a human-readable T / B / M string.
 * Used by CompactFundamentals to keep the pill cells narrow.
 */
function formatMarketCap(cap: number | null): string {
  if (cap === null) return "—";
  if (cap >= 1_000_000_000_000) return `$${(cap / 1_000_000_000_000).toFixed(2)}T`;
  if (cap >= 1_000_000_000)     return `$${(cap / 1_000_000_000).toFixed(1)}B`;
  if (cap >= 1_000_000)         return `$${(cap / 1_000_000).toFixed(1)}M`;
  return `$${cap.toLocaleString()}`;
}

// ── Small helpers ─────────────────────────────────────────────────────────────
function SignalBar({ score }: { score: number }) {
  const pct   = Math.min(100, Math.max(0, score));
  const color = pct >= 80 ? GREEN : pct >= 65 ? BLUE : AMBER;
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: BORDER }}>
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-xs font-semibold w-8 text-right" style={{ color }}>{Math.round(pct)}</span>
    </div>
  );
}

function SectorTag({ sector }: { sector: string }) {
  return (
    <span className="text-xs px-2 py-0.5 rounded-full"
      style={{ background: `${sc(sector)}20`, color: sc(sector) }}>
      {sector}
    </span>
  );
}

function DirectionTag({ direction }: { direction: string }) {
  if (direction === "leads_your_holdings")
    return <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: GREEN_DIM, color: GREEN }}>Leads your stocks</span>;
  if (direction === "follows_your_holdings")
    return <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: BLUE_DIM, color: BLUE }}>Follows your stocks</span>;
  return <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: "hsla(38,92%,50%,0.15)", color: AMBER }}>Bidirectional</span>;
}

function SectionHeader({ icon, label, count, color, dimColor, subtitle }: {
  icon: React.ReactNode; label: string; count: number;
  color: string; dimColor: string; subtitle: string;
}) {
  return (
    <div className="mb-4">
      <div className="flex items-center gap-2 mb-1">
        <span style={{ color }}>{icon}</span>
        <h2 className="text-base font-semibold" style={{ color: TEXT_PRI }}>{label}</h2>
        <span className="text-xs px-2 py-0.5 rounded-full ml-1" style={{ background: dimColor, color }}>
          {count} stock{count !== 1 ? "s" : ""}
        </span>
      </div>
      <p className="text-xs" style={{ color: TEXT_MUT }}>{subtitle}</p>
    </div>
  );
}

// ── Factor explanations panel ─────────────────────────────────────────────────
function FactorExplanations() {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl overflow-hidden mb-6" style={CARD}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full px-5 py-4 flex items-center justify-between text-left"
        style={{ background: open ? CARD_H : "transparent" }}
      >
        <div className="flex items-center gap-2">
          <Info className="w-4 h-4 flex-shrink-0" style={{ color: BLUE }} />
          <span className="text-sm font-semibold" style={{ color: TEXT_PRI }}>
            How we score recommendations
          </span>
          <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: BLUE_DIM, color: BLUE }}>
            5 factors
          </span>
        </div>
        {open
          ? <ChevronUp className="w-4 h-4" style={{ color: TEXT_MUT }} />
          : <ChevronDown className="w-4 h-4" style={{ color: TEXT_MUT }} />}
      </button>
      {open && (
        <div className="px-5 pb-5" style={{ borderTop: `1px solid ${BORDER_D}` }}>
          <p className="text-xs mt-4 mb-5 leading-relaxed" style={{ color: TEXT_SEC }}>
            Each recommended stock receives a composite score from 0–100 built from five independent
            dimensions. The score tells you not just <em>what</em> to consider adding, but{" "}
            <em>why</em> — and which aspect of diversification it addresses.
          </p>
          <div className="grid grid-cols-2 gap-4">
            {FACTOR_EXPLANATIONS.map(f => (
              <div key={f.label} className="rounded-lg p-4"
                style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER_D}` }}>
                <div className="flex items-center gap-2 mb-2">
                  <f.Icon className="w-4 h-4 flex-shrink-0" style={{ color: f.color }} />
                  <span className="text-xs font-bold" style={{ color: f.color }}>{f.label}</span>
                  <span className="ml-auto text-xs px-1.5 py-0.5 rounded"
                    style={{ background: `${f.color}20`, color: f.color }}>
                    {f.weight}
                  </span>
                </div>
                <p className="text-xs mb-2 leading-relaxed font-medium" style={{ color: TEXT_PRI }}>{f.what}</p>
                <p className="text-xs mb-1.5 leading-relaxed" style={{ color: TEXT_SEC }}>
                  <span className="font-medium" style={{ color: TEXT_MUT }}>How: </span>{f.how}
                </p>
                <p className="text-xs leading-relaxed" style={{ color: TEXT_SEC }}>
                  <span className="font-medium" style={{ color: TEXT_MUT }}>Why: </span>{f.why}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── CompactFundamentals ───────────────────────────────────────────────────────
// Fetches StockDetail for a ticker lazily on mount (only renders when its
// parent card is in the expanded/open state). Displays Market Cap, P/E,
// 52W High, and 52W Low as four small pill cells in a horizontal row.
// Sector and Industry are intentionally omitted — they're already visible
// on every card's header area.

function CompactFundamentals({ symbol }: { symbol: string }) {
  const [detail,  setDetail]  = useState<StockDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!symbol) return;
    setLoading(true);
    fetchStockDetail(symbol)
      .then(setDetail)
      .catch(() => setDetail(null))
      .finally(() => setLoading(false));
  }, [symbol]);

  if (loading) return (
    <div className="flex justify-center py-3">
      <Loader2 className="w-4 h-4 animate-spin" style={{ color: BLUE }} />
    </div>
  );

  if (!detail) return (
    <p className="text-xs py-2" style={{ color: TEXT_MUT }}>No fundamentals available.</p>
  );

  const items = [
    { label: "Market Cap", value: formatMarketCap(detail.market_cap) },
    { label: "P/E Ratio",  value: detail.pe_ratio != null ? detail.pe_ratio.toFixed(1) : "—" },
    { label: "52W High",   value: detail.high_52w != null ? `$${detail.high_52w.toFixed(2)}` : "—" },
    { label: "52W Low",    value: detail.low_52w  != null ? `$${detail.low_52w.toFixed(2)}`  : "—" },
  ];

  return (
    <div className="grid grid-cols-4 gap-2">
      {items.map(({ label, value }) => (
        <div key={label} className="flex flex-col gap-0.5 px-3 py-2.5 rounded-lg text-center"
          style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER_D}` }}>
          <span className="text-[10px]" style={{ color: TEXT_MUT }}>{label}</span>
          <span className="text-xs font-semibold" style={{ color: TEXT_PRI }}>{value}</span>
        </div>
      ))}
    </div>
  );
}

// ── MiniPairChart ─────────────────────────────────────────────────────────────
// Fetches 3M OHLCV for both the recommended stock (recTicker) and the user's
// most-connected holding (holdingTicker) in parallel. Both series are normalized
// to base 100 so movements are visually comparable regardless of absolute price.
// The rec stock renders as a solid blue line; the holding as a muted dashed line.
// No inputs or controls — everything is pre-configured from the parent card's data.

const MINI_REC_COLOR     = "hsl(217, 91%, 60%)";   // blue  — recommended stock
const MINI_HOLDING_COLOR = "hsl(215, 15%, 55%)";    // muted — user's holding

function MiniPairChart({
  recTicker,
  holdingTicker,
}: {
  recTicker:     string;
  holdingTicker: string;
}) {
  const [chartData, setChartData] = useState<
    { date: string; rec: number; holding: number }[]
  >([]);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  useEffect(() => {
    if (!recTicker || !holdingTicker) return;
    setLoading(true); setError(null);
    Promise.all([
      fetchOHLCV(recTicker,     "3M"),
      fetchOHLCV(holdingTicker, "3M"),
    ])
      .then(([recCandles, holdingCandles]) => {
        const n = Math.min(recCandles.length, holdingCandles.length);
        if (!n) { setChartData([]); return; }
        const rn = normalizeCandles(recCandles.slice(0, n));
        const hn = normalizeCandles(holdingCandles.slice(0, n));
        setChartData(
          recCandles.slice(0, n).map((c, i) => ({
            date:    c.date,
            rec:     rn[i],
            holding: hn[i],
          }))
        );
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [recTicker, holdingTicker]);

  const labelStyle = { fill: TEXT_MUT, fontSize: 10 } as const;

  if (loading) return (
    <div className="h-36 flex items-center justify-center rounded-lg"
      style={{ background: "hsl(215,25%,9%)" }}>
      <Loader2 className="w-4 h-4 animate-spin" style={{ color: BLUE }} />
    </div>
  );

  if (error) return (
    <div className="h-36 flex items-center justify-center gap-2 rounded-lg"
      style={{ background: "hsl(215,25%,9%)" }}>
      <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" style={{ color: RED }} />
      <p className="text-xs" style={{ color: RED }}>{error}</p>
    </div>
  );

  if (!chartData.length) return null;

  return (
    <div>
      <div className="h-36">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={BORDER} />
            <XAxis
              dataKey="date"
              axisLine={false}
              tickLine={false}
              tick={labelStyle}
              interval="preserveStartEnd"
              minTickGap={40}
            />
            <YAxis
              domain={["auto", "auto"]}
              axisLine={false}
              tickLine={false}
              tick={labelStyle}
              width={38}
              tickFormatter={v => v.toFixed(0)}
            />
            <Tooltip
              contentStyle={{
                background: "hsl(215,25%,13%)",
                border: `1px solid ${BORDER}`,
                borderRadius: 8,
              }}
              labelStyle={{ color: TEXT_SEC, fontSize: 10 }}
              formatter={(v: number, name: string) => [
                v.toFixed(1),
                name === "rec" ? `${recTicker} (candidate)` : `${holdingTicker} (holding)`,
              ]}
            />
            {/* Baseline — both series start at 100 */}
            <ReferenceLine y={100} stroke={BORDER} strokeDasharray="4 4" />
            {/* Recommended stock — solid blue */}
            <Line
              type="monotone"
              dataKey="rec"
              stroke={MINI_REC_COLOR}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            {/* User's holding — muted dashed */}
            <Line
              type="monotone"
              dataKey="holding"
              stroke={MINI_HOLDING_COLOR}
              strokeWidth={1.5}
              strokeDasharray="4 3"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex gap-5 mt-2">
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0.5 rounded" style={{ background: MINI_REC_COLOR }} />
          <span className="text-xs" style={{ color: TEXT_SEC }}>{recTicker} (candidate)</span>
        </div>
        <div className="flex items-center gap-1.5">
          {/* Dashed line swatch — two segments to mirror the chart dash pattern */}
          <svg width="20" height="2" style={{ display: "block" }}>
            <line x1="0" y1="1" x2="10" y2="1" stroke={MINI_HOLDING_COLOR} strokeWidth="1.5" />
            <line x1="13" y1="1" x2="20" y2="1" stroke={MINI_HOLDING_COLOR} strokeWidth="1.5" />
          </svg>
          <span className="text-xs" style={{ color: TEXT_SEC }}>{holdingTicker} (your holding)</span>
        </div>
      </div>
    </div>
  );
}

// ── SignalRecCard ─────────────────────────────────────────────────────────────
// Expandable card for each signal-connected recommendation.
//
// COLLAPSED: rank badge · ticker (opens StockModal) · company name · sector tag
//            · direction tag · composite score · signal bar
//
// EXPANDED (two-tier):
//   1. MiniPairChart — rec vs rec.related_holdings[0], 3M, read-only
//   2. CompactFundamentals — Market Cap / P/E / 52W High / 52W Low for the rec ticker
//   3. Reasoning text from the backend
//
// Both MiniPairChart and CompactFundamentals only mount when the card is open,
// so there is zero network cost until the user actually expands a card.

function SignalRecCard({
  rec,
  rank,
  companyNames,
  onTickerClick,
}: {
  rec:           Recommendation;
  rank:          number;
  companyNames:  Record<string, string>;
  onTickerClick: (ticker: string) => void;
}) {
  const [open, setOpen] = useState(false);
  // Use the first related holding as the chart counterpart.
  // If a rec has multiple related holdings, [0] is the strongest connection
  // (the backend returns them sorted by relationship strength).
  const primaryHolding = rec.related_holdings[0] ?? null;
  const rankColor = RANK_COLORS[(rank - 1) % RANK_COLORS.length];

  return (
    <div className="rounded-xl overflow-hidden" style={CARD}>

      {/* ── Collapsed header ── */}
      <div
        className="px-5 py-4 cursor-pointer"
        onClick={() => setOpen(o => !o)}
        onMouseEnter={e => (e.currentTarget.style.background = CARD_H)}
        onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
      >
        <div className="flex items-center gap-3">
          {/* Rank badge */}
          <div
            className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold"
            style={{ background: `${rankColor}20`, color: rankColor }}
          >
            {rank}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              {/* Ticker — stop-propagation so clicking it opens the modal
                  without also toggling the accordion */}
              <button
                className="text-sm font-bold hover:underline"
                style={{ color: TEXT_PRI }}
                onClick={e => { e.stopPropagation(); onTickerClick(rec.ticker); }}
              >
                {rec.ticker}
              </button>
              {companyNames[rec.ticker] && (
                <span className="text-xs" style={{ color: TEXT_MUT }}>
                  {companyNames[rec.ticker]}
                </span>
              )}
              <SectorTag sector={rec.sector} />
              <DirectionTag direction={rec.direction} />
            </div>

            {/* Sub-row: score + connected holdings */}
            <div className="flex items-center gap-4 mt-1 flex-wrap">
              <span className="text-xs" style={{ color: TEXT_MUT }}>
                Score{" "}
                <span className="font-semibold" style={{ color: rankColor }}>
                  {rec.composite_score.toFixed(0)}
                </span>
              </span>
              <span className="text-xs" style={{ color: TEXT_MUT }}>
                Connected to{" "}
                <span className="font-semibold" style={{ color: TEXT_SEC }}>
                  {rec.related_holdings.join(", ")}
                </span>
              </span>
            </div>
          </div>

          {/* Signal strength bar — right-aligned, fixed width */}
          <div className="w-28 flex-shrink-0">
            <SignalBar score={rec.signal_score} />
          </div>

          {open
            ? <ChevronUp   className="w-4 h-4 flex-shrink-0" style={{ color: TEXT_MUT }} />
            : <ChevronDown className="w-4 h-4 flex-shrink-0" style={{ color: TEXT_MUT }} />}
        </div>
      </div>

      {/* ── Expanded body ── */}
      {open && (
        <div className="px-5 pb-5" style={{ borderTop: `1px solid ${BORDER_D}` }}>

          {/* 1. Mini pair chart (only shown when a related holding exists) */}
          {primaryHolding ? (
            <div className="mt-4">
              <p className="text-[10px] font-bold uppercase tracking-widest mb-3"
                style={{ color: TEXT_MUT }}>
                3-Month Price Comparison — normalized to 100
              </p>
              <MiniPairChart recTicker={rec.ticker} holdingTicker={primaryHolding} />
            </div>
          ) : (
            <div className="mt-4 px-4 py-3 rounded-lg"
              style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER_D}` }}>
              <p className="text-xs" style={{ color: TEXT_MUT }}>
                No related holding available for price comparison.
              </p>
            </div>
          )}

          {/* Divider */}
          <div className="my-4" style={{ borderTop: `1px solid ${BORDER_D}` }} />

          {/* 2. Compact fundamentals */}
          <div>
            <p className="text-[10px] font-bold uppercase tracking-widest mb-3"
              style={{ color: TEXT_MUT }}>
              Fundamentals — {rec.ticker}
            </p>
            <CompactFundamentals symbol={rec.ticker} />
          </div>

          {/* 3. Reasoning text */}
          {rec.reasoning && (
            <p className="text-xs mt-4 leading-relaxed" style={{ color: TEXT_SEC }}>
              {rec.reasoning}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// ── OverlapCard ───────────────────────────────────────────────────────────────
function OverlapCard({ overlap, companyNames, onTickerClick }: {
  overlap: OverlapResult;
  companyNames: Record<string, string>;
  onTickerClick: (ticker: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const same = overlap.sector_leader === overlap.sector_follower;
  return (
    <div className="rounded-xl overflow-hidden" style={CARD}>
      {/* Header row */}
      <div className="px-5 py-4 cursor-pointer" onClick={() => setOpen(o => !o)}
        onMouseEnter={e => (e.currentTarget.style.background = CARD_H)}
        onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <button className="text-sm font-bold hover:underline"
              style={{ color: TEXT_PRI }}
              onClick={e => { e.stopPropagation(); onTickerClick(overlap.ticker_leader); }}>
              {overlap.ticker_leader}
            </button>
            {companyNames[overlap.ticker_leader] && (
              <span className="text-xs truncate" style={{ color: TEXT_MUT }}>
                {companyNames[overlap.ticker_leader]}
              </span>
            )}
            <ArrowRight className="w-3.5 h-3.5 flex-shrink-0" style={{ color: AMBER }} />
            <button className="text-sm font-bold hover:underline"
              style={{ color: TEXT_PRI }}
              onClick={e => { e.stopPropagation(); onTickerClick(overlap.ticker_follower); }}>
              {overlap.ticker_follower}
            </button>
            {companyNames[overlap.ticker_follower] && (
              <span className="text-xs truncate" style={{ color: TEXT_MUT }}>
                {companyNames[overlap.ticker_follower]}
              </span>
            )}
            <span className="text-xs ml-1 flex-shrink-0" style={{ color: TEXT_MUT }}>
              lag {overlap.best_lag}d
            </span>
            {same && (
              <span className="text-xs px-2 py-0.5 rounded-full flex-shrink-0"
                style={{ background: "hsla(38,92%,50%,0.12)", color: AMBER }}>Same sector</span>
            )}
          </div>
          <div className="w-28 flex-shrink-0"><SignalBar score={overlap.signal_strength} /></div>
          {open
            ? <ChevronUp className="w-4 h-4 flex-shrink-0" style={{ color: TEXT_MUT }} />
            : <ChevronDown className="w-4 h-4 flex-shrink-0" style={{ color: TEXT_MUT }} />}
        </div>
        {/* Sector tags row */}
        <div className="flex items-center gap-2 mt-2">
          <SectorTag sector={overlap.sector_leader} />
          <ArrowRight className="w-3 h-3" style={{ color: TEXT_MUT }} />
          <SectorTag sector={overlap.sector_follower} />
        </div>
      </div>

      {/* Expanded — 6 stats */}
      {open && (
        <div className="px-5 pb-4" style={{ borderTop: `1px solid ${BORDER_D}` }}>
          <p className="text-sm mt-3 leading-relaxed" style={{ color: TEXT_SEC }}>
            {overlap.interpretation}
          </p>
          <div className="grid grid-cols-3 gap-3 mt-4">
            {[
              { label: "dCor",       val: overlap.mean_dcor.toFixed(3),              desc: "Distance correlation at best lag" },
              { label: "OOS Sharpe", val: overlap.oos_sharpe_net.toFixed(2),         desc: "Net out-of-sample Sharpe ratio" },
              { label: "Half-life",  val: `${Math.round(overlap.half_life)}d`,       desc: "Days until signal decays 50%" },
              { label: "Frequency",  val: `${Math.round(overlap.frequency * 100)}%`, desc: "% of 15-yr windows significant" },
              { label: "Sharpness",  val: overlap.sharpness.toFixed(2),              desc: "Signal concentration at one lag" },
              { label: "Best Lag",   val: `${overlap.best_lag}d`,                    desc: "Lead-lag in trading days" },
            ].map(({ label, val, desc }) => (
              <div key={label} className="rounded-lg px-3 py-2 text-center"
                style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER_D}` }}>
                <p className="text-xs mb-0.5" style={{ color: TEXT_MUT }}>{label}</p>
                <p className="text-sm font-semibold" style={{ color: TEXT_PRI }}>{val}</p>
                <p className="text-xs mt-0.5" style={{ color: TEXT_MUT, fontSize: "10px" }}>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── DcorCandidateCard ─────────────────────────────────────────────────────────
// Component 1 output card. Shows stocks that passed the mean dCor filter
// against the user's portfolio, sorted most-independent first.
//
// The key metric is mean_dcor_to_portfolio (lower = more independent).
// We display an "independence score" = (1 − dCor / 0.3) × 100 so that
// higher bar = better, consistent with the rest of the UI.
// Stocks with zero network pairs (dCor = 0) get a perfect score of 100
// and a "No overlap detected" badge.
//
// COLLAPSED: rank badge · ticker · company name · sector tag · dCor badge ·
//            independence bar · pairs sub-text
// EXPANDED:  holding chips (select which portfolio stock to compare) ·
//            MiniPairChart (candidate vs selected holding) ·
//            reasoning · 3 stat chips · CompactFundamentals

function DcorCandidateCard({ candidate, rank, companyNames, onTickerClick, portfolioHoldings }: {
  candidate:         DcorCandidate;
  rank:              number;
  companyNames:      Record<string, string>;
  onTickerClick:     (ticker: string) => void;
  portfolioHoldings: string[];   // all tickers in the user's analyzed portfolio
}) {
  const [open, setOpen] = useState(false);
  const rankColor = RANK_COLORS[(rank - 1) % RANK_COLORS.length];

  // Default to the first paired holding if one exists, otherwise the first portfolio holding.
  const defaultHolding = (() => {
    const paired = portfolioHoldings.find(h => h in candidate.paired_holdings);
    return paired ?? portfolioHoldings[0] ?? null;
  })();
  const [selectedHolding, setSelectedHolding] = useState<string | null>(defaultHolding);

  // Re-derive the default if the card is opened for a new candidate (rank changes)
  // and the previous selectedHolding is no longer in the portfolio list.
  const holdingIsValid = selectedHolding !== null && portfolioHoldings.includes(selectedHolding);

  // Independence score: dCor=0 → 100, dCor=0.3 → 0
  const independenceScore = candidate.n_portfolio_pairs === 0
    ? 100
    : Math.max(0, (1 - candidate.mean_dcor_to_portfolio / 0.3) * 100);
  const scoreColor =
    independenceScore >= 80 ? GREEN :
    independenceScore >= 50 ? BLUE  : AMBER;

  return (
    <div className="rounded-xl overflow-hidden" style={CARD}>

      {/* ── Collapsed header ── */}
      <div className="px-5 py-4 cursor-pointer" onClick={() => setOpen(o => !o)}
        onMouseEnter={e => (e.currentTarget.style.background = CARD_H)}
        onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
        <div className="flex items-center gap-3">

          {/* Rank badge */}
          <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold"
            style={{ background: `${rankColor}20`, color: rankColor }}>
            {rank}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <button className="text-sm font-bold hover:underline"
                style={{ color: TEXT_PRI }}
                onClick={e => { e.stopPropagation(); onTickerClick(candidate.ticker); }}>
                {candidate.ticker}
              </button>
              {companyNames[candidate.ticker] && (
                <span className="text-xs" style={{ color: TEXT_MUT }}>
                  {companyNames[candidate.ticker]}
                </span>
              )}
              <SectorTag sector={candidate.sector} />
              {candidate.n_portfolio_pairs === 0 ? (
                <span className="text-xs px-2 py-0.5 rounded-full"
                  style={{ background: GREEN_DIM, color: GREEN }}>
                  No overlap detected
                </span>
              ) : (
                <span className="text-xs px-2 py-0.5 rounded-full"
                  style={{ background: `${scoreColor}20`, color: scoreColor }}>
                  dCor {candidate.mean_dcor_to_portfolio.toFixed(3)}
                </span>
              )}
            </div>

            {/* Sub-row */}
            <div className="flex items-center gap-4 mt-1.5 flex-wrap">
              <span className="text-xs" style={{ color: TEXT_MUT }}>
                Independence{" "}
                <span className="font-semibold" style={{ color: scoreColor }}>
                  {Math.round(independenceScore)}/100
                </span>
              </span>
              <span className="text-xs" style={{ color: TEXT_MUT }}>
                {candidate.n_portfolio_pairs === 0
                  ? "Not present in any portfolio pair"
                  : `Paired with ${candidate.n_portfolio_pairs} holding${candidate.n_portfolio_pairs !== 1 ? "s" : ""}`}
              </span>
            </div>
          </div>

          {/* Independence bar — wider is more independent */}
          <div className="w-28 flex-shrink-0">
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: BORDER }}>
                <div className="h-full rounded-full"
                  style={{ width: `${independenceScore}%`, background: scoreColor }} />
              </div>
              <span className="text-xs font-semibold w-8 text-right" style={{ color: scoreColor }}>
                {Math.round(independenceScore)}
              </span>
            </div>
          </div>

          {open
            ? <ChevronUp   className="w-4 h-4 flex-shrink-0" style={{ color: TEXT_MUT }} />
            : <ChevronDown className="w-4 h-4 flex-shrink-0" style={{ color: TEXT_MUT }} />}
        </div>
      </div>

      {/* ── Expanded body ── */}
      {open && (
        <div className="px-5 pb-4" style={{ borderTop: `1px solid ${BORDER_D}` }}>

          {/* ── Holding selector + price comparison ── */}
          {portfolioHoldings.length > 0 && (
            <div className="mt-4">
              <p className="text-[10px] font-bold uppercase tracking-widest mb-2"
                style={{ color: TEXT_MUT }}>
                Compare against your holding
              </p>

              {/* Chip row — one chip per portfolio holding */}
              <div className="flex flex-wrap gap-2 mb-3">
                {portfolioHoldings.map(holding => {
                  const pairDcor = candidate.paired_holdings[holding];
                  const hasPair  = pairDcor !== undefined;
                  const isActive = holding === selectedHolding;

                  // Color scheme:
                  //   Active + has pair  → amber (there's a measured relationship)
                  //   Active + no pair   → green (genuinely independent)
                  //   Inactive           → muted border
                  const activeColor = hasPair ? AMBER : GREEN;
                  const chipBg      = isActive
                    ? `${activeColor}20`
                    : "hsl(215,25%,14%)";
                  const chipBorder  = isActive
                    ? `1px solid ${activeColor}60`
                    : `1px solid ${BORDER}`;
                  const chipText    = isActive ? activeColor : TEXT_MUT;

                  return (
                    <button
                      key={holding}
                      onClick={() => setSelectedHolding(holding)}
                      className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold transition-colors"
                      style={{ background: chipBg, border: chipBorder, color: chipText }}
                    >
                      {holding}
                      {/* Pair annotation badge */}
                      <span
                        className="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
                        style={{
                          background: hasPair ? `${AMBER}20` : `${GREEN}20`,
                          color:       hasPair ? AMBER          : GREEN,
                        }}
                      >
                        {hasPair ? `dCor ${pairDcor.toFixed(3)}` : "Independent"}
                      </span>
                    </button>
                  );
                })}
              </div>

              {/* Price comparison chart — loads on demand when expanded */}
              {holdingIsValid && selectedHolding && (
                <div>
                  <MiniPairChart
                    recTicker={candidate.ticker}
                    holdingTicker={selectedHolding}
                  />
                </div>
              )}
            </div>
          )}

          {/* Divider */}
          <div className="my-4" style={{ borderTop: `1px solid ${BORDER_D}` }} />

          {/* Reasoning */}
          {candidate.reasoning && (
            <p className="text-sm leading-relaxed" style={{ color: TEXT_SEC }}>
              {candidate.reasoning}
            </p>
          )}

          {/* 3 stat chips */}
          <div className="grid grid-cols-3 gap-3 mt-4">
            {[
              {
                label: "Mean dCor",
                val:   candidate.n_portfolio_pairs === 0 ? "0.000" : candidate.mean_dcor_to_portfolio.toFixed(3),
                desc:  "Avg distance correlation to your portfolio",
              },
              {
                label: "Portfolio Pairs",
                val:   `${candidate.n_portfolio_pairs}`,
                desc:  "Holdings with a network pair",
              },
              {
                label: "Independence",
                val:   `${Math.round(independenceScore)}/100`,
                desc:  "Lower dCor → higher independence score",
              },
            ].map(({ label, val, desc }) => (
              <div key={label} className="rounded-lg px-3 py-2 text-center"
                style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER_D}` }}>
                <p className="text-xs mb-0.5" style={{ color: TEXT_MUT }}>{label}</p>
                <p className="text-sm font-semibold" style={{ color: TEXT_PRI }}>{val}</p>
                <p style={{ color: TEXT_MUT, fontSize: "10px" }} className="mt-0.5">{desc}</p>
              </div>
            ))}
          </div>

          {/* Divider */}
          <div className="my-4" style={{ borderTop: `1px solid ${BORDER_D}` }} />

          {/* Compact fundamentals */}
          <div>
            <p className="text-[10px] font-bold uppercase tracking-widest mb-3"
              style={{ color: TEXT_MUT }}>
              Fundamentals — {candidate.ticker}
            </p>
            <CompactFundamentals symbol={candidate.ticker} />
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function DiversifyPage() {
  const { savedPortfolios } = useAuth();

  // Merge saved portfolios (user's own) before the built-in presets so they
  // appear first and are visually distinguished with a briefcase icon.
  const allPresets = useMemo(() => [
    ...savedPortfolios.map(p => ({ label: p.name, tickers: p.tickers, saved: true })),
    ...BUILTIN_PRESETS,
  ], [savedPortfolios]);

  const [analysisMode,  setAnalysisMode]  = useState<AnalysisMode>("broad_market");
  const [resultMode,    setResultMode]    = useState<AnalysisMode>("broad_market");
  const [inputVal,      setInputVal]      = useState("");
  const [tickers,       setTickers]       = useState<string[]>([]);
  const [loading,       setLoading]       = useState(false);
  const [error,         setError]         = useState<string | null>(null);
  const [result,        setResult]        = useState<{
    tickers_analyzed:         string[];
    unknown_tickers:          string[];
    overlaps:                 OverlapResult[];
    signal_recommendations:   Recommendation[];
    dcor_filtered_candidates: DcorCandidate[];
    holdings_sectors:         Record<string, string>;
  } | null>(null);
  const [activeSpiderIdx,        setActiveSpiderIdx]        = useState<number | null>(0);
  const [companyNames,           setCompanyNames]           = useState<Record<string, string>>({});
  const [modalStock,             setModalStock]             = useState<Stock | null>(null);
  const [visibleCandidateCount,  setVisibleCandidateCount]  = useState(10);

  const hoveredTicker = activeSpiderIdx !== null
    ? (result?.signal_recommendations[activeSpiderIdx]?.ticker ?? null)
    : null;
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  // currentSectors from holdings_sectors
  const currentSectors = (() => {
    if (!result?.holdings_sectors) return {};
    const dist: Record<string, number> = {};
    Object.values(result.holdings_sectors).forEach(sector => {
      dist[sector] = (dist[sector] ?? 0) + 1;
    });
    return dist;
  })();

  const hoveredRec = result?.signal_recommendations.find(r => r.ticker === hoveredTicker) ?? null;

  const addTicker = (raw: string) => {
    const parts = raw.toUpperCase().split(/[\s,]+/).filter(Boolean);
    setTickers(prev => {
      const next = [...prev];
      for (const t of parts) if (t.length <= 10 && !next.includes(t)) next.push(t);
      return next;
    });
    setInputVal("");
  };

  const removeTicker = (t: string) => setTickers(prev => prev.filter(x => x !== t));

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (["Enter", ",", " ", "Tab"].includes(e.key)) { e.preventDefault(); if (inputVal.trim()) addTicker(inputVal.trim()); }
    if (e.key === "Backspace" && !inputVal && tickers.length) setTickers(prev => prev.slice(0, -1));
  };

  const handleAnalyze = async () => {
    if (inputVal.trim()) addTicker(inputVal.trim());
    const toAnalyze = inputVal.trim()
      ? [...tickers, ...inputVal.trim().toUpperCase().split(/[\s,]+/).filter(Boolean)]
      : tickers;
    if (!toAnalyze.length) return;
    setLoading(true); setError(null); setResult(null); setActiveSpiderIdx(0); setVisibleCandidateCount(10);
    try {
      const data = await analyzePortfolio(toAnalyze, analysisMode);
      setResult(data);
      setResultMode(analysisMode);
      const allTickers = [
        ...data.tickers_analyzed,
        ...data.signal_recommendations.map(r => r.ticker),
        ...(data.dcor_filtered_candidates ?? []).map(r => r.ticker),
      ];
      const uniqueTickers = [...new Set(allTickers)];
      if (uniqueTickers.length > 0) {
        fetchStockSummaries(uniqueTickers)
          .then(summaries => {
            const names: Record<string, string> = {};
            summaries.forEach(s => { names[s.symbol] = s.name; });
            setCompanyNames(names);
          })
          .catch(() => {});
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setTickers([]); setInputVal(""); setResult(null); setError(null);
    setActiveSpiderIdx(null); setCompanyNames({}); setModalStock(null); setVisibleCandidateCount(10);
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  const hasResult = result !== null;

  const handleTickerClick = (ticker: string) => {
    setModalStock({
      symbol:   ticker,
      name:     companyNames[ticker] ?? ticker,
      price:    "—",
      change:   "—",
      volume:   "—",
      positive: true,
    });
  };

  const signalRecs = result?.signal_recommendations ?? [];

  return (
    <div className="min-h-screen" style={{ background: BG }}>
      <Sidebar />
      <main className="pt-14">
        <div className="max-w-5xl mx-auto px-6 py-8">

          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: BLUE_DIM }}>
                <Sparkles className="w-4 h-4" style={{ color: BLUE }} />
              </div>
              <h1 className="text-xl font-bold" style={{ color: TEXT_PRI }}>Portfolio Diversifier</h1>
            </div>
            <p className="text-sm leading-relaxed" style={{ color: TEXT_SEC }}>
              Enter your holdings to uncover hidden lead-lag concentration risk, receive
              signal-backed picks, and find genuinely independent diversification candidates
              filtered by distance correlation.
            </p>
          </div>

          {/* ── Input — always visible ── */}
          <div className="rounded-xl p-5 mb-6" style={CARD}>
            <p className="text-xs font-medium mb-3" style={{ color: TEXT_SEC }}>
              YOUR HOLDINGS — type a ticker and press Enter, Space, or comma
            </p>
            <div className="flex flex-wrap gap-2 min-h-12 p-2 rounded-lg cursor-text"
              style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER}` }}
              onClick={() => inputRef.current?.focus()}>
              {tickers.map(t => (
                <span key={t} className="flex items-center gap-1 px-2.5 py-1 rounded-md text-sm font-semibold"
                  style={{ background: BLUE_DIM, color: BLUE }}>
                  {t}
                  <button onClick={e => { e.stopPropagation(); removeTicker(t); }} className="hover:opacity-60 ml-0.5">
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
              <input ref={inputRef} value={inputVal}
                onChange={e => setInputVal(e.target.value.toUpperCase())}
                onKeyDown={handleKeyDown}
                onBlur={() => { if (inputVal.trim()) addTicker(inputVal.trim()); }}
                placeholder={tickers.length === 0 ? "AAPL, MSFT, NVDA..." : ""}
                className="flex-1 min-w-24 bg-transparent outline-none text-sm"
                style={{ color: TEXT_PRI }} />
            </div>

            {/* Preset buttons */}
            <div className="flex items-center gap-2 mt-3 flex-wrap">
              <span className="text-xs" style={{ color: TEXT_MUT }}>Presets:</span>
              {allPresets.map(p => (
                <button key={p.label} onClick={() => { setTickers(p.tickers); setResult(null); }}
                  className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md transition-colors"
                  style={{
                    background: p.saved ? "hsla(142,71%,45%,0.12)" : "hsl(215,25%,14%)",
                    border:     p.saved ? "1px solid hsla(142,71%,45%,0.3)" : `1px solid ${BORDER}`,
                    color:      p.saved ? GREEN : TEXT_SEC,
                  }}
                  onMouseEnter={e => (e.currentTarget.style.opacity = "0.75")}
                  onMouseLeave={e => (e.currentTarget.style.opacity = "1")}>
                  {p.saved && <Briefcase className="w-3 h-3 flex-shrink-0" />}
                  {p.label}
                </button>
              ))}
            </div>

            {/* Analysis mode toggle */}
            <div className="flex items-center gap-2 mt-4">
              <span className="text-xs" style={{ color: TEXT_MUT }}>Analysis scope:</span>
              <div className="flex items-center p-0.5 rounded-lg"
                style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER}` }}>
                {(["broad_market", "in_sector"] as const).map(mode => (
                  <button
                    key={mode}
                    onClick={() => { setAnalysisMode(mode); setResult(null); }}
                    className="px-3 py-1.5 rounded-md text-xs font-semibold transition-colors"
                    style={analysisMode === mode
                      ? { background: BLUE, color: "white" }
                      : { color: TEXT_SEC, background: "transparent" }}>
                    {mode === "broad_market" ? "Broad Market" : "In-Sector"}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-3 mt-3">
              <button onClick={handleAnalyze}
                disabled={loading || (tickers.length === 0 && !inputVal.trim())}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold disabled:opacity-40"
                style={{ background: BLUE, color: "white" }}>
                {loading
                  ? <><Loader2 className="w-4 h-4 animate-spin" />Analyzing…</>
                  : <><Sparkles className="w-4 h-4" />Analyze Portfolio</>}
              </button>
              {hasResult && (
                <button onClick={reset}
                  className="text-xs px-3 py-2 rounded-lg"
                  style={{ background: "hsl(215,25%,16%)", border: `1px solid ${BORDER}`, color: TEXT_SEC }}
                  onMouseEnter={e => (e.currentTarget.style.background = "hsl(215,25%,20%)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "hsl(215,25%,16%)")}>
                  Clear results
                </button>
              )}
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="rounded-xl px-5 py-4 mb-6 flex items-center gap-3"
              style={{ background: "hsla(0,84%,60%,0.1)", border: "1px solid hsla(0,84%,60%,0.3)" }}>
              <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: RED }} />
              <p className="text-sm" style={{ color: RED }}>{error}</p>
            </div>
          )}

          {/* Results */}
          {hasResult && result && (
            <div>
              {/* Active mode banner */}
              <div className="rounded-xl px-5 py-3 mb-4 flex items-center gap-2"
                style={{
                  background: resultMode === "in_sector" ? "hsla(270,70%,65%,0.08)" : BLUE_DIM,
                  border: `1px solid ${resultMode === "in_sector" ? "hsla(270,70%,65%,0.3)" : "hsla(217,91%,60%,0.3)"}`,
                }}>
                <BarChart3 className="w-4 h-4 flex-shrink-0"
                  style={{ color: resultMode === "in_sector" ? PURPLE : BLUE }} />
                <p className="text-xs" style={{ color: resultMode === "in_sector" ? PURPLE : BLUE }}>
                  {resultMode === "in_sector"
                    ? "In-Sector analysis — pairs are residualized against both market and sector returns, and only stocks within the same sector can form relationships."
                    : "Broad Market analysis — pairs are residualized against market returns across all sectors."}
                </p>
              </div>

              {result.unknown_tickers.length > 0 && (
                <div className="rounded-xl px-5 py-3 mb-4 flex items-center gap-2"
                  style={{ background: "hsla(38,92%,50%,0.1)", border: "1px solid hsla(38,92%,50%,0.3)" }}>
                  <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: AMBER }} />
                  <p className="text-xs" style={{ color: AMBER }}>
                    Not found in our universe: <strong>{result.unknown_tickers.join(", ")}</strong>
                  </p>
                </div>
              )}

              {/* Stat cards */}
              <div className="grid grid-cols-4 gap-3 mb-6">
                {[
                  { label: "Analyzed",            value: result.tickers_analyzed.length,                       color: BLUE,  icon: <BarChart3 className="w-4 h-4" /> },
                  { label: "Overlaps",            value: result.overlaps.length,                               color: result.overlaps.length > 0 ? AMBER : GREEN, icon: <ShieldAlert className="w-4 h-4" /> },
                  { label: "Signal Picks",        value: result.signal_recommendations.length,                 color: BLUE,  icon: <Sparkles className="w-4 h-4" /> },
                  { label: "Diversification Picks", value: (result.dcor_filtered_candidates ?? []).length,     color: GREEN, icon: <Unlink className="w-4 h-4" /> },
                ].map(({ label, value, color, icon }) => (
                  <div key={label} className="rounded-xl p-4" style={CARD}>
                    <div className="flex items-center gap-2 mb-2">
                      <span style={{ color }}>{icon}</span>
                      <p className="text-xs" style={{ color: TEXT_SEC }}>{label}</p>
                    </div>
                    <p className="text-2xl font-bold" style={{ color }}>{value}</p>
                  </div>
                ))}
              </div>

              {/* Factor explanations */}
              <FactorExplanations />

              {/* ── Concentration Risk ── */}
              <section className="mb-8">
                <SectionHeader
                  icon={<ShieldAlert className="w-4 h-4" />}
                  label="Hidden Concentration Risk" count={result.overlaps.length}
                  color={AMBER} dimColor="hsla(38,92%,50%,0.15)"
                  subtitle="Lead-lag relationships detected within your existing holdings — you may be less diversified than you think"
                />
                {result.overlaps.length === 0 ? (
                  <div className="rounded-xl px-6 py-8 text-center" style={CARD}>
                    <TrendingUp className="w-8 h-8 mx-auto mb-3" style={{ color: GREEN }} />
                    <p className="text-sm font-medium mb-1" style={{ color: TEXT_PRI }}>No significant overlaps found</p>
                    <p className="text-xs" style={{ color: TEXT_SEC }}>Your holdings don't show strong lead-lag dependencies.</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {result.overlaps.map((o, i) => (
                      <OverlapCard key={i} overlap={o} companyNames={companyNames} onTickerClick={handleTickerClick} />
                    ))}
                  </div>
                )}
              </section>

              {/* ── Signal-connected recommendations ── */}
              {signalRecs.length > 0 && (
                <section className="mb-8">
                  <SectionHeader
                    icon={<Sparkles className="w-4 h-4" />}
                    label="Signal-Connected Recommendations" count={signalRecs.length}
                    color={BLUE} dimColor={BLUE_DIM}
                    subtitle="Stocks with detected lead-lag relationships to your holdings — use the chart to compare scores, then expand a card below for a price comparison and fundamentals"
                  />

                  {/* Spider chart — overview / score comparison */}
                  <div className="mb-4">
                    <SpiderChart
                      recommendations={signalRecs}
                      activeIdx={activeSpiderIdx}
                      onActiveChange={setActiveSpiderIdx}
                      onTickerClick={handleTickerClick}
                      companyNames={companyNames}
                    />
                  </div>

                  {/* Sector donut — portfolio sector distribution preview */}
                  <div className="mb-6">
                    <SectorDonut
                      currentSectors={currentSectors}
                      previewSector={hoveredRec?.sector ?? null}
                      previewTicker={hoveredRec?.ticker ?? null}
                    />
                  </div>

                  {/* Expandable per-stock cards — drill-down layer.
                      Each card loads MiniPairChart + CompactFundamentals on demand,
                      so there is no network cost until the user opens a card. */}
                  <div className="space-y-2">
                    {signalRecs.map((rec, i) => (
                      <SignalRecCard
                        key={rec.ticker}
                        rec={rec}
                        rank={i + 1}
                        companyNames={companyNames}
                        onTickerClick={handleTickerClick}
                      />
                    ))}
                  </div>
                </section>
              )}

              {/* ── Diversification Candidate Pool (Component 1) ── */}
              {(() => {
                const allCandidates = result.dcor_filtered_candidates ?? [];
                const visibleCandidates = allCandidates.slice(0, visibleCandidateCount);
                const remaining = allCandidates.length - visibleCandidateCount;

                return (
                  <section className="mb-8">
                    <SectionHeader
                      icon={<Unlink className="w-4 h-4" />}
                      label="Diversification Candidate Pool"
                      count={allCandidates.length}
                      color={GREEN} dimColor={GREEN_DIM}
                      subtitle="Top candidates ranked by independence from your portfolio — sorted by lowest mean dCor, then market centrality"
                    />
                    {allCandidates.length > 0 && (
                      <p className="text-xs mb-4 px-1 leading-relaxed" style={{ color: TEXT_MUT }}>
                        These candidates are ranked by how independent they are from your holdings.
                        Distance correlation (dCor) captures both linear and nonlinear dependencies —
                        a lower dCor means genuinely independent behavior. Stocks with detected network
                        pairs are ranked by measured dCor; stocks with no detected pairs are ranked by
                        market centrality so the most network-connected names surface first.
                        Expand a card to see key fundamentals and reasoning.
                      </p>
                    )}
                    {allCandidates.length === 0 ? (
                      <div className="rounded-xl px-6 py-8 text-center" style={CARD}>
                        <p className="text-sm" style={{ color: TEXT_SEC }}>
                          No diversification candidates found.
                        </p>
                      </div>
                    ) : (
                      <>
                        <div className="space-y-2">
                          {visibleCandidates.map((candidate, i) => (
                            <DcorCandidateCard
                              key={candidate.ticker} candidate={candidate} rank={i + 1}
                              companyNames={companyNames} onTickerClick={handleTickerClick}
                              portfolioHoldings={result.tickers_analyzed}
                            />
                          ))}
                        </div>

                        {/* Show more / show less controls */}
                        {allCandidates.length > 10 && (
                          <div className="flex items-center justify-center gap-3 mt-4">
                            {remaining > 0 && (
                              <button
                                onClick={() => setVisibleCandidateCount(c => c + 10)}
                                className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-colors"
                                style={{ background: GREEN_DIM, color: GREEN, border: `1px solid ${GREEN}40` }}
                                onMouseEnter={e => (e.currentTarget.style.background = `${GREEN}25`)}
                                onMouseLeave={e => (e.currentTarget.style.background = GREEN_DIM)}>
                                <ChevronDown className="w-3.5 h-3.5" />
                                Show {Math.min(remaining, 10)} more
                                <span style={{ color: `${GREEN}99` }}>({remaining} remaining)</span>
                              </button>
                            )}
                            {visibleCandidateCount > 10 && (
                              <button
                                onClick={() => setVisibleCandidateCount(10)}
                                className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-colors"
                                style={{ background: "hsl(215,25%,14%)", color: TEXT_SEC, border: `1px solid ${BORDER}` }}
                                onMouseEnter={e => (e.currentTarget.style.background = "hsl(215,25%,18%)")}
                                onMouseLeave={e => (e.currentTarget.style.background = "hsl(215,25%,14%)")}>
                                <ChevronUp className="w-3.5 h-3.5" />
                                Collapse to top 10
                              </button>
                            )}
                          </div>
                        )}
                      </>
                    )}
                  </section>
                );
              })()}
            </div>
          )}

          {/* Empty state */}
          {!hasResult && !loading && tickers.length === 0 && (
            <div className="rounded-xl px-6 py-12 text-center" style={CARD}>
              <div className="w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-4"
                style={{ background: BLUE_DIM }}>
                <Plus className="w-6 h-6" style={{ color: BLUE }} />
              </div>
              <p className="text-sm font-medium mb-1" style={{ color: TEXT_PRI }}>Add your stock holdings above</p>
              <p className="text-xs leading-relaxed max-w-xs mx-auto" style={{ color: TEXT_SEC }}>
                We'll identify hidden lead-lag relationships within your portfolio, surface
                signal-connected picks, and use distance correlation to build a pool of
                genuinely independent diversification candidates.
              </p>
            </div>
          )}

        </div>
      </main>

      {modalStock && (
        <StockModal stock={modalStock} onClose={() => setModalStock(null)} />
      )}
    </div>
  );
}
