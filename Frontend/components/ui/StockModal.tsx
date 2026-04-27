"use client";
import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { X, Star, TrendingUp, TrendingDown, Loader2, AlertTriangle } from "lucide-react";
import { useAuth } from "@/src/app/context/AuthContext";
import {
  fetchOHLCV, fetchStockDetail,
  type OHLCVCandle, type TimeRange, type StockDetail,
} from "@/src/app/lib/api";
import { getCachedOHLCV, setCachedOHLCV } from "@/src/app/lib/stockCache";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Stock {
  symbol:   string;
  name:     string;
  price:    string;
  change:   string;
  volume:   string;
  positive: boolean;
}

// ── Time range config ─────────────────────────────────────────────────────────

type UIRange = "1D" | "5D" | "1W" | "1M" | "6M" | "1Y" | "5Y";

const UI_RANGES: UIRange[] = ["1D", "5D", "1W", "1M", "6M", "1Y", "5Y"];

// Map UI label → backend TimeRange (backend supports: 1D 1W 1M 3M 1Y 5Y)
const TO_API: Record<UIRange, TimeRange> = {
  "1D": "1D",
  "5D": "1W",   // backend returns weekly data; we slice the last 5 points client-side
  "1W": "1W",
  "1M": "1M",
  "6M": "3M",   // backend 3M is the closest available
  "1Y": "1Y",
  "5Y": "5Y",
};

// How many tail candles to show when slicing (null = show all)
const SLICE: Record<UIRange, number | null> = {
  "1D": null,
  "5D": 5,
  "1W": null,
  "1M": null,
  "6M": null,
  "1Y": null,
  "5Y": null,
};

// ── Chart constants ───────────────────────────────────────────────────────────

const CHART_W  = 600;
const CHART_H  = 220;
const PAD_T    = 12;
const PAD_B    = 28;   // room for x-axis labels
const PAD_L    = 8;
const PAD_R    = 8;
const INNER_W  = CHART_W - PAD_L - PAD_R;
const INNER_H  = CHART_H - PAD_T - PAD_B;

// ── Helpers ───────────────────────────────────────────────────────────────────

interface ChartPoint {
  x:      number;
  y:      number;
  candle: OHLCVCandle;
}

function buildPoints(candles: OHLCVCandle[]): ChartPoint[] {
  if (candles.length < 2) return [];
  const closes = candles.map(c => parseFloat(c.close));
  const minP   = Math.min(...closes);
  const maxP   = Math.max(...closes);
  const range  = maxP - minP || 1;
  const step   = INNER_W / (candles.length - 1);
  return candles.map((c, i) => ({
    x:      PAD_L + i * step,
    y:      PAD_T + INNER_H - ((parseFloat(c.close) - minP) / range) * INNER_H,
    candle: c,
  }));
}

function polylineStr(pts: ChartPoint[]): string {
  return pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
}

function fillStr(pts: ChartPoint[]): string {
  if (pts.length < 2) return "";
  const bottom = (PAD_T + INNER_H).toFixed(1);
  return [
    `${pts[0].x.toFixed(1)},${bottom}`,
    ...pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`),
    `${pts[pts.length - 1].x.toFixed(1)},${bottom}`,
  ].join(" ");
}

// Pick evenly-spaced x-axis tick labels (max 6)
function xTicks(pts: ChartPoint[], maxTicks = 6): ChartPoint[] {
  if (pts.length === 0) return [];
  const step = Math.max(1, Math.floor(pts.length / maxTicks));
  const ticks: ChartPoint[] = [];
  for (let i = 0; i < pts.length; i += step) ticks.push(pts[i]);
  // Always include the last point
  if (ticks[ticks.length - 1] !== pts[pts.length - 1]) ticks.push(pts[pts.length - 1]);
  return ticks;
}

// ── Analysis tab helpers ──────────────────────────────────────────────────────

const YEAR_COLORS_MODAL = [
  "hsl(240, 80%, 65%)",
  "hsl(10,  80%, 60%)",
  "hsl(170, 70%, 50%)",
];

const MONTHS_M = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
function monthFromDayIdxM(i: number) {
  return MONTHS_M[Math.min(Math.floor((i / 252) * 12), 11)];
}
function parseYear5YM(label: string): number | null {
  const m = label.match(/'(\d{2})$/);
  return m ? 2000 + parseInt(m[1]) : null;
}
function formatMarketCapM(cap: number | null): string {
  if (cap === null) return "—";
  if (cap >= 1e12) return `$${(cap / 1e12).toFixed(2)}T`;
  if (cap >= 1e9)  return `$${(cap / 1e9).toFixed(1)}B`;
  if (cap >= 1e6)  return `$${(cap / 1e6).toFixed(1)}M`;
  return `$${cap.toLocaleString()}`;
}

const LABEL_STYLE = { fill: "hsl(215, 15%, 55%)", fontSize: 11 } as const;
const BORDER_M    = "hsl(215, 20%, 18%)";
const BORDER_D_M  = "hsl(215, 20%, 16%)";
const TEXT_PRI_M  = "hsl(210, 40%, 92%)";
const TEXT_SEC_M  = "hsl(215, 15%, 55%)";
const TEXT_MUT_M  = "hsl(215, 15%, 40%)";

// ── Year-over-Year chart (inside modal) ───────────────────────────────────────

function YoYChart({ symbol }: { symbol: string }) {
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

  const { years, chartData } = useMemo(() => {
    if (!candles.length) return { years: [], chartData: [] };
    const byYear = new Map<number, OHLCVCandle[]>();
    for (const c of candles) {
      const yr = parseYear5YM(c.date);
      if (yr === null) continue;
      if (!byYear.has(yr)) byYear.set(yr, []);
      byYear.get(yr)!.push(c);
    }
    const sortedYears = [...byYear.keys()].sort().slice(-3);
    const maxLen = Math.max(...sortedYears.map(y => byYear.get(y)!.length));
    const data = Array.from({ length: maxLen }, (_, i) => {
      const point: Record<string, number | string> = { day: i, month: monthFromDayIdxM(i) };
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
    <div className="h-52 flex items-center justify-center">
      <Loader2 className="w-5 h-5 animate-spin" style={{ color: "hsl(217,91%,60%)" }} />
    </div>
  );
  if (error) return (
    <div className="h-52 flex items-center justify-center gap-2">
      <AlertTriangle className="w-4 h-4" style={{ color: "hsl(0,84%,60%)" }} />
      <p className="text-xs" style={{ color: "hsl(0,84%,60%)" }}>{error}</p>
    </div>
  );

  return (
    <div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={BORDER_M} />
            <XAxis dataKey="month" axisLine={false} tickLine={false} tick={LABEL_STYLE} interval={20} />
            <YAxis domain={["auto","auto"]} axisLine={false} tickLine={false} tick={LABEL_STYLE}
              width={44} tickFormatter={v => `${v.toFixed(0)}`} />
            <Tooltip
              contentStyle={{ background: "hsl(215,25%,13%)", border: `1px solid ${BORDER_M}`, borderRadius: 8 }}
              labelStyle={{ color: TEXT_SEC_M, fontSize: 11 }}
              formatter={(v: number, name: string) => [`${v.toFixed(1)}`, name]}
            />
            <ReferenceLine y={100} stroke={BORDER_M} strokeDasharray="4 4" />
            {years.map((yr, i) => (
              <Line key={yr} type="monotone" dataKey={String(yr)}
                stroke={YEAR_COLORS_MODAL[i % YEAR_COLORS_MODAL.length]} strokeWidth={2}
                dot={false} isAnimationActive={false} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-5 mt-2">
        {years.map((yr, i) => (
          <div key={yr} className="flex items-center gap-2">
            <div className="w-5 h-0.5 rounded" style={{ background: YEAR_COLORS_MODAL[i % YEAR_COLORS_MODAL.length] }} />
            <span className="text-xs" style={{ color: TEXT_SEC_M }}>{yr}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Fundamentals panel (inside modal) ────────────────────────────────────────

function FundamentalsGrid({ symbol }: { symbol: string }) {
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

  if (loading) return (
    <div className="h-24 flex items-center justify-center">
      <Loader2 className="w-5 h-5 animate-spin" style={{ color: "hsl(217,91%,60%)" }} />
    </div>
  );

  if (!detail) return (
    <p className="text-xs py-6 text-center" style={{ color: TEXT_MUT_M }}>No fundamentals available.</p>
  );

  const rows = [
    { label: "Sector",     value: detail.sector     ?? "—" },
    { label: "Industry",   value: detail.industry   ?? "—" },
    { label: "Market Cap", value: formatMarketCapM(detail.market_cap) },
    { label: "P/E Ratio",  value: detail.pe_ratio != null ? detail.pe_ratio.toFixed(1) : "—" },
    { label: "52W High",   value: detail.high_52w   != null ? `$${detail.high_52w.toFixed(2)}`  : "—" },
    { label: "52W Low",    value: detail.low_52w    != null ? `$${detail.low_52w.toFixed(2)}`   : "—" },
  ];

  return (
    <div className="grid grid-cols-2 gap-2">
      {rows.map(({ label, value }) => (
        <div key={label} className="flex items-center justify-between px-3 py-2.5 rounded-lg"
          style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER_D_M}` }}>
          <span className="text-xs" style={{ color: TEXT_SEC_M }}>{label}</span>
          <span className="text-xs font-semibold" style={{ color: TEXT_PRI_M }}>{value}</span>
        </div>
      ))}
    </div>
  );
}

// ── Component ─────────────────────────────────────────────────────────────────

interface StockModalProps {
  stock:   Stock;
  onClose: () => void;
}

type ModalTab = "chart" | "analysis";

export default function StockModal({ stock, onClose }: StockModalProps) {
  const { isSaved, toggleSave } = useAuth();
  const saved      = isSaved(stock.symbol);
  const svgRef     = useRef<SVGSVGElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  // Derived from stock prop — kept at component level for use across both tabs
  const positive  = stock.positive;
  const lineColor = positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)";

  const [activeTab, setActiveTab] = useState<ModalTab>("chart");
  const [uiRange,   setUiRange]   = useState<UIRange>("1M");
  const [candles,   setCandles]   = useState<OHLCVCandle[]>([]);
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState<string | null>(null);
  const [hoverPt,   setHoverPt]   = useState<ChartPoint | null>(null);

  // ── Data loading ────────────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    const apiRange = TO_API[uiRange];

    async function load() {
      setLoading(true);
      setError(null);
      setHoverPt(null);

      // 1. Firestore cache first
      const cached = await getCachedOHLCV(stock.symbol, apiRange);
      if (cached && !cancelled) {
        const slice = SLICE[uiRange];
        setCandles(slice ? cached.candles.slice(-slice) : cached.candles);
        setLoading(false);
        if (!cached.stale) return;
      }

      // 2. Backend API fallback
      try {
        const fresh = await fetchOHLCV(stock.symbol, apiRange);
        if (!cancelled) {
          const slice = SLICE[uiRange];
          setCandles(slice ? fresh.slice(-slice) : fresh);
          setLoading(false);
          setError(null);
        }
      } catch (err: unknown) {
        if (!cancelled && !cached) {
          setError(err instanceof Error ? err.message : "Failed to load");
          setLoading(false);
        }
      }
    }

    load();
    return () => { cancelled = true; };
  }, [stock.symbol, uiRange]);

  // ── Keyboard close ──────────────────────────────────────────────────────────
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);

  // ── Chart data ──────────────────────────────────────────────────────────────
  const chartPts  = buildPoints(candles);
  const ticks     = xTicks(chartPts);

  // ── Hover handling ──────────────────────────────────────────────────────────
  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg || chartPts.length < 2) return;
    const rect = svg.getBoundingClientRect();
    const xRaw = ((e.clientX - rect.left) / rect.width) * CHART_W;
    let nearest = chartPts[0];
    let minDist = Infinity;
    for (const pt of chartPts) {
      const d = Math.abs(pt.x - xRaw);
      if (d < minDist) { minDist = d; nearest = pt; }
    }
    setHoverPt(nearest);
  }, [chartPts]);

  // ── Derived display values ──────────────────────────────────────────────────
  const activeCandle = hoverPt?.candle ?? candles[candles.length - 1] ?? null;
  const displayPrice = hoverPt
    ? `$${parseFloat(hoverPt.candle.close).toFixed(2)}`
    : stock.price;
  const displayDate  = hoverPt?.candle.date ?? null;

  // Compute change vs first candle when hovering
  const hoverChange = hoverPt && candles.length > 1
    ? (() => {
        const first = parseFloat(candles[0].close);
        const curr  = parseFloat(hoverPt.candle.close);
        const diff  = curr - first;
        const pct   = (diff / first) * 100;
        const sign  = diff >= 0 ? "+" : "";
        return { text: `${sign}${pct.toFixed(2)}%`, positive: diff >= 0 };
      })()
    : null;

  return (
    <div
      ref={overlayRef}
      onClick={(e) => { if (e.target === overlayRef.current) onClose(); }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(6, 10, 16, 0.88)", backdropFilter: "blur(6px)" }}
    >
      <div
        className="w-full rounded-2xl overflow-hidden"
        style={{
          maxWidth: 680,
          background:  "hsl(215, 25%, 11%)",
          border:      "1px solid hsl(215, 20%, 20%)",
          boxShadow:   "0 32px 80px rgba(0,0,0,0.75)",
        }}
      >
        {/* ── Header ── */}
        <div
          className="flex items-center justify-between px-6 py-4"
          style={{ borderBottom: "1px solid hsl(215, 20%, 16%)" }}
        >
          <div>
            <div className="flex items-baseline gap-2.5 mb-1">
              <span className="text-lg font-bold" style={{ color: "hsl(210, 40%, 92%)" }}>
                {stock.symbol}
              </span>
              <span className="text-sm" style={{ color: "hsl(215, 15%, 50%)" }}>
                {stock.name}
              </span>
            </div>
            <div className="flex items-center gap-2.5">
              <span
                className="text-3xl font-bold"
                style={{ color: "hsl(210, 40%, 96%)", fontVariantNumeric: "tabular-nums" }}
              >
                {displayPrice}
              </span>
              {hoverChange ? (
                <>
                  <span
                    className="text-sm font-semibold"
                    style={{ color: hoverChange.positive ? "hsl(142,71%,45%)" : "hsl(0,84%,60%)" }}
                  >
                    {hoverChange.text}
                  </span>
                  <span className="text-xs" style={{ color: "hsl(215,15%,45%)" }}>
                    {displayDate}
                  </span>
                </>
              ) : (
                <span
                  className="text-sm font-semibold flex items-center gap-1"
                  style={{ color: lineColor }}
                >
                  {positive
                    ? <TrendingUp  className="w-3.5 h-3.5" />
                    : <TrendingDown className="w-3.5 h-3.5" />}
                  {stock.change}
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Tab switcher */}
            <div className="flex items-center p-0.5 rounded-lg mr-2"
              style={{ background: "hsl(215,25%,9%)", border: "1px solid hsl(215,20%,18%)" }}>
              {(["chart", "analysis"] as const).map(tab => (
                <button key={tab} onClick={() => setActiveTab(tab)}
                  className="px-3 py-1 rounded-md text-xs font-semibold"
                  style={activeTab === tab
                    ? { background: "hsl(217,91%,60%)", color: "white" }
                    : { color: "hsl(215,15%,55%)", background: "transparent" }}>
                  {tab === "chart" ? "Price" : "Analysis"}
                </button>
              ))}
            </div>

            <button
              onClick={() => toggleSave({ symbol: stock.symbol, name: stock.name })}
              className="p-2 rounded-lg transition-all hover:opacity-80"
              style={{
                background: saved ? "hsla(48,96%,53%,0.15)" : "hsl(215,25%,16%)",
                border: `1px solid ${saved ? "hsla(48,96%,53%,0.4)" : "hsl(215,20%,22%)"}`,
              }}
              aria-label={saved ? "Unsave stock" : "Save stock"}
            >
              <Star
                className="w-4 h-4"
                style={{
                  fill:  saved ? "hsl(48,96%,53%)" : "transparent",
                  color: saved ? "hsl(48,96%,53%)" : "hsl(215,15%,55%)",
                }}
              />
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:opacity-80 transition-opacity"
              style={{ background: "hsl(215,25%,16%)", border: "1px solid hsl(215,20%,22%)" }}
              aria-label="Close"
            >
              <X className="w-4 h-4" style={{ color: "hsl(215,15%,55%)" }} />
            </button>
          </div>
        </div>

        {/* ── Time range selector (Price tab only) ── */}
        {activeTab === "chart" && (
          <div className="flex items-center gap-1 px-6 pt-4 pb-1">
            {UI_RANGES.map((r) => (
              <button
                key={r}
                onClick={() => setUiRange(r)}
                className="px-3 py-1.5 rounded-md text-xs font-semibold transition-all"
                style={{
                  background: r === uiRange ? "hsl(217, 91%, 60%)"      : "transparent",
                  color:      r === uiRange ? "white"                    : "hsl(215, 15%, 55%)",
                }}
                onMouseEnter={(e) => { if (r !== uiRange) e.currentTarget.style.color = "hsl(210,40%,80%)"; }}
                onMouseLeave={(e) => { if (r !== uiRange) e.currentTarget.style.color = "hsl(215,15%,55%)"; }}
              >
                {r}
              </button>
            ))}
          </div>
        )}

        {/* ── Chart + OHLCV table (Price tab) ── */}
        {activeTab === "chart" && (
          <>
            <div className="px-6 pt-3 pb-1" style={{ cursor: "crosshair" }}>
              {loading ? (
                <div
                  className="flex items-center justify-center"
                  style={{ height: CHART_H }}
                >
                  <Loader2 className="w-5 h-5 animate-spin" style={{ color: "hsl(217,91%,60%)" }} />
                </div>
              ) : error ? (
                <div
                  className="flex flex-col items-center justify-center gap-1"
                  style={{ height: CHART_H }}
                >
                  <span className="text-sm" style={{ color: "hsl(0,84%,60%)" }}>Failed to load chart</span>
                  <span className="text-xs" style={{ color: "hsl(215,15%,45%)" }}>{error}</span>
                </div>
              ) : chartPts.length < 2 ? (
                <div
                  className="flex items-center justify-center text-sm"
                  style={{ height: CHART_H, color: "hsl(215,15%,50%)" }}
                >
                  No data available for this range
                </div>
              ) : (
                <svg
                  ref={svgRef}
                  width="100%"
                  viewBox={`0 0 ${CHART_W} ${CHART_H}`}
                  preserveAspectRatio="none"
                  style={{ display: "block", overflow: "visible" }}
                  onMouseMove={handleMouseMove}
                  onMouseLeave={() => setHoverPt(null)}
                >
                  <defs>
                    <linearGradient id={`fill-${stock.symbol}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%"   stopColor={lineColor} stopOpacity="0.25" />
                      <stop offset="100%" stopColor={lineColor} stopOpacity="0"    />
                    </linearGradient>
                  </defs>

                  {/* Horizontal grid lines */}
                  {[0.25, 0.5, 0.75].map((f) => (
                    <line
                      key={f}
                      x1={PAD_L} y1={PAD_T + INNER_H * f}
                      x2={CHART_W - PAD_R} y2={PAD_T + INNER_H * f}
                      stroke="hsl(215,20%,17%)" strokeWidth="1"
                    />
                  ))}

                  {/* X-axis tick labels */}
                  {ticks.map((pt) => (
                    <text
                      key={pt.candle.date}
                      x={pt.x}
                      y={CHART_H - 4}
                      textAnchor="middle"
                      fontSize="9"
                      fill="hsl(215,15%,38%)"
                      style={{ userSelect: "none" }}
                    >
                      {pt.candle.date}
                    </text>
                  ))}

                  {/* Area fill */}
                  <polygon
                    fill={`url(#fill-${stock.symbol})`}
                    points={fillStr(chartPts)}
                  />

                  {/* Price line */}
                  <polyline
                    fill="none"
                    stroke={lineColor}
                    strokeWidth="1.75"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={polylineStr(chartPts)}
                  />

                  {/* Hover elements */}
                  {hoverPt && (
                    <>
                      {/* Vertical crosshair */}
                      <line
                        x1={hoverPt.x} y1={PAD_T}
                        x2={hoverPt.x} y2={PAD_T + INNER_H}
                        stroke="hsl(215,20%,45%)"
                        strokeWidth="1"
                        strokeDasharray="4 3"
                      />
                      {/* Dot on line */}
                      <circle
                        cx={hoverPt.x} cy={hoverPt.y} r="4.5"
                        fill={lineColor}
                        stroke="hsl(215,25%,11%)"
                        strokeWidth="2"
                      />
                    </>
                  )}
                </svg>
              )}
            </div>

            {/* ── OHLCV hover panel ── */}
            <div
              className="mx-6 mb-5 rounded-xl overflow-hidden"
              style={{ border: "1px solid hsl(215,20%,17%)" }}
            >
              {/* Section label */}
              <div
                className="px-4 py-2 flex items-center justify-between"
                style={{
                  background:   "hsl(215,25%,14%)",
                  borderBottom: "1px solid hsl(215,20%,17%)",
                }}
              >
                <span
                  className="text-xs font-semibold tracking-wider"
                  style={{ color: "hsl(215,15%,48%)" }}
                >
                  OHLCV
                </span>
                {activeCandle && (
                  <span className="text-xs" style={{ color: "hsl(215,15%,45%)" }}>
                    {activeCandle.date}
                  </span>
                )}
              </div>

              {loading ? (
                <div className="flex items-center justify-center py-5">
                  <Loader2 className="w-4 h-4 animate-spin" style={{ color: "hsl(217,91%,60%)" }} />
                </div>
              ) : !activeCandle ? (
                <div className="px-4 py-4 text-xs" style={{ color: "hsl(215,15%,50%)" }}>
                  No data available
                </div>
              ) : (
                <div className="grid grid-cols-5">
                  {(
                    [
                      { label: "Open",   value: `$${parseFloat(activeCandle.open).toFixed(2)}` },
                      { label: "High",   value: `$${parseFloat(activeCandle.high).toFixed(2)}` },
                      { label: "Low",    value: `$${parseFloat(activeCandle.low).toFixed(2)}`  },
                      { label: "Close",  value: `$${parseFloat(activeCandle.close).toFixed(2)}`},
                      { label: "Volume", value: activeCandle.volume                             },
                    ] as { label: string; value: string }[]
                  ).map(({ label, value }, i) => {
                    const isHigh   = label === "High";
                    const isLow    = label === "Low";
                    const valColor = isHigh
                      ? "hsl(142,71%,45%)"
                      : isLow
                      ? "hsl(0,84%,60%)"
                      : "hsl(210,40%,92%)";

                    return (
                      <div
                        key={label}
                        className="px-4 py-3 flex flex-col gap-1.5 transition-colors"
                        style={{
                          borderRight: i < 4 ? "1px solid hsl(215,20%,17%)" : "none",
                          background:  hoverPt ? "hsl(215,25%,12%)" : "transparent",
                        }}
                      >
                        <span className="text-xs" style={{ color: "hsl(215,15%,48%)" }}>
                          {label}
                        </span>
                        <span
                          className="text-sm font-semibold"
                          style={{ color: valColor, fontVariantNumeric: "tabular-nums" }}
                        >
                          {value}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </>
        )}

        {/* ── Analysis tab ── */}
        {activeTab === "analysis" && (
          <div className="px-6 py-5 space-y-5">
            {/* Year-over-year price performance */}
            <div>
              <p className="text-[10px] font-bold uppercase tracking-widest mb-3"
                style={{ color: TEXT_MUT_M }}>
                Year-over-Year Price Performance (normalized to 100)
              </p>
              <YoYChart symbol={stock.symbol} />
            </div>

            {/* Divider */}
            <div style={{ borderTop: `1px solid ${BORDER_D_M}` }} />

            {/* Fundamentals */}
            <div>
              <p className="text-[10px] font-bold uppercase tracking-widest mb-3"
                style={{ color: TEXT_MUT_M }}>
                Fundamentals
              </p>
              <FundamentalsGrid symbol={stock.symbol} />
            </div>
          </div>
        )}

      </div>
    </div>
  );
}