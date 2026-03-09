// Frontend/components/ui/StockModal.tsx
"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { X, Star, TrendingUp, TrendingDown } from "lucide-react";
import { useAuth } from "@/src/app/context/AuthContext";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Stock {
  symbol:   string;
  name:     string;
  price:    string;
  change:   string;
  volume:   string;
  positive: boolean;
}

interface OHLCVData {
  open: string; high: string; low: string; close: string; volume: string;
}

interface ChartPoint {
  x:     number; // 0–480 (SVG units)
  y:     number; // 0–180 (SVG units)
  price: number; // raw price value
  label: string; // x-axis label e.g. "Mar 01"
}

// ─── Constants ────────────────────────────────────────────────────────────────

const TIME_RANGES = ["1D", "5D", "1W", "1M", "6M", "1Y", "5Y"] as const;
type TimeRange = typeof TIME_RANGES[number];

const CHART_W = 480;
const CHART_H = 180;

const mockOHLCV: Record<string, OHLCVData> = {
  AAPL:  { open: "188.42", high: "191.05", low: "187.33", close: "189.84", volume: "52.3M" },
  MSFT:  { open: "427.10", high: "429.80", low: "422.55", close: "425.22", volume: "18.7M" },
  GOOGL: { open: "153.90", high: "156.88", low: "153.12", close: "155.72", volume: "24.1M" },
  AMZN:  { open: "182.50", high: "187.90", low: "181.70", close: "186.13", volume: "31.5M" },
  NVDA:  { open: "862.00", high: "882.14", low: "858.30", close: "878.36", volume: "41.2M" },
  TSLA:  { open: "183.10", high: "184.50", low: "175.22", close: "177.48", volume: "67.8M" },
  META:  { open: "497.30", high: "508.44", low: "495.80", close: "505.95", volume: "14.3M" },
  JPM:   { open: "196.80", high: "199.95", low: "196.10", close: "198.47", volume:  "9.8M" },
  JNJ:   { open: "243.50", high: "244.80", low: "242.10", close: "243.33", volume:  "9.1M" },
  AVGO:  { open: "320.00", high: "335.90", low: "318.50", close: "332.54", volume: "18.3M" },
  SMCI:  { open:  "85.10", high:  "88.90", low:  "84.50", close:  "87.42", volume: "12.1M" },
  INTC:  { open:  "31.20", high:  "31.80", low:  "29.90", close:  "30.12", volume: "45.2M" },
  BA:    { open: "175.40", high: "176.20", low: "171.80", close: "172.55", volume: "11.4M" },
  NFLX:  { open: "621.00", high: "623.50", low: "610.20", close: "612.44", volume:  "5.3M" },
};

// ─── Chart data generator ─────────────────────────────────────────────────────
// Generates deterministic but visually realistic mock price series per symbol+range

function generateChartData(symbol: string, range: TimeRange, positive: boolean): ChartPoint[] {
  const seed    = symbol.split("").reduce((a, c) => a + c.charCodeAt(0), 0);
  const counts: Record<TimeRange, number> = {
    "1D": 78, "5D": 65, "1W": 35, "1M": 30, "6M": 26, "1Y": 52, "5Y": 60,
  };
  const n     = counts[range];
  const step  = CHART_W / (n - 1);
  const baseP = parseFloat(mockOHLCV[symbol]?.close ?? "100");

  // Volatility per range
  const vol: Record<TimeRange, number> = {
    "1D": 0.008, "5D": 0.015, "1W": 0.025, "1M": 0.04,
    "6M": 0.07,  "1Y": 0.12,  "5Y": 0.22,
  };
  const v = vol[range];

  // Generate prices using seeded pseudo-random walk
  const prices: number[] = [baseP];
  for (let i = 1; i < n; i++) {
    const r  = Math.sin(seed * i * 0.7 + i * 1.3) * 0.5 + Math.cos(i * seed * 0.4) * 0.3;
    const drift = positive ? 0.0003 : -0.0003;
    prices.push(Math.max(1, prices[i - 1] * (1 + drift + r * v)));
  }

  const minP = Math.min(...prices);
  const maxP = Math.max(...prices);
  const range_ = maxP - minP || 1;

  // Date labels
  const now  = new Date();
  const labels: string[] = Array.from({ length: n }, (_, i) => {
    const d = new Date(now);
    if (range === "1D") {
      const minutesBack = (n - 1 - i) * (390 / (n - 1));
      d.setMinutes(d.getMinutes() - minutesBack);
      return d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", hour12: true });
    }
    if (range === "5D") { d.setDate(d.getDate() - (n - 1 - i)); return d.toLocaleDateString("en-US", { month: "short", day: "numeric" }); }
    if (range === "1W") { d.setDate(d.getDate() - (n - 1 - i)); return d.toLocaleDateString("en-US", { month: "short", day: "numeric" }); }
    if (range === "1M") { d.setDate(d.getDate() - (n - 1 - i) * 1); return d.toLocaleDateString("en-US", { month: "short", day: "numeric" }); }
    if (range === "6M") { d.setDate(d.getDate() - (n - 1 - i) * 7); return d.toLocaleDateString("en-US", { month: "short", day: "numeric" }); }
    if (range === "1Y") { d.setDate(d.getDate() - (n - 1 - i) * 7); return d.toLocaleDateString("en-US", { month: "short", year: "2-digit" }); }
    d.setDate(d.getDate() - (n - 1 - i) * 30); return d.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
  });

  return prices.map((p, i) => ({
    x:     i * step,
    y:     CHART_H - 8 - ((p - minP) / range_) * (CHART_H - 16),
    price: p,
    label: labels[i],
  }));
}

function pointsString(pts: ChartPoint[]): string {
  return pts.map(p => `${p.x},${p.y}`).join(" ");
}

// ─── Component ────────────────────────────────────────────────────────────────

interface StockModalProps {
  stock:   Stock;
  onClose: () => void;
}

export default function StockModal({ stock, onClose }: StockModalProps) {
  const { isSaved, toggleSave } = useAuth();
  const saved      = isSaved(stock.symbol);
  const ohlcv      = mockOHLCV[stock.symbol] ?? mockOHLCV["AAPL"];
  const overlayRef = useRef<HTMLDivElement>(null);
  const svgRef     = useRef<SVGSVGElement>(null);
  const color      = stock.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)";

  const [range,    setRange]    = useState<TimeRange>("1M");
  const [hovering, setHovering] = useState(false);
  const [hoverPt,  setHoverPt]  = useState<ChartPoint | null>(null);

  const chartData = generateChartData(stock.symbol, range, stock.positive);
  const polyline  = pointsString(chartData);
  const fillPoly  = `0,${CHART_H} ${polyline} ${CHART_W},${CHART_H}`;

  // Close on Escape
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);

  // Mouse move handler — find nearest chart point
  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const svg  = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const xRaw = ((e.clientX - rect.left) / rect.width) * CHART_W;
    // Find nearest point by x
    let nearest = chartData[0];
    let minDist = Math.abs(chartData[0].x - xRaw);
    for (const pt of chartData) {
      const d = Math.abs(pt.x - xRaw);
      if (d < minDist) { minDist = d; nearest = pt; }
    }
    setHoverPt(nearest);
  }, [chartData]);

  const ohlcvRows = [
    { label: "Open",   value: `$${ohlcv.open}`  },
    { label: "High",   value: `$${ohlcv.high}`  },
    { label: "Low",    value: `$${ohlcv.low}`   },
    { label: "Close",  value: `$${ohlcv.close}` },
    { label: "Volume", value: ohlcv.volume       },
  ];

  // Display price — show hovered price or current price
  const displayPrice = hovering && hoverPt
    ? `$${hoverPt.price.toFixed(2)}`
    : stock.price;
  const displayLabel = hovering && hoverPt ? hoverPt.label : null;

  return (
    <div
      ref={overlayRef}
      onClick={(e) => { if (e.target === overlayRef.current) onClose(); }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(8,12,18,0.85)", backdropFilter: "blur(4px)" }}
    >
      <div
        className="w-full max-w-xl rounded-2xl overflow-hidden"
        style={{
          background: "hsl(215, 25%, 11%)",
          border:     "1px solid hsl(215, 20%, 20%)",
          boxShadow:  "0 24px 64px rgba(0,0,0,0.7)",
        }}
      >
        {/* ── Header ── */}
        <div
          className="flex items-center justify-between px-6 py-4"
          style={{ borderBottom: "1px solid hsl(215, 20%, 16%)" }}
        >
          <div>
            <div className="flex items-center gap-2 mb-0.5">
              <span className="text-lg font-bold" style={{ color: "hsl(210, 40%, 92%)" }}>
                {stock.symbol}
              </span>
              <span className="text-sm" style={{ color: "hsl(215, 15%, 50%)" }}>
                {stock.name}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span
                className="text-2xl font-bold transition-all"
                style={{ color: "hsl(210, 40%, 95%)", fontVariantNumeric: "tabular-nums" }}
              >
                {displayPrice}
              </span>
              {displayLabel ? (
                <span className="text-xs" style={{ color: "hsl(215, 15%, 50%)" }}>
                  {displayLabel}
                </span>
              ) : (
                <span className="text-sm font-semibold flex items-center gap-1" style={{ color }}>
                  {stock.positive ? <TrendingUp className="w-3.5 h-3.5" /> : <TrendingDown className="w-3.5 h-3.5" />}
                  {stock.change}
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Star */}
            <button
              onClick={() => toggleSave({ symbol: stock.symbol, name: stock.name })}
              className="p-2 rounded-lg transition-all hover:opacity-80"
              style={{
                background: saved ? "hsla(48,96%,53%,0.15)" : "hsl(215,25%,16%)",
                border:     `1px solid ${saved ? "hsla(48,96%,53%,0.4)" : "hsl(215,20%,22%)"}`,
              }}
              aria-label={saved ? "Remove from saved" : "Save stock"}
            >
              <Star
                className="w-4 h-4"
                style={{
                  fill:  saved ? "hsl(48,96%,53%)" : "transparent",
                  color: saved ? "hsl(48,96%,53%)" : "hsl(215,15%,55%)",
                }}
              />
            </button>

            {/* Close */}
            <button
              onClick={onClose}
              className="p-2 rounded-lg transition-all hover:opacity-80"
              style={{ background: "hsl(215,25%,16%)", border: "1px solid hsl(215,20%,22%)" }}
            >
              <X className="w-4 h-4" style={{ color: "hsl(215,15%,55%)" }} />
            </button>
          </div>
        </div>

        {/* ── Time range selector ── */}
        <div className="flex items-center gap-1 px-6 pt-4">
          {TIME_RANGES.map((r) => (
            <button
              key={r}
              onClick={() => { setRange(r); setHovering(false); setHoverPt(null); }}
              className="px-2.5 py-1 rounded-md text-xs font-semibold transition-all"
              style={{
                background: r === range ? "hsl(217,91%,60%)" : "transparent",
                color:      r === range ? "white" : "hsl(215,15%,55%)",
              }}
            >
              {r}
            </button>
          ))}
        </div>

        {/* ── Chart ── */}
        <div className="px-6 pt-3 pb-1" style={{ cursor: "crosshair" }}>
          <svg
            ref={svgRef}
            width="100%"
            viewBox={`0 0 ${CHART_W} ${CHART_H}`}
            preserveAspectRatio="none"
            style={{ display: "block", overflow: "visible" }}
            onMouseMove={handleMouseMove}
            onMouseEnter={() => setHovering(true)}
            onMouseLeave={() => { setHovering(false); setHoverPt(null); }}
          >
            <defs>
              <linearGradient id={`grad-${stock.symbol}-${range}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor={color} stopOpacity="0.3" />
                <stop offset="100%" stopColor={color} stopOpacity="0"   />
              </linearGradient>
            </defs>

            {/* Grid lines */}
            {[0.2, 0.4, 0.6, 0.8].map((p) => (
              <line key={p} x1="0" y1={CHART_H * p} x2={CHART_W} y2={CHART_H * p}
                stroke="hsl(215,20%,17%)" strokeWidth="1" />
            ))}

            {/* Fill */}
            <polygon fill={`url(#grad-${stock.symbol}-${range})`} points={fillPoly} />

            {/* Line */}
            <polyline
              fill="none" stroke={color} strokeWidth="2"
              strokeLinecap="round" strokeLinejoin="round"
              points={polyline}
            />

            {/* Hover crosshair + dot */}
            {hovering && hoverPt && (
              <>
                {/* Vertical line */}
                <line
                  x1={hoverPt.x} y1={0} x2={hoverPt.x} y2={CHART_H}
                  stroke="hsl(215,20%,40%)" strokeWidth="1" strokeDasharray="4 3"
                />
                {/* Dot */}
                <circle
                  cx={hoverPt.x} cy={hoverPt.y} r="4"
                  fill={color} stroke="hsl(215,25%,11%)" strokeWidth="2"
                />
                {/* Price tooltip box */}
                <g transform={`translate(${Math.min(hoverPt.x + 8, CHART_W - 80)}, ${Math.max(hoverPt.y - 28, 4)})`}>
                  <rect x="0" y="0" width="72" height="20" rx="4"
                    fill="hsl(215,28%,18%)" stroke="hsl(215,20%,28%)" strokeWidth="1" />
                  <text x="36" y="14" textAnchor="middle" fontSize="11"
                    fill="hsl(210,40%,92%)" fontFamily="monospace">
                    ${hoverPt.price.toFixed(2)}
                  </text>
                </g>
              </>
            )}
          </svg>
        </div>

        {/* ── OHLCV ── */}
        <div
          className="mx-6 mb-5 rounded-xl overflow-hidden"
          style={{ border: "1px solid hsl(215,20%,16%)" }}
        >
          <div
            className="px-4 py-2 text-xs font-semibold tracking-wider"
            style={{
              background:   "hsl(215,25%,14%)",
              color:        "hsl(215,15%,50%)",
              borderBottom: "1px solid hsl(215,20%,16%)",
            }}
          >
            TODAY&apos;S OHLCV
          </div>
          <div className="grid grid-cols-5">
            {ohlcvRows.map(({ label, value }, i) => (
              <div
                key={label}
                className="px-3 py-3 flex flex-col gap-1"
                style={{ borderRight: i < 4 ? "1px solid hsl(215,20%,16%)" : "none" }}
              >
                <span className="text-xs" style={{ color: "hsl(215,15%,50%)" }}>{label}</span>
                <span className="text-sm font-semibold" style={{ color: "hsl(210,40%,92%)" }}>{value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}