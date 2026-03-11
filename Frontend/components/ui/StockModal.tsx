// Frontend/components/ui/StockModal.tsx
"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { X, Star, TrendingUp, TrendingDown, Loader2 } from "lucide-react";
import { useAuth } from "@/src/app/context/AuthContext";
import { fetchOHLCV, type OHLCVCandle, type TimeRange } from "@/src/app/lib/api";
import { getCachedOHLCV, setCachedOHLCV } from "@/src/app/lib/stockCache";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Stock {
  symbol:   string;
  name:     string;
  price:    string;
  change:   string;
  volume:   string;
  positive: boolean;
}

// ── Constants ─────────────────────────────────────────────────────────────────

// Exactly matches backend TimeRange enum
const TIME_RANGES: TimeRange[] = ["1D", "1W", "1M", "3M", "1Y", "5Y"];

const CHART_W = 480;
const CHART_H = 180;

// ── Chart helpers ─────────────────────────────────────────────────────────────

interface ChartPoint {
  x:     number;
  y:     number;
  price: number;
  label: string;
}

function candlesToPoints(candles: OHLCVCandle[]): ChartPoint[] {
  if (candles.length < 2) return [];
  const prices     = candles.map(c => parseFloat(c.close));
  const minP       = Math.min(...prices);
  const maxP       = Math.max(...prices);
  const priceRange = maxP - minP || 1;
  const step       = CHART_W / (candles.length - 1);

  return candles.map((c, i) => ({
    x:     i * step,
    y:     CHART_H - 8 - ((parseFloat(c.close) - minP) / priceRange) * (CHART_H - 16),
    price: parseFloat(c.close),
    label: c.date,
  }));
}

function pts(points: ChartPoint[]): string {
  return points.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
}

// ── Component ─────────────────────────────────────────────────────────────────

interface StockModalProps {
  stock:   Stock;
  onClose: () => void;
}

export default function StockModal({ stock, onClose }: StockModalProps) {
  const { isSaved, toggleSave } = useAuth();
  const saved      = isSaved(stock.symbol);
  const overlayRef = useRef<HTMLDivElement>(null);
  const svgRef     = useRef<SVGSVGElement>(null);
  const color      = stock.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)";

  const [range,    setRange]    = useState<TimeRange>("1M");
  const [candles,  setCandles]  = useState<OHLCVCandle[]>([]);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState<string | null>(null);
  const [hovering, setHovering] = useState(false);
  const [hoverPt,  setHoverPt]  = useState<ChartPoint | null>(null);

  // Load OHLCV: Firestore cache first, then BigQuery
  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      setHovering(false);
      setHoverPt(null);

      // ── Step 1: Try Firestore cache (instant) ──────────────────────────
      const cached = await getCachedOHLCV(stock.symbol, range);
      if (cached && !cancelled) {
        setCandles(cached.candles);
        setLoading(false);
        // If cache is fresh, done
        if (!cached.stale) return;
      }

      // ── Step 2: Fetch from BigQuery via backend ────────────────────────
      try {
        const fresh = await fetchOHLCV(stock.symbol, range);
        if (!cancelled) {
          setCandles(fresh);
          setLoading(false);
          setError(null);
          setCachedOHLCV(stock.symbol, range, fresh); // update cache
        }
      } catch (err: unknown) {
        if (!cancelled) {
          const msg = err instanceof Error ? err.message : "Failed to load";
          if (!cached) {
            setError(msg);
            setLoading(false);
          }
          // If we had cached data, keep showing it silently
        }
      }
    }

    load();
    return () => { cancelled = true; };
  }, [stock.symbol, range]);

  // Close on Escape
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);

  const chartPts = candlesToPoints(candles);
  const polyline = pts(chartPts);
  const fillPoly = chartPts.length >= 2
    ? `0,${CHART_H} ${polyline} ${chartPts[chartPts.length - 1].x.toFixed(1)},${CHART_H}`
    : "";

  const latest = candles[candles.length - 1] ?? null;

  // Hover: find nearest point by SVG x
  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg || chartPts.length < 2) return;
    const rect = svg.getBoundingClientRect();
    const xRaw = ((e.clientX - rect.left) / rect.width) * CHART_W;
    let nearest = chartPts[0];
    let minDist = Math.abs(chartPts[0].x - xRaw);
    for (const pt of chartPts) {
      const d = Math.abs(pt.x - xRaw);
      if (d < minDist) { minDist = d; nearest = pt; }
    }
    setHoverPt(nearest);
  }, [chartPts]);

  const displayPrice = hovering && hoverPt ? `$${hoverPt.price.toFixed(2)}` : stock.price;
  const displayLabel = hovering && hoverPt ? hoverPt.label : null;

  const ohlcvRows = latest ? [
    { label: "Open",   value: `$${latest.open}`  },
    { label: "High",   value: `$${latest.high}`  },
    { label: "Low",    value: `$${latest.low}`   },
    { label: "Close",  value: `$${latest.close}` },
    { label: "Volume", value: latest.volume       },
  ] : [];

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
                className="text-2xl font-bold"
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
                  {stock.positive
                    ? <TrendingUp className="w-3.5 h-3.5" />
                    : <TrendingDown className="w-3.5 h-3.5" />}
                  {stock.change}
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => toggleSave({ symbol: stock.symbol, name: stock.name })}
              className="p-2 rounded-lg transition-all hover:opacity-80"
              style={{
                background: saved ? "hsla(48,96%,53%,0.15)" : "hsl(215,25%,16%)",
                border: `1px solid ${saved ? "hsla(48,96%,53%,0.4)" : "hsl(215,20%,22%)"}`,
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
              onClick={() => setRange(r)}
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
          {loading ? (
            <div className="flex items-center justify-center" style={{ height: CHART_H }}>
              <Loader2 className="w-5 h-5 animate-spin" style={{ color: "hsl(217,91%,60%)" }} />
            </div>
          ) : error ? (
            <div
              className="flex flex-col items-center justify-center gap-1"
              style={{ height: CHART_H }}
            >
              <span className="text-sm" style={{ color: "hsl(0,84%,60%)" }}>
                Failed to load chart
              </span>
              <span className="text-xs" style={{ color: "hsl(215,15%,45%)" }}>
                {error}
              </span>
            </div>
          ) : chartPts.length < 2 ? (
            <div
              className="flex items-center justify-center text-sm"
              style={{ height: CHART_H, color: "hsl(215,15%,50%)" }}
            >
              No data for this range
            </div>
          ) : (
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
                <linearGradient id={`grad-${stock.symbol}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor={color} stopOpacity="0.3" />
                  <stop offset="100%" stopColor={color} stopOpacity="0"   />
                </linearGradient>
              </defs>

              {[0.2, 0.4, 0.6, 0.8].map(p => (
                <line key={p} x1="0" y1={CHART_H * p} x2={CHART_W} y2={CHART_H * p}
                  stroke="hsl(215,20%,17%)" strokeWidth="1" />
              ))}

              <polygon fill={`url(#grad-${stock.symbol})`} points={fillPoly} />

              <polyline
                fill="none" stroke={color} strokeWidth="2"
                strokeLinecap="round" strokeLinejoin="round"
                points={polyline}
              />

              {hovering && hoverPt && (
                <>
                  <line
                    x1={hoverPt.x} y1={0} x2={hoverPt.x} y2={CHART_H}
                    stroke="hsl(215,20%,40%)" strokeWidth="1" strokeDasharray="4 3"
                  />
                  <circle
                    cx={hoverPt.x} cy={hoverPt.y} r="4"
                    fill={color} stroke="hsl(215,25%,11%)" strokeWidth="2"
                  />
                  <g transform={`translate(${Math.min(hoverPt.x + 8, CHART_W - 82)},${Math.max(hoverPt.y - 30, 4)})`}>
                    <rect x="0" y="0" width="74" height="22" rx="4"
                      fill="hsl(215,28%,18%)" stroke="hsl(215,20%,28%)" strokeWidth="1" />
                    <text x="37" y="15" textAnchor="middle" fontSize="11"
                      fill="hsl(210,40%,92%)" fontFamily="monospace">
                      ${hoverPt.price.toFixed(2)}
                    </text>
                  </g>
                </>
              )}
            </svg>
          )}
        </div>

        {/* ── OHLCV table ── */}
        <div
          className="mx-6 mb-5 mt-2 rounded-xl overflow-hidden"
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
            LATEST OHLCV
          </div>
          {loading ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="w-4 h-4 animate-spin" style={{ color: "hsl(217,91%,60%)" }} />
            </div>
          ) : ohlcvRows.length > 0 ? (
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
          ) : (
            <div className="px-4 py-3 text-xs" style={{ color: "hsl(215,15%,50%)" }}>
              No OHLCV data available
            </div>
          )}
        </div>

      </div>
    </div>
  );
}