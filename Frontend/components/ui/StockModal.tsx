// frontend/src/components/ui/StockModal.tsx
"use client";
import { useState, useEffect, useRef } from "react";
import { X, TrendingUp, TrendingDown } from "lucide-react";

const TIME_RANGES = ["1D", "1W", "1M", "3M", "1Y", "5Y"];

interface OHLCVRow {
  date: string;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
}

interface StockModalProps {
  stock: {
    symbol: string;
    name: string;
    price: string;
    change: string;
    positive: boolean;
  } | null;
  onClose: () => void;
}

function generateOHLCV(symbol: string, range: string): { ohlcv: OHLCVRow[]; points: { x: number; y: number }[] } {
  const seed = symbol.charCodeAt(0) + symbol.charCodeAt(1);
  const counts: Record<string, number> = { "1D": 24, "1W": 7, "1M": 30, "3M": 90, "1Y": 52, "5Y": 60 };
  const count = counts[range] || 30;
  const labels: string[] = [];
  const now = new Date();

  for (let i = count - 1; i >= 0; i--) {
    const d = new Date(now);
    if (range === "1D") { d.setHours(now.getHours() - i); labels.push(d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })); }
    else if (range === "1W" || range === "1M") { d.setDate(now.getDate() - i); labels.push(d.toLocaleDateString([], { month: "short", day: "numeric" })); }
    else if (range === "3M") { d.setDate(now.getDate() - i); labels.push(d.toLocaleDateString([], { month: "short", day: "numeric" })); }
    else if (range === "1Y") { d.setDate(now.getDate() - i * 7); labels.push(d.toLocaleDateString([], { month: "short", day: "numeric" })); }
    else { d.setMonth(now.getMonth() - i); labels.push(d.toLocaleDateString([], { month: "short", year: "2-digit" })); }
  }

  let price = 100 + (seed % 200);
  const ohlcv: OHLCVRow[] = [];
  const points: { x: number; y: number }[] = [];
  const prices: number[] = [];

  for (let i = 0; i < count; i++) {
    const change = (Math.sin((i + seed) * 0.7) * 3 + Math.cos((i + seed) * 0.3) * 2);
    price = Math.max(10, price + change);
    const open = price - Math.abs(change) * 0.3;
    const high = price + Math.random() * 3;
    const low = price - Math.random() * 3;
    const vol = (5 + Math.abs(Math.sin(i * 0.5)) * 50).toFixed(1) + "M";
    ohlcv.push({ date: labels[i], open: open.toFixed(2), high: high.toFixed(2), low: low.toFixed(2), close: price.toFixed(2), volume: vol });
    prices.push(price);
  }

  const minP = Math.min(...prices);
  const maxP = Math.max(...prices);
  const range2 = maxP - minP || 1;
  points.push(...prices.map((p, i) => ({ x: i / (count - 1), y: 1 - (p - minP) / range2 })));

  return { ohlcv, points };
}

export default function StockModal({ stock, onClose }: StockModalProps) {
  const [range, setRange] = useState("1M");
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  if (!stock) return null;

  const { ohlcv, points } = generateOHLCV(stock.symbol, range);
  const W = 560, H = 200, PAD = 20;

  const svgPoints = points.map(p => `${PAD + p.x * (W - PAD * 2)},${PAD + p.y * (H - PAD * 2)}`).join(" ");
  const areaPath = `M ${PAD},${H - PAD} ` + points.map(p => `L ${PAD + p.x * (W - PAD * 2)},${PAD + p.y * (H - PAD * 2)}`).join(" ") + ` L ${W - PAD},${H - PAD} Z`;

  const color = stock.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)";
  const areaColor = stock.positive ? "hsl(142, 71%, 20%)" : "hsl(0, 84%, 25%)";

  const hovered = hoveredIdx !== null ? ohlcv[hoveredIdx] : ohlcv[ohlcv.length - 1];

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const idx = Math.round(((x - PAD) / (W - PAD * 2)) * (points.length - 1));
    setHoveredIdx(Math.max(0, Math.min(points.length - 1, idx)));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose} style={{ background: "rgba(8,12,20,0.85)", backdropFilter: "blur(4px)" }}>
      <div
        className="relative w-full max-w-2xl rounded-2xl p-6 shadow-2xl"
        style={{ background: "hsl(215, 28%, 10%)", border: "1px solid hsl(215, 20%, 20%)" }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-5">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-bold" style={{ color: "hsl(210, 40%, 95%)" }}>{stock.symbol}</h2>
              <span className="text-sm font-semibold px-2 py-0.5 rounded-md" style={{ background: stock.positive ? "hsl(142, 71%, 20%)" : "hsl(0, 84%, 20%)", color }}>
                {stock.change}
              </span>
            </div>
            <p className="text-sm mt-0.5" style={{ color: "hsl(215, 15%, 55%)" }}>{stock.name}</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <p className="text-2xl font-bold" style={{ color: "hsl(210, 40%, 95%)" }}>{stock.price}</p>
              <div className="flex items-center justify-end gap-1 mt-0.5">
                {stock.positive ? <TrendingUp className="w-3.5 h-3.5" style={{ color }} /> : <TrendingDown className="w-3.5 h-3.5" style={{ color }} />}
                <span className="text-sm font-medium" style={{ color }}>{stock.change}</span>
              </div>
            </div>
            <button onClick={onClose} className="p-1.5 rounded-lg transition-colors hover:opacity-70" style={{ color: "hsl(215, 15%, 55%)" }}>
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Time Range Pills */}
        <div className="flex gap-1 mb-4">
          {TIME_RANGES.map(r => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className="px-3 py-1 rounded-md text-xs font-medium transition-all"
              style={{
                background: r === range ? "hsl(217, 91%, 60%)" : "hsl(215, 25%, 15%)",
                color: r === range ? "white" : "hsl(215, 15%, 55%)",
                border: r === range ? "none" : "1px solid hsl(215, 20%, 20%)",
              }}
            >
              {r}
            </button>
          ))}
        </div>

        {/* Chart */}
        <div className="rounded-xl overflow-hidden mb-4" style={{ background: "hsl(215, 25%, 8%)", border: "1px solid hsl(215, 20%, 16%)" }}>
          {/* Hovered data point info */}
          <div className="px-4 py-2 flex gap-6" style={{ borderBottom: "1px solid hsl(215, 20%, 14%)" }}>
            {[["Date", hovered.date], ["O", `$${hovered.open}`], ["H", `$${hovered.high}`], ["L", `$${hovered.low}`], ["C", `$${hovered.close}`], ["Vol", hovered.volume]].map(([label, val]) => (
              <div key={label}>
                <span className="text-xs" style={{ color: "hsl(215, 15%, 45%)" }}>{label} </span>
                <span className="text-xs font-semibold" style={{ color: "hsl(210, 40%, 88%)" }}>{val}</span>
              </div>
            ))}
          </div>
          <svg
            ref={svgRef}
            viewBox={`0 0 ${W} ${H}`}
            width="100%"
            height="200"
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setHoveredIdx(null)}
            style={{ cursor: "crosshair" }}
          >
            <defs>
              <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={areaColor} stopOpacity="0.8" />
                <stop offset="100%" stopColor={areaColor} stopOpacity="0" />
              </linearGradient>
            </defs>
            {/* Grid lines */}
            {[0.25, 0.5, 0.75].map(f => (
              <line key={f} x1={PAD} y1={PAD + f * (H - PAD * 2)} x2={W - PAD} y2={PAD + f * (H - PAD * 2)} stroke="hsl(215, 20%, 15%)" strokeWidth="1" />
            ))}
            <path d={areaPath} fill="url(#areaGrad)" />
            <polyline fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" points={svgPoints} />
            {hoveredIdx !== null && (() => {
              const p = points[hoveredIdx];
              const cx = PAD + p.x * (W - PAD * 2);
              const cy = PAD + p.y * (H - PAD * 2);
              return (
                <>
                  <line x1={cx} y1={PAD} x2={cx} y2={H - PAD} stroke="hsl(215, 20%, 35%)" strokeWidth="1" strokeDasharray="4,3" />
                  <circle cx={cx} cy={cy} r="4" fill={color} stroke="hsl(215, 28%, 10%)" strokeWidth="2" />
                </>
              );
            })()}
          </svg>
        </div>

        {/* OHLCV Table */}
        <div className="rounded-xl overflow-hidden" style={{ background: "hsl(215, 25%, 8%)", border: "1px solid hsl(215, 20%, 16%)" }}>
          <div className="grid text-xs font-medium px-4 py-2" style={{ gridTemplateColumns: "1.5fr 1fr 1fr 1fr 1fr 1fr", color: "hsl(215, 15%, 45%)", borderBottom: "1px solid hsl(215, 20%, 14%)" }}>
            {["Date", "Open", "High", "Low", "Close", "Volume"].map(h => <span key={h} className={h !== "Date" ? "text-right" : ""}>{h}</span>)}
          </div>
          <div className="overflow-y-auto" style={{ maxHeight: "140px" }}>
            {[...ohlcv].reverse().slice(0, 10).map((row, i) => (
              <div
                key={i}
                className="grid px-4 py-1.5 text-xs"
                style={{ gridTemplateColumns: "1.5fr 1fr 1fr 1fr 1fr 1fr", borderTop: "1px solid hsl(215, 20%, 13%)" }}
              >
                <span style={{ color: "hsl(215, 15%, 60%)" }}>{row.date}</span>
                <span className="text-right" style={{ color: "hsl(210, 40%, 85%)" }}>${row.open}</span>
                <span className="text-right" style={{ color: "hsl(142, 71%, 50%)" }}>${row.high}</span>
                <span className="text-right" style={{ color: "hsl(0, 84%, 60%)" }}>${row.low}</span>
                <span className="text-right font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>${row.close}</span>
                <span className="text-right" style={{ color: "hsl(215, 15%, 55%)" }}>{row.volume}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}