// Frontend/components/ui/SpiderChart.tsx
"use client";
import { TrendingUp, Globe, Layers, Activity, BarChart2 } from "lucide-react";
import type { QualityRecommendation } from "@/src/app/lib/api";

// ── Axes — 5 quality factors ──────────────────────────────────────────────────
const AXES = [
  {
    key:   "momentum_score",
    label: "Momentum",
    short: "Momentum",
    Icon:  TrendingUp,
    desc:  "6-month price return rank-normalised across the filtered universe. Higher = stronger recent trend.",
  },
  {
    key:   "fundamental_quality_score",
    label: "Fundamental Quality",
    short: "Fundamentals",
    Icon:  BarChart2,
    desc:  "Blend of P/E ratio (vs. sector peers) and log market-cap size. Higher = better value + larger cap.",
  },
  {
    key:   "sector_diversity_score",
    label: "Sector Diversity",
    short: "Diversity",
    Icon:  Layers,
    desc:  "How much new sector exposure this stock adds relative to your existing portfolio.",
  },
  {
    key:   "volatility_compatibility_score",
    label: "Volatility Fit",
    short: "Vol Fit",
    Icon:  Activity,
    desc:  "How closely this stock's annualised volatility matches the average volatility of your portfolio.",
  },
  {
    key:   "centrality_score",
    label: "Market Centrality",
    short: "Centrality",
    Icon:  Globe,
    desc:  "Eigenvector centrality in the full 2000-stock lead-lag network — how interconnected this stock is.",
  },
] as const;

const N = AXES.length;

// ── Layout — larger chart to give the spider more room ───────────────────────
// Increase R and VIEWBOX to make the chart itself bigger and more prominent.
// LABEL_R controls how far labels sit outside the outer ring.
const CX       = 320;
const CY       = 320;
const R        = 200;          // was 160 — larger polygon
const LABEL_R  = 1.1;
const VIEWBOX_W = 740;         // was 600
const VIEWBOX_H = 700;         // was 600
const TICK_LEVELS = [25, 50, 75, 100];

export const RANK_COLORS = [
  "hsl(217,91%,60%)",  // 1  blue
  "hsl(142,71%,45%)",  // 2  green
  "hsl(38,92%,50%)",   // 3  amber
  "hsl(280,70%,65%)",  // 4  purple
  "hsl(195,80%,50%)",  // 5  cyan
  "hsl(15,90%,58%)",   // 6  orange
  "hsl(330,70%,60%)",  // 7  pink
  "hsl(90,60%,50%)",   // 8  lime
  "hsl(260,60%,65%)",  // 9  violet
  "hsl(50,80%,55%)",   // 10 yellow
];

// ── Geometry ──────────────────────────────────────────────────────────────────
function axisPoint(axisIdx: number, pct: number): [number, number] {
  const angle = (axisIdx / N) * 2 * Math.PI - Math.PI / 2;
  return [
    CX + pct * R * Math.cos(angle),
    CY + pct * R * Math.sin(angle),
  ];
}

function polygonPoints(rec: QualityRecommendation): string {
  return AXES.map((ax, i) => {
    const val = rec[ax.key as keyof QualityRecommendation] as number;
    const pct = Math.min(100, Math.max(0, val)) / 100;
    const [x, y] = axisPoint(i, pct);
    return `${x},${y}`;
  }).join(" ");
}

// ── Main component ────────────────────────────────────────────────────────────
interface SpiderChartProps {
  recommendations:  QualityRecommendation[];
  activeIdx:        number | null;
  onActiveChange:   (idx: number | null) => void;
  onTickerClick?:   (ticker: string) => void;
  companyNames?:    Record<string, string>;
}

export default function SpiderChart({
  recommendations,
  activeIdx,
  onActiveChange,
  onTickerClick,
  companyNames = {},
}: SpiderChartProps) {
  const recs        = recommendations.slice(0, 10);
  const setActiveIdx = onActiveChange;

  if (recs.length === 0) return null;

  const activeRec   = activeIdx !== null ? recs[activeIdx] : null;
  const activeColor = activeIdx !== null ? RANK_COLORS[activeIdx] : "hsl(215,15%,55%)";

  const TEXT_PRI = "hsl(210,40%,92%)";
  const TEXT_SEC = "hsl(215,15%,55%)";
  const TEXT_MUT = "hsl(215,15%,40%)";
  const BORDER_D = "hsl(215,20%,16%)";

  return (
    <div className="rounded-xl p-6"
      style={{ background: "hsl(215,25%,11%)", border: "1px solid hsl(215,20%,18%)" }}>

      {/* Header */}
      <div className="mb-5">
        <p className="text-sm font-semibold" style={{ color: TEXT_PRI }}>Score Comparison</p>
        <p className="text-xs mt-0.5" style={{ color: TEXT_SEC }}>
          Click a ticker below to highlight its profile — unselected stocks show as outlines only
        </p>
      </div>

      {/* Main body: larger chart+legend left, detail panel right */}
      <div className="flex gap-6 items-start">

        {/* ── Left column: wider to give chart more room ── */}
        <div className="flex-shrink-0" style={{ width: "520px" }}>
          <svg viewBox={`0 0 ${VIEWBOX_W} ${VIEWBOX_H}`} width="100%" height="auto">

            {/* Tick ring polygons */}
            {TICK_LEVELS.map(pct => {
              const pts = Array.from({ length: N }, (_, i) => {
                const [x, y] = axisPoint(i, pct / 100);
                return `${x},${y}`;
              }).join(" ");
              return (
                <polygon key={pct} points={pts} fill="none"
                  stroke="hsl(215,20%,18%)" strokeWidth="1" />
              );
            })}

            {/* Tick value labels on first axis */}
            {TICK_LEVELS.map(pct => {
              const [x, y] = axisPoint(0, pct / 100);
              return (
                <text key={pct} x={x + 5} y={y - 4}
                  fontSize="10" fill="hsl(215,15%,38%)" textAnchor="start">
                  {pct}
                </text>
              );
            })}

            {/* Axis lines */}
            {AXES.map((_, i) => {
              const [x, y] = axisPoint(i, 1);
              return (
                <line key={i} x1={CX} y1={CY} x2={x} y2={y}
                  stroke="hsl(215,20%,22%)" strokeWidth="1" />
              );
            })}

            {/* Axis labels — full words now, sized for readability */}
            {AXES.map((ax, i) => {
              const [x, y] = axisPoint(i, LABEL_R);
              // For 5 axes, determine text anchor by x-position relative to center
              const anchor = x < CX - 15 ? "end" : x > CX + 15 ? "start" : "middle";
              const dy = y < CY - 15 ? -10 : y > CY + 15 ? 20 : 5;
              return (
                <text key={ax.key} x={x} y={y + dy}
                  textAnchor={anchor} fontSize="14" fontWeight="700"
                  fill="hsl(215,15%,78%)">
                  {ax.short}
                </text>
              );
            })}

            {/* Unselected polygons — transparent fill, colored stroke */}
            {recs.map((rec, i) => {
              if (activeIdx === i) return null;
              return (
                <polygon
                  key={rec.ticker}
                  points={polygonPoints(rec)}
                  fill="transparent"
                  stroke={RANK_COLORS[i]}
                  strokeWidth="1.5"
                  strokeOpacity="0.45"
                  className="cursor-pointer transition-all hover:stroke-opacity-80"
                  onClick={() => setActiveIdx(i)}
                />
              );
            })}

            {/* Selected polygon — filled + vertex dots */}
            {activeRec && activeIdx !== null && (
              <>
                <polygon
                  points={polygonPoints(activeRec)}
                  fill={activeColor}
                  fillOpacity="0.18"
                  stroke={activeColor}
                  strokeWidth="2.5"
                  strokeOpacity="0.9"
                />
                {AXES.map((ax, i) => {
                  const val = activeRec[ax.key as keyof QualityRecommendation] as number;
                  const pct = Math.min(100, Math.max(0, val)) / 100;
                  const [x, y] = axisPoint(i, pct);
                  return (
                    <circle key={ax.key} cx={x} cy={y} r="5"
                      fill={activeColor} fillOpacity="0.95" />
                  );
                })}
              </>
            )}

            {/* Center dot */}
            <circle cx={CX} cy={CY} r="3" fill="hsl(215,20%,28%)" />
          </svg>

          {/* ── Ranked legend below the chart ── */}
          <div className="mt-3">
            <p className="text-xs font-medium mb-2" style={{ color: "hsl(215,15%,40%)" }}>
              Ranked by composite score — click to inspect
            </p>
            <div className="grid grid-cols-2 gap-1.5">
              {recs.map((rec, i) => (
                <button
                  key={rec.ticker}
                  onClick={() => setActiveIdx(activeIdx === i ? null : i)}
                  className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs font-semibold transition-all text-left"
                  style={{
                    background: activeIdx === i ? `${RANK_COLORS[i]}20` : "hsl(215,25%,9%)",
                    border:     `1px solid ${activeIdx === i ? RANK_COLORS[i] : "hsl(215,20%,18%)"}`,
                    color:      activeIdx === i ? RANK_COLORS[i] : "hsl(215,15%,55%)",
                  }}
                >
                  <span className="w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 font-bold"
                    style={{ background: `${RANK_COLORS[i]}30`, color: RANK_COLORS[i], fontSize: "10px" }}>
                    {i + 1}
                  </span>
                  <span className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ background: RANK_COLORS[i] }} />
                  {/* Ticker — clicking opens OHLCV modal without toggling selection */}
                  <span
                    className="truncate hover:underline cursor-pointer"
                    onClick={e => { e.stopPropagation(); onTickerClick?.(rec.ticker); }}
                    title={`View ${rec.ticker} price chart`}
                  >
                    {rec.ticker}
                  </span>
                  <span className="ml-auto tabular-nums flex-shrink-0"
                    style={{ color: activeIdx === i ? RANK_COLORS[i] : "hsl(215,15%,38%)", fontSize: "10px" }}>
                    {rec.composite_score.toFixed(0)}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* ── Right panel: detail ── */}
        <div className="flex-1 min-w-0">
          {activeRec && activeIdx !== null ? (
            <div>
              {/* Stock header */}
              <div className="mb-4 pb-4" style={{ borderBottom: `1px solid ${BORDER_D}` }}>
                <div className="flex items-center gap-2 mb-0.5 flex-wrap">
                  <span className="text-lg font-bold" style={{ color: activeColor }}>
                    {activeRec.ticker}
                  </span>
                  <span className="text-xs px-2 py-0.5 rounded-full"
                    style={{ background: "hsl(215,25%,16%)", color: TEXT_SEC }}>
                    #{activeIdx + 1}
                  </span>
                  <span className="text-xs px-2 py-0.5 rounded-full font-semibold"
                    style={{ background: `${activeColor}18`, color: activeColor }}>
                    {activeRec.sector}
                  </span>
                </div>

                {companyNames[activeRec.ticker] && (
                  <p className="text-xs mb-1" style={{ color: TEXT_SEC }}>
                    {companyNames[activeRec.ticker]}
                  </p>
                )}

                {/* Explicit "View Price Chart" button — always visible */}
                <button
                  onClick={() => onTickerClick?.(activeRec.ticker)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold mb-3 transition-all hover:opacity-80"
                  style={{
                    background: `${activeColor}18`,
                    border:     `1px solid ${activeColor}50`,
                    color:      activeColor,
                  }}
                >
                  <BarChart2 className="w-3.5 h-3.5" />
                  View Price Chart
                </button>

                {/* Summary stat pills */}
                <div className="flex items-center gap-3 mt-3 flex-wrap">
                  <div className="text-center">
                    <p className="text-xs" style={{ color: TEXT_MUT }}>Composite</p>
                    <p className="text-xl font-bold" style={{ color: activeColor }}>
                      {activeRec.composite_score.toFixed(1)}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs" style={{ color: TEXT_MUT }}>Momentum</p>
                    <p className="text-xl font-bold" style={{ color: TEXT_PRI }}>
                      {Math.round(activeRec.momentum_score)}
                      <span className="text-xs font-normal" style={{ color: TEXT_MUT }}>/100</span>
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs" style={{ color: TEXT_MUT }}>Fundamentals</p>
                    <p className="text-xl font-bold" style={{ color: TEXT_PRI }}>
                      {Math.round(activeRec.fundamental_quality_score)}
                      <span className="text-xs font-normal" style={{ color: TEXT_MUT }}>/100</span>
                    </p>
                  </div>
                </div>
              </div>

              {/* Five factor bars with descriptions */}
              <div className="space-y-4">
                {AXES.map(ax => {
                  const val = activeRec[ax.key as keyof QualityRecommendation] as number;
                  return (
                    <div key={ax.key}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-semibold flex items-center gap-1.5"
                          style={{ color: TEXT_PRI }}>
                          <ax.Icon className="w-3.5 h-3.5 flex-shrink-0" style={{ color: activeColor }} />
                          {ax.label}
                        </span>
                        <span className="text-sm font-bold tabular-nums"
                          style={{ color: activeColor }}>
                          {Math.round(val)}
                          <span className="text-xs font-normal" style={{ color: TEXT_MUT }}>/100</span>
                        </span>
                      </div>
                      <div className="h-1.5 rounded-full overflow-hidden mb-1.5"
                        style={{ background: "hsl(215,20%,16%)" }}>
                        <div className="h-full rounded-full"
                          style={{ width: `${val}%`, background: activeColor }} />
                      </div>
                      <p className="text-xs leading-relaxed" style={{ color: TEXT_MUT }}>
                        {ax.desc}
                      </p>
                    </div>
                  );
                })}
              </div>

              {/* Reasoning */}
              <div className="mt-4 pt-4 rounded-lg px-3 py-3"
                style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER_D}` }}>
                <p className="text-xs leading-relaxed" style={{ color: TEXT_SEC }}>
                  {activeRec.reasoning}
                </p>
              </div>
            </div>
          ) : (
            /* Empty state */
            <div className="flex flex-col items-center justify-center py-10 text-center">
              <div className="w-10 h-10 rounded-full flex items-center justify-center mb-3"
                style={{ background: "hsl(215,25%,16%)" }}>
                <span className="text-lg">👆</span>
              </div>
              <p className="text-sm font-medium mb-1" style={{ color: TEXT_SEC }}>
                No stock selected
              </p>
              <p className="text-xs max-w-[180px]" style={{ color: TEXT_MUT }}>
                Click a ticker or shape on the chart to see its full profile
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
