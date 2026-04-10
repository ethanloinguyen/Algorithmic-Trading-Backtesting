// Frontend/components/ui/SpiderChart.tsx
"use client";
import { Zap, Globe, Link2, Layers, ExternalLink } from "lucide-react";
import type { Recommendation } from "@/src/app/lib/api";

// ── Axes ──────────────────────────────────────────────────────────────────────
const AXES = [
  {
    key:   "signal_score",
    label: "Signal Strength",
    short: "Signal",
    Icon:  Zap,
    desc:  "How statistically robust and economically meaningful the lead-lag relationship is to your holdings.",
  },
  {
    key:   "centrality_score",
    label: "Market Centrality",
    short: "Centrality",
    Icon:  Globe,
    desc:  "How connected this stock is across the entire 2000-stock market network.",
  },
  {
    key:   "coverage_score",
    label: "Portfolio Coverage",
    short: "Coverage",
    Icon:  Link2,
    desc:  "How many of your existing holdings this stock has a detected relationship with.",
  },
  {
    key:   "sector_diversity_score",
    label: "Sector Diversity",
    short: "Diversity",
    Icon:  Layers,
    desc:  "How much new sector exposure this stock adds relative to what you already own.",
  },
] as const;

const N  = AXES.length;
// ── Dimension guide — edit these to resize/reposition the chart ───────────────
// CX, CY   : center of the chart within the SVG viewBox coordinate space
// R         : radius of the outermost tick ring in viewBox units
// LABEL_R   : how far beyond R to place axis labels (1.38 = 38% past the ring)
// VIEWBOX_W : must satisfy: CX + LABEL_R*R + (longest label px) < VIEWBOX_W
//             "Centrality" at fontSize 13 ≈ 75px → need 290 + 221 + 75 = 586 → use 600
//             Increase VIEWBOX_W if right label is still clipped.
// To make chart bigger: increase R and VIEWBOX_W/VIEWBOX_H proportionally.
// To push labels further out: increase LABEL_R and VIEWBOX_W/VIEWBOX_H.
const CX = 290; const CY = 290; const R = 160;
const LABEL_R   = 1.38;
const VIEWBOX_W = 600;
const VIEWBOX_H = 600;
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

function polygonPoints(rec: Recommendation): string {
  return AXES.map((ax, i) => {
    const val = rec[ax.key as keyof Recommendation] as number;
    const pct = Math.min(100, Math.max(0, val)) / 100;
    const [x, y] = axisPoint(i, pct);
    return `${x},${y}`;
  }).join(" ");
}

// ── Direction label ───────────────────────────────────────────────────────────
function DirectionBadge({ direction }: { direction: string }) {
  const GREEN  = "hsl(142,71%,45%)";
  const BLUE   = "hsl(217,91%,60%)";
  const AMBER  = "hsl(38,92%,50%)";
  if (direction === "leads_your_holdings")
    return <span className="text-xs px-2 py-0.5 rounded-full"
      style={{ background: "hsla(142,71%,45%,0.15)", color: GREEN }}>Leads your stocks</span>;
  if (direction === "follows_your_holdings")
    return <span className="text-xs px-2 py-0.5 rounded-full"
      style={{ background: "hsla(217,91%,60%,0.15)", color: BLUE }}>Follows your stocks</span>;
  return <span className="text-xs px-2 py-0.5 rounded-full"
    style={{ background: "hsla(38,92%,50%,0.15)", color: AMBER }}>Bidirectional</span>;
}

// ── Main component ────────────────────────────────────────────────────────────
interface SpiderChartProps {
  recommendations:  Recommendation[];
  activeIdx:        number | null;
  onActiveChange:   (idx: number | null) => void;
  onTickerClick?:   (ticker: string) => void;   // opens OHLCV modal
  companyNames?:    Record<string, string>;      // ticker → company name
}

export default function SpiderChart({ recommendations, activeIdx, onActiveChange, onTickerClick, companyNames = {} }: SpiderChartProps) {
  const recs = recommendations.slice(0, 10);
  // Use parent-controlled activeIdx; setActiveIdx is an alias for onActiveChange
  const setActiveIdx = onActiveChange;

  if (recs.length === 0) return null;

  const activeRec = activeIdx !== null ? recs[activeIdx] : null;
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

      {/* Main body: chart+legend left, detail panel right */}
      <div className="flex gap-6 items-start">

        {/* ── Left column: chart + ranked legend ── */}
        <div className="flex-shrink-0" style={{ width: "400px" }}>
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

            {/* Tick value labels on top axis */}
            {TICK_LEVELS.map(pct => {
              const [x, y] = axisPoint(0, pct / 100);
              return (
                <text key={pct} x={x + 5} y={y - 4}
                  fontSize="9" fill="hsl(215,15%,38%)" textAnchor="start">
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

            {/* Axis labels */}
            {AXES.map((ax, i) => {
              const [x, y] = axisPoint(i, LABEL_R);
              // 0=top, 1=right, 2=bottom, 3=left
              const anchor = i === 1 ? "start" : i === 3 ? "end" : "middle";
              const dy = i === 0 ? -8 : i === 2 ? 18 : 5;
              return (
                <text key={ax.key} x={x} y={y + dy}
                  textAnchor={anchor} fontSize="14" fontWeight="700"
                  fill="hsl(215,15%,78%)">
                  {ax.short}
                </text>
              );
            })}

            {/* Unselected polygons: stroke only, completely transparent fill */}
            {recs.map((rec, i) => {
              if (activeIdx === i) return null; // drawn separately on top
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

            {/* Selected polygon: filled at 80% opacity, bold stroke */}
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
                {/* Vertex dots */}
                {AXES.map((ax, i) => {
                  const val = activeRec[ax.key as keyof Recommendation] as number;
                  const pct = Math.min(100, Math.max(0, val)) / 100;
                  const [x, y] = axisPoint(i, pct);
                  return (
                    <circle key={ax.key} cx={x} cy={y} r="4.5"
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
                  <span className="truncate">{rec.ticker}</span>
                  <span className="ml-auto tabular-nums flex-shrink-0"
                    style={{ color: activeIdx === i ? RANK_COLORS[i] : "hsl(215,15%,38%)", fontSize: "10px" }}>
                    {rec.composite_score.toFixed(0)}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* ── Right panel: factor breakdown ── */}
        <div className="flex-1 min-w-0">
          {activeRec && activeIdx !== null ? (
            <div>
              {/* Stock header */}
              <div className="mb-4 pb-4" style={{ borderBottom: `1px solid ${BORDER_D}` }}>
                <div className="flex items-center gap-2 mb-0.5">
                  {/* Clickable ticker — opens OHLCV modal */}
                  <button
                    onClick={() => onTickerClick?.(activeRec.ticker)}
                    className="flex items-center gap-1.5 group"
                    title="Click to view price chart"
                  >
                    <span className="text-lg font-bold group-hover:underline"
                      style={{ color: activeColor }}>
                      {activeRec.ticker}
                    </span>
                    <ExternalLink className="w-3.5 h-3.5 opacity-50 group-hover:opacity-100 transition-opacity"
                      style={{ color: activeColor }} />
                  </button>
                  <span className="text-xs px-2 py-0.5 rounded-full"
                    style={{ background: "hsl(215,25%,16%)", color: TEXT_SEC }}>
                    #{activeIdx + 1}
                  </span>
                  <span className="text-xs px-2 py-0.5 rounded-full font-semibold"
                    style={{ background: `${activeColor}18`, color: activeColor }}>
                    {activeRec.sector}
                  </span>
                </div>
                {/* Company name */}
                {companyNames[activeRec.ticker] && (
                  <p className="text-xs mb-1" style={{ color: TEXT_SEC }}>
                    {companyNames[activeRec.ticker]}
                  </p>
                )}

                <div className="flex items-center gap-2 flex-wrap mb-2">
                  <DirectionBadge direction={activeRec.direction} />
                </div>

                <p className="text-xs" style={{ color: TEXT_SEC }}>
                  <span style={{ color: TEXT_MUT }}>Connected to: </span>
                  <span style={{ color: TEXT_PRI }}>{activeRec.related_holdings.join(", ")}</span>
                </p>

                <div className="flex items-center gap-3 mt-2">
                  <div className="text-center">
                    <p className="text-xs" style={{ color: TEXT_MUT }}>Composite Score</p>
                    <p className="text-xl font-bold" style={{ color: activeColor }}>
                      {activeRec.composite_score.toFixed(1)}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs" style={{ color: TEXT_MUT }}>Centrality</p>
                    <p className="text-xl font-bold" style={{ color: TEXT_PRI }}>
                      {Math.round(activeRec.centrality * 100)}
                      <span className="text-xs font-normal" style={{ color: TEXT_MUT }}>th</span>
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs" style={{ color: TEXT_MUT }}>Connections</p>
                    <p className="text-xl font-bold" style={{ color: TEXT_PRI }}>
                      {activeRec.n_portfolio_relationships}
                    </p>
                  </div>
                </div>
              </div>

              {/* Four factor bars with descriptions */}
              <div className="space-y-4">
                {AXES.map(ax => {
                  const val = activeRec[ax.key as keyof Recommendation] as number;
                  const barColor = activeColor;
                  return (
                    <div key={ax.key}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-semibold flex items-center gap-1.5"
                          style={{ color: TEXT_PRI }}>
                          <ax.Icon className="w-3.5 h-3.5 flex-shrink-0" style={{ color: barColor }} />
                          {ax.label}
                        </span>
                        <span className="text-sm font-bold tabular-nums"
                          style={{ color: barColor }}>
                          {Math.round(val)}
                          <span className="text-xs font-normal" style={{ color: TEXT_MUT }}>/100</span>
                        </span>
                      </div>
                      {/* Bar */}
                      <div className="h-1.5 rounded-full overflow-hidden mb-1.5"
                        style={{ background: "hsl(215,20%,16%)" }}>
                        <div className="h-full rounded-full"
                          style={{ width: `${val}%`, background: barColor }} />
                      </div>
                      {/* Description */}
                      <p className="text-xs leading-relaxed" style={{ color: TEXT_MUT }}>
                        {ax.desc}
                      </p>
                    </div>
                  );
                })}
              </div>

              {/* Plain-English reasoning */}
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
