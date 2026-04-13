// frontend/src/app/dashboard/analysis/page.tsx
"use client";
import { useState, useMemo, useRef, useEffect, useCallback } from "react";
import Sidebar from "@/components/ui/Sidebar";
import { ChevronDown } from "lucide-react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, ReferenceLine, ReferenceDot,
} from "recharts";

// ─── Constants ────────────────────────────────────────────────────────────────

const timeRanges = ["1W", "1M", "3M", "1Y"];

const stockOptions = [
  { symbol: "AAPL", name: "Apple Inc." },
  { symbol: "MSFT", name: "Microsoft Corp." },
  { symbol: "GOOGL", name: "Alphabet Inc." },
  { symbol: "AMZN", name: "Amazon.com Inc." },
  { symbol: "NVDA", name: "NVIDIA Corp." },
  { symbol: "TSLA", name: "Tesla Inc." },
  { symbol: "META", name: "Meta Platforms" },
];

const fundamentals: Record<string, Record<string, string>> = {
  AAPL:  { sector: "Technology",        marketCap: "$2.94T", pe: "29.8",  high52: "$199.62", low52: "$164.08", avgVol: "54.2M",  dividend: "0.51%" },
  MSFT:  { sector: "Technology",        marketCap: "$3.16T", pe: "36.2",  high52: "$468.35", low52: "$362.90", avgVol: "20.1M",  dividend: "0.72%" },
  GOOGL: { sector: "Communication",     marketCap: "$1.94T", pe: "25.4",  high52: "$193.31", low52: "$130.67", avgVol: "25.3M",  dividend: "—"     },
  AMZN:  { sector: "Consumer Discret.", marketCap: "$1.97T", pe: "41.8",  high52: "$201.20", low52: "$151.61", avgVol: "35.0M",  dividend: "—"     },
  NVDA:  { sector: "Technology",        marketCap: "$2.16T", pe: "72.3",  high52: "$974.00", low52: "$505.25", avgVol: "42.8M",  dividend: "0.03%" },
  TSLA:  { sector: "Consumer Discret.", marketCap: "$566B",  pe: "43.1",  high52: "$299.29", low52: "$138.80", avgVol: "98.3M",  dividend: "—"     },
  META:  { sector: "Communication",     marketCap: "$1.29T", pe: "24.7",  high52: "$531.49", low52: "$279.40", avgVol: "15.6M",  dividend: "—"     },
};

const CHART_COLORS = {
  "2020": "hsl(240, 80%, 65%)",
  "2021": "hsl(10, 80%, 60%)",
  "2022": "hsl(170, 70%, 50%)",
};

// ─── Network Graph Data ────────────────────────────────────────────────────────

type NetworkNode = {
  id: string; sector: string; centrality: number; outDegree: number;
  x?: number; y?: number; fx?: number; fy?: number;
};
type NetworkEdge = { source: string | NetworkNode; target: string | NetworkNode; weight: number };

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

// ─── Lag Lab Data ──────────────────────────────────────────────────────────────

const lagScores = [
  { lag: 1, score: 0.08 },
  { lag: 2, score: 0.15 },
  { lag: 3, score: 0.42 },
  { lag: 4, score: 0.22 },
  { lag: 5, score: 0.11 },
];

function generateLagData() {
  return Array.from({ length: 40 }, (_, i) => {
    const base = 100 + Math.sin(i / 2) * 10;
    return {
      t: i,
      rawL: base + (Math.random() - 0.5) * 2,
      rawF: i >= 3 ? 100 + Math.sin((i - 3) / 2) * 10 + (Math.random() - 0.5) * 2 : 100,
      resL: Math.sin(i / 2) + (Math.random() - 0.5) * 0.5,
      resF: i >= 3 ? Math.sin((i - 3) / 2) + (Math.random() - 0.5) * 0.5 : 0,
    };
  });
}

// ─── Chart helpers ────────────────────────────────────────────────────────────

function generatePath(seed: number, points = 12, width = 740, height = 370): string {
  const step = width / (points - 1);
  return Array.from({ length: points }, (_, i) => {
    const y = height * 0.1 + height * 0.8 * (0.5 + 0.45 * Math.sin((i + seed) * 0.9) * Math.cos((i + seed * 1.3) * 0.5));
    return `${i * step},${height - y}`;
  }).join(" ");
}

// ─── Sub-components ───────────────────────────────────────────────────────────

/** Lag Alignment Lab */
function LagAlignmentLab() {
  const [showResiduals, setShowResiduals] = useState(true);
  const [selectedLag, setSelectedLag] = useState(3);
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const data = useMemo(() => generateLagData(), []);

  const leaderKey  = showResiduals ? "resL" : "rawL";
  const followerKey = showResiduals ? "resF" : "rawF";
  const predictionIdx = activeIdx !== null ? activeIdx + selectedLag : null;
  const hasPrediction =
    activeIdx !== null &&
    activeIdx < data.length &&
    predictionIdx !== null &&
    predictionIdx < data.length;

  const panelStyle = {
    background: "hsl(215, 25%, 11%)",
    border: "1px solid hsl(215, 20%, 18%)",
  };
  const labelStyle = { color: "hsl(215, 15%, 45%)", fontSize: 10, fontWeight: "bold" as const };

  return (
    <div className="rounded-xl p-5" style={panelStyle}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-5 pb-4"
        style={{ borderBottom: "1px solid hsl(215, 20%, 18%)" }}>
        <div>
          <h2 className="text-base font-bold" style={{ color: "hsl(210, 40%, 92%)" }}>
            Lead Lag Hypothesis Lab
          </h2>
          <p className="text-xs mt-0.5 uppercase tracking-widest" style={{ color: "hsl(215, 15%, 45%)" }}>
          </p>
        </div>
        <div className="flex items-end gap-4 p-3 rounded-lg"
          style={{ background: "hsl(215, 25%, 14%)", border: "1px solid hsl(215, 20%, 22%)" }}>
          <div className="flex flex-col gap-1.5">
            <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: "hsl(215, 15%, 45%)" }}>
              Test Lag (Days)
            </span>
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5].map((l) => (
                <button
                  key={l}
                  onClick={() => setSelectedLag(l)}
                  className="w-9 h-9 rounded-lg text-sm font-bold transition-all"
                  style={{
                    background: l === selectedLag ? "hsl(217, 91%, 60%)" : "hsl(215, 25%, 17%)",
                    color: l === selectedLag ? "white" : "hsl(210, 40%, 70%)",
                    border: l === selectedLag ? "none" : "1px solid hsl(215, 20%, 24%)",
                  }}
                >
                  {l}
                </button>
              ))}
            </div>
          </div>
          <button
            onClick={() => setShowResiduals(!showResiduals)}
            className="h-9 px-3 text-[10px] font-bold uppercase rounded-lg transition-colors"
            style={{
              background: "hsl(215, 25%, 17%)",
              color: "hsl(210, 40%, 70%)",
              border: "1px solid hsl(215, 20%, 24%)",
            }}
          >
            {showResiduals ? "View Raw" : "View Residuals"}
          </button>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Time Series */}
        <div className="lg:col-span-3 h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              onMouseMove={(e) => {
                if (e?.activeTooltipIndex !== undefined && e.activeTooltipIndex < data.length) {
                  setActiveIdx(e.activeTooltipIndex);
                } else {
                  setActiveIdx(null);
                }
              }}
              onMouseLeave={() => setActiveIdx(null)}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(215, 20%, 18%)" />
              <XAxis dataKey="t" axisLine={false} tickLine={false} tick={labelStyle} minTickGap={20}
                label={{ value: "TIME (DAYS)", position: "insideBottom", offset: -2, ...labelStyle }} height={36} />
              <YAxis domain={["auto", "auto"]} axisLine={false} tickLine={false} tick={labelStyle} />
              <Tooltip content={() => null} />

              <Line type="monotone" dataKey={leaderKey}  stroke="hsl(240, 80%, 65%)" strokeWidth={2.5} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey={followerKey} stroke="hsl(160, 70%, 50%)" strokeWidth={2.5} dot={false} isAnimationActive={false} />

              {activeIdx !== null && activeIdx < data.length && data[activeIdx] !== undefined && (
                <ReferenceLine x={data[activeIdx].t} stroke="hsl(240, 80%, 65%)" strokeDasharray="4 4"
                  label={{ position: "insideTopLeft", value: `Day ${data[activeIdx].t}`, fill: "hsl(240, 80%, 65%)", fontSize: 11, fontWeight: "bold" }} />
              )}
              {hasPrediction && predictionIdx !== null && (
                <>
                  <ReferenceLine x={data[predictionIdx].t} stroke="hsl(160, 70%, 50%)" strokeDasharray="4 4" />
                  <ReferenceDot
                    x={data[predictionIdx].t}
                    y={(data[predictionIdx] as Record<string, number>)[followerKey]}
                    r={5} fill="hsl(160, 70%, 50%)" stroke="hsl(215, 25%, 11%)" strokeWidth={2}
                  />
                  <ReferenceLine
                    segment={[
                      { x: data[activeIdx!].t, y: showResiduals ? 1.8 : 115 },
                      { x: data[predictionIdx].t, y: showResiduals ? 1.8 : 115 },
                    ]}
                    stroke="hsl(215, 20%, 30%)"
                    label={{ position: "top", value: `${selectedLag}d Lag`, fill: "hsl(215, 15%, 55%)", fontSize: 10, fontWeight: "bold" }}
                  />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Sidebar: correlation bar + hint */}
        <div className="flex flex-col gap-4">
          <div className="rounded-lg p-4 flex-1" style={{ background: "hsl(215, 25%, 14%)", border: "1px solid hsl(215, 20%, 22%)" }}>
            <p className="text-[10px] font-bold uppercase tracking-widest text-center mb-3" style={{ color: "hsl(215, 15%, 45%)" }}>
              Correlation Strength
            </p>
            <div className="h-40">
              <ResponsiveContainer>
                <BarChart data={lagScores} layout="vertical" margin={{ left: 8, bottom: 18, right: 8 }}>
                  <XAxis type="number" domain={[0, "auto"]} axisLine={false} tickLine={false} tick={labelStyle}
                    label={{ value: "dCor", position: "insideBottom", offset: -12, ...labelStyle }} />
                  <YAxis dataKey="lag" type="category" width={30} axisLine={false} tickLine={false}
                    tick={{ fill: "hsl(210, 40%, 70%)", fontSize: 11, fontWeight: "bold" }} />
                  <Bar dataKey="score" radius={[0, 4, 4, 0]} onClick={(d) => d && setSelectedLag(d.lag)}>
                    {lagScores.map((entry) => (
                      <Cell key={entry.lag} cursor="pointer"
                        fill={entry.lag === selectedLag ? "hsl(217, 91%, 60%)" : "hsl(215, 20%, 28%)"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="rounded-lg p-4" style={{ background: "hsl(217, 91%, 55% / 0.15)", border: "1px solid hsl(217, 91%, 60% / 0.25)" }}>
            <p className="text-xs leading-relaxed" style={{ color: "hsl(217, 91%, 78%)" }}>
              Can adjust lag to see how follower is affected by leader
            </p>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex gap-5 mt-3">
        {[
          { label: "Leader",   color: "hsl(240, 80%, 65%)" },
          { label: "Follower", color: "hsl(160, 70%, 50%)" },
        ].map(({ label, color }) => (
          <div key={label} className="flex items-center gap-2">
            <div className="w-5 h-0.5 rounded" style={{ background: color }} />
            <span className="text-xs" style={{ color: "hsl(215, 15%, 55%)" }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Lead-Lag Network — canvas-based, no external force-graph lib */
function LeadLagNetwork() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [minSignal, setMinSignal] = useState(0.4);
  const [selectedSector, setSelectedSector] = useState("All");
  const [activeNode, setActiveNode] = useState<NetworkNode>(MOCK_NODES[0]);
  const [hoverNodeId, setHoverNodeId] = useState<string | null>(null);
  const animRef = useRef<number>(0);

  // Simple static layout positions (normalised to canvas coords set in draw)
  const nodePositions: Record<string, { x: number; y: number }> = {
    MSFT: { x: 0.30, y: 0.25 },
    NVDA: { x: 0.70, y: 0.25 },
    AAPL: { x: 0.50, y: 0.50 },
    INTC: { x: 0.15, y: 0.55 },
    CSCO: { x: 0.20, y: 0.78 },
    JPM:  { x: 0.50, y: 0.82 },
    GS:   { x: 0.75, y: 0.68 },
    V:    { x: 0.82, y: 0.45 },
    MA:   { x: 0.85, y: 0.72 },
  };

  const filteredNodes = useMemo(
    () => MOCK_NODES.filter((n) => selectedSector === "All" || n.sector === selectedSector),
    [selectedSector]
  );
  const filteredNodeIds = useMemo(() => new Set(filteredNodes.map((n) => n.id)), [filteredNodes]);

  const filteredEdges = useMemo(
    () =>
      MOCK_EDGES.filter((e) => {
        const s = e.source as string;
        const t = e.target as string;
        return e.weight >= minSignal && filteredNodeIds.has(s) && filteredNodeIds.has(t);
      }),
    [minSignal, filteredNodeIds]
  );

  const activeFollowers = useMemo(
    () =>
      MOCK_EDGES.filter((e) => (e.source as string) === activeNode.id && (e.weight as number) >= minSignal)
        .sort((a, b) => (b.weight as number) - (a.weight as number)),
    [activeNode, minSignal]
  );

  const particleRef = useRef<{ edge: NetworkEdge; t: number }[]>([]);
  useEffect(() => {
    particleRef.current = filteredEdges.map((e) => ({ edge: e, t: Math.random() }));
  }, [filteredEdges]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "hsl(215, 30%, 8%)";
    ctx.fillRect(0, 0, W, H);

    const px = (n: NetworkNode) => nodePositions[n.id].x * W;
    const py = (n: NetworkNode) => nodePositions[n.id].y * H;

    // Draw edges
    filteredEdges.forEach((e) => {
      const sNode = filteredNodes.find((n) => n.id === (e.source as string));
      const tNode = filteredNodes.find((n) => n.id === (e.target as string));
      if (!sNode || !tNode) return;
      const x1 = px(sNode), y1 = py(sNode), x2 = px(tNode), y2 = py(tNode);
      const sId = e.source as string;
      const tId = e.target as string;

      let edgeColor = `rgba(148, 163, 184, ${e.weight * 0.5})`;
      if (hoverNodeId) {
        if (sId === hoverNodeId) edgeColor = "rgba(34, 197, 94, 0.85)";
        else if (tId === hoverNodeId) edgeColor = "rgba(239, 68, 68, 0.85)";
        else edgeColor = "rgba(100, 116, 139, 0.08)";
      }

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = edgeColor;
      ctx.lineWidth = e.weight * 2.5;
      ctx.stroke();

      // Arrow
      const angle = Math.atan2(y2 - y1, x2 - x1);
      const arrowSize = 7;
      const endX = x2 - Math.cos(angle) * 14;
      const endY = y2 - Math.sin(angle) * 14;
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(endX - arrowSize * Math.cos(angle - 0.4), endY - arrowSize * Math.sin(angle - 0.4));
      ctx.lineTo(endX - arrowSize * Math.cos(angle + 0.4), endY - arrowSize * Math.sin(angle + 0.4));
      ctx.closePath();
      ctx.fillStyle = edgeColor;
      ctx.fill();
    });

    // Animate particles
    particleRef.current.forEach((p) => {
      p.t = (p.t + (p.edge.weight as number) * 0.004) % 1;
      const sNode = filteredNodes.find((n) => n.id === (p.edge.source as string));
      const tNode = filteredNodes.find((n) => n.id === (p.edge.target as string));
      if (!sNode || !tNode) return;
      const x = px(sNode) + (px(tNode) - px(sNode)) * p.t;
      const y = py(sNode) + (py(tNode) - py(sNode)) * p.t;
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.fill();
    });

    // Draw nodes
    filteredNodes.forEach((n) => {
      const x = px(n), y = py(n);
      const r = Math.max(8, n.centrality / 12);
      const isHovered = n.id === hoverNodeId;
      const isActive = n.id === activeNode.id;
      const dimmed = hoverNodeId && !isHovered &&
        !filteredEdges.some((e) => {
          const s = e.source as string; const t = e.target as string;
          return (s === hoverNodeId && t === n.id) || (t === hoverNodeId && s === n.id);
        });

      const baseColor = n.sector === "Tech" ? "99, 102, 241" : "16, 185, 129";
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = dimmed ? "rgba(100, 116, 139, 0.2)" : `rgba(${baseColor}, ${isActive ? 1 : 0.85})`;
      ctx.fill();

      if (isActive || isHovered) {
        ctx.beginPath();
        ctx.arc(x, y, r + 4, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${baseColor}, 0.4)`;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      ctx.font = `bold 11px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = dimmed ? "rgba(255,255,255,0.15)" : "white";
      ctx.fillText(n.id, x, y);
    });

    animRef.current = requestAnimationFrame(draw);
  }, [filteredNodes, filteredEdges, hoverNodeId, activeNode]);

  useEffect(() => {
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [draw]);

  const getNodeAtPos = useCallback(
    (mx: number, my: number, canvas: HTMLCanvasElement) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const cx = (mx - rect.left) * scaleX;
      const cy = (my - rect.top) * scaleY;
      return filteredNodes.find((n) => {
        const nx = nodePositions[n.id].x * canvas.width;
        const ny = nodePositions[n.id].y * canvas.height;
        const r = Math.max(8, n.centrality / 12);
        return Math.hypot(cx - nx, cy - ny) <= r + 4;
      }) ?? null;
    },
    [filteredNodes]
  );

  const panelStyle = { background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" };

  return (
    <div className="rounded-xl p-5" style={panelStyle}>
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-5 pb-4"
        style={{ borderBottom: "1px solid hsl(215, 20%, 18%)" }}>
        <div>
          <h2 className="text-base font-bold" style={{ color: "hsl(210, 40%, 92%)" }}>
            Lead-Lag Network Analytics Lab
          </h2>
          <p className="text-xs mt-0.5 uppercase tracking-widest" style={{ color: "hsl(215, 15%, 45%)" }}>
          </p>
        </div>
        <div className="flex gap-4 text-xs font-bold" style={{ color: "hsl(215, 15%, 55%)" }}>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-500" /> Tech
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Finance
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Left sidebar: filters + active node */}
        <div className="space-y-4">
          {/* Filters */}
          <div className="rounded-lg p-4" style={{ background: "hsl(215, 25%, 14%)", border: "1px solid hsl(215, 20%, 22%)" }}>
            <p className="text-[10px] font-bold uppercase tracking-widest mb-3" style={{ color: "hsl(215, 15%, 45%)" }}>
              Network Filters
            </p>
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-semibold mb-1.5" style={{ color: "hsl(210, 40%, 75%)" }}>
                  Sector
                </label>
                <select
                  value={selectedSector}
                  onChange={(e) => setSelectedSector(e.target.value)}
                  className="w-full text-xs rounded-md px-3 py-2"
                  style={{
                    background: "hsl(215, 25%, 11%)",
                    border: "1px solid hsl(215, 20%, 24%)",
                    color: "hsl(210, 40%, 85%)",
                  }}
                >
                  {["All", "Tech", "Finance"].map((s) => (
                    <option key={s}>{s}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold mb-1" style={{ color: "hsl(210, 40%, 75%)" }}>
                  Min Signal: <span style={{ color: "hsl(217, 91%, 70%)" }}>{minSignal.toFixed(2)}</span>
                </label>
                <input
                  type="range" min={0} max={0.95} step={0.05} value={minSignal}
                  onChange={(e) => setMinSignal(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500"
                />
              </div>
            </div>
          </div>

          {/* Active node info */}
          <div className="rounded-lg p-4" style={{ background: "hsl(215, 25%, 14%)", border: "1px solid hsl(215, 20%, 22%)" }}>
            <p className="text-[10px] font-bold uppercase tracking-widest mb-3" style={{ color: "hsl(215, 15%, 45%)" }}>
              Selected Node
            </p>
            <p className="text-base font-bold mb-0.5" style={{ color: "hsl(210, 40%, 92%)" }}>{activeNode.id}</p>
            <p className="text-xs mb-3" style={{ color: "hsl(215, 15%, 50%)" }}>{activeNode.sector}</p>
            {[
              { label: "Centrality", value: activeNode.centrality },
              { label: "Out-Degree", value: activeNode.outDegree },
            ].map(({ label, value }) => (
              <div key={label} className="flex justify-between py-2" style={{ borderBottom: "1px solid hsl(215, 20%, 18%)" }}>
                <span className="text-xs" style={{ color: "hsl(215, 15%, 50%)" }}>{label}</span>
                <span className="text-xs font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>{value}</span>
              </div>
            ))}
            <p className="text-[10px] font-bold uppercase tracking-widest mt-3 mb-2" style={{ color: "hsl(215, 15%, 45%)" }}>
              Leads
            </p>
            {activeFollowers.length === 0 ? (
              <p className="text-xs italic" style={{ color: "hsl(215, 15%, 40%)" }}>None above threshold</p>
            ) : (
              <ul className="space-y-1">
                {activeFollowers.map((e) => (
                  <li key={e.target as string} className="flex justify-between text-xs">
                    <span style={{ color: "hsl(210, 40%, 80%)" }}>{e.target as string}</span>
                    <span className="font-medium px-1.5 py-0.5 rounded text-xs"
                      style={{ background: "hsl(160, 70%, 50% / 0.15)", color: "hsl(160, 70%, 60%)" }}>
                      {(e.weight as number).toFixed(2)} sig
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Canvas graph */}
        <div className="lg:col-span-3 rounded-lg overflow-hidden" style={{ background: "hsl(215, 30%, 8%)", minHeight: 380 }}>
          <canvas
            ref={canvasRef}
            width={800}
            height={420}
            className="w-full h-full"
            style={{ cursor: hoverNodeId ? "pointer" : "default" }}
            onMouseMove={(e) => {
              const node = getNodeAtPos(e.clientX, e.clientY, canvasRef.current!);
              setHoverNodeId(node ? node.id : null);
            }}
            onMouseLeave={() => setHoverNodeId(null)}
            onClick={(e) => {
              const node = getNodeAtPos(e.clientX, e.clientY, canvasRef.current!);
              if (node) setActiveNode(node);
            }}
          />
        </div>
      </div>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function AnalysisPage() {
  const [selectedStock, setSelectedStock] = useState("AAPL");
  const [selectedRange, setSelectedRange] = useState("3M");
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const stock = stockOptions.find((s) => s.symbol === selectedStock)!;
  const info = fundamentals[selectedStock];

  return (
    <div className="min-h-screen" style={{ background: "hsl(213, 27%, 7%)" }}>
      <Sidebar />

      <main className="pt-14">
        <div className="max-w-7xl mx-auto px-6 py-6 space-y-6">

          {/* ── Stock Selector + Time Range ── */}
          <div className="flex items-center gap-4">
            <div className="relative">
              <button
                onClick={() => setDropdownOpen(!dropdownOpen)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all"
                style={{
                  background: "hsl(215, 25%, 14%)",
                  border: "1px solid hsl(215, 20%, 22%)",
                  color: "hsl(210, 40%, 92%)",
                }}
              >
                <span>{stock.symbol} — {stock.name.substring(0, 8)}...</span>
                <ChevronDown className="w-3.5 h-3.5" style={{ color: "hsl(215, 15%, 55%)" }} />
              </button>

              {dropdownOpen && (
                <div
                  className="absolute top-full left-0 mt-1 w-52 rounded-lg overflow-hidden z-20"
                  style={{
                    background: "hsl(215, 25%, 13%)",
                    border: "1px solid hsl(215, 20%, 22%)",
                    boxShadow: "0 8px 24px hsl(213, 27%, 4% / 0.8)",
                  }}
                >
                  {stockOptions.map((s) => (
                    <button
                      key={s.symbol}
                      onClick={() => { setSelectedStock(s.symbol); setDropdownOpen(false); }}
                      className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm transition-colors"
                      style={{
                        background: s.symbol === selectedStock ? "hsl(217, 91%, 60% / 0.15)" : "transparent",
                        color: s.symbol === selectedStock ? "hsl(217, 91%, 70%)" : "hsl(210, 40%, 80%)",
                      }}
                      onMouseEnter={(e) => { if (s.symbol !== selectedStock) e.currentTarget.style.background = "hsl(215, 25%, 17%)"; }}
                      onMouseLeave={(e) => { if (s.symbol !== selectedStock) e.currentTarget.style.background = "transparent"; }}
                    >
                      <span className="font-semibold w-12">{s.symbol}</span>
                      <span style={{ color: "hsl(215, 15%, 55%)" }}>{s.name}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="flex items-center gap-1">
              {timeRanges.map((r) => (
                <button
                  key={r}
                  onClick={() => setSelectedRange(r)}
                  className="px-3 py-1.5 rounded-md text-sm font-medium transition-all"
                  style={{
                    background: r === selectedRange ? "hsl(217, 91%, 60%)" : "transparent",
                    color: r === selectedRange ? "white" : "hsl(215, 15%, 55%)",
                  }}
                >
                  {r}
                </button>
              ))}
            </div>
          </div>

          {/* ── Price Chart + Fundamentals ── */}
          <div className="flex gap-4">
            <div
              className="flex-1 rounded-xl p-5"
              style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
            >
              <svg width="100%" viewBox="0 0 760 400" preserveAspectRatio="xMidYMid meet" style={{ overflow: "visible" }}>
                {[0, 1, 2, 3, 4, 5].map((i) => (
                  <line key={i} x1="40" y1={20 + i * 60} x2="740" y2={20 + i * 60}
                    stroke="hsl(215, 20%, 18%)" strokeWidth="1" />
                ))}
                {[100, 80, 60, 40, 20, 0].map((val, i) => (
                  <text key={val} x="32" y={24 + i * 60} textAnchor="end" fontSize="11" fill="hsl(215, 15%, 45%)">{val}</text>
                ))}
                {Object.entries(CHART_COLORS).map(([year, color], idx) => (
                  <polyline
                    key={year}
                    fill="none"
                    stroke={color}
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={generatePath(idx * 2.5 + stockOptions.findIndex((s) => s.symbol === selectedStock), 13, 700, 300)
                      .split(" ")
                      .map((pt) => {
                        const [x, y] = pt.split(",");
                        return `${parseFloat(x) + 40},${parseFloat(y) + 20}`;
                      })
                      .join(" ")}
                  />
                ))}
              </svg>
              <div className="flex items-center gap-6 mt-3 px-2">
                {Object.entries(CHART_COLORS).map(([year, color]) => (
                  <div key={year} className="flex items-center gap-2">
                    <svg width="20" height="10">
                      <line x1="0" y1="5" x2="12" y2="5" stroke={color} strokeWidth="2" />
                      <circle cx="16" cy="5" r="3" fill="none" stroke={color} strokeWidth="1.5" />
                    </svg>
                    <span className="text-xs" style={{ color: "hsl(215, 15%, 55%)" }}>{year}</span>
                  </div>
                ))}
              </div>
            </div>

            <div
              className="w-64 rounded-xl p-5"
              style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
            >
              <p className="text-lg font-bold mb-0.5" style={{ color: "hsl(210, 40%, 92%)" }}>{selectedStock}</p>
              <p className="text-sm mb-5" style={{ color: "hsl(215, 15%, 50%)" }}>{stock.name}</p>
              {[
                { label: "Sector",     value: info.sector },
                { label: "Market Cap", value: info.marketCap },
                { label: "P/E Ratio",  value: info.pe },
                { label: "52W High",   value: info.high52 },
                { label: "52W Low",    value: info.low52 },
                { label: "Avg Volume", value: info.avgVol },
                { label: "Dividend",   value: info.dividend },
              ].map(({ label, value }) => (
                <div key={label} className="flex items-center justify-between py-3"
                  style={{ borderBottom: "1px solid hsl(215, 20%, 16%)" }}>
                  <span className="text-sm" style={{ color: "hsl(215, 15%, 50%)" }}>{label}</span>
                  <span className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>{value}</span>
                </div>
              ))}
            </div>
          </div>

          <LagAlignmentLab />

          <LeadLagNetwork />

        </div>
      </main>
    </div>
  );
}
