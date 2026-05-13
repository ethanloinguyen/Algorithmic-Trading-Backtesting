// frontend/src/components/analysis/MonteCarloLab.tsx
//
// Drop this component into analysis/page.tsx — renders below LeadLagNetwork.
//
// Usage in page.tsx:
//   import MonteCarloLab from "@/components/analysis/MonteCarloLab";
//   ...
//   {/* ── Monte Carlo Lab ── */}
//   <MonteCarloLab stocks={stocks} stocksLoading={stocksLoading} />
//
// API calls expected (add to lib/api.ts — see bottom of this file):
//   fetchMonteCarlo(symbol: string): Promise<MonteCarloResult>
//   fetchPairMonteCarlo(leader: string, follower: string): Promise<PairMonteCarloResult>

"use client";
import { useState, useCallback, useMemo } from "react";
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";
import { Loader2, AlertTriangle, FlaskConical, TrendingUp, ArrowRight } from "lucide-react";
import type { StockSummary } from "@/src/app/lib/api";

// ─── Design tokens (match analysis/page.tsx) ──────────────────────────────────
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

// Cone colors — match the Python output visually
const CONE_FILL    = "hsla(217, 91%, 60%, 0.18)";   // 1σ band
const CONE_FILL_2  = "hsla(217, 91%, 60%, 0.08)";   // 2σ band
const MEDIAN_CLR   = "hsl(217, 91%, 60%)";           // MC median line
const ACTUAL_CLR   = "hsl(0, 84%, 60%)";             // actual price line
const TRAIN_CLR    = "hsl(215, 15%, 55%)";           // training period line

const labelStyle = { fill: TEXT_SEC, fontSize: 11 } as const;

// ─── Types ────────────────────────────────────────────────────────────────────

export interface BandPoint {
  day: number;         // trading day index (0 = split date)
  date?: string;
  p5:   number;
  p16:  number;
  p50:  number;        // median
  p84:  number;
  p95:  number;
  actual?: number;     // real price (null for future days)
}

export interface TrainPoint {
  day: number;
  date?: string;
  price: number;
}

export interface MonteCarloResult {
  symbol:       string;
  train:        TrainPoint[];   // historical prices (training period)
  bands:        BandPoint[];    // percentile bands (test / forecast period)
  sigma_annual: number;         // annualised vol used
  mu_annual:    number;         // annualised drift used
  n_sims:       number;
  coverage_1s:  number;         // fraction of actual prices inside 1σ band
}

export interface PairMonteCarloResult {
  leader:   string;
  follower: string;
  lag:      number;
  beta:     number;             // OLS slope
  pearson:  number;             // Pearson ρ on training data
  with_ll:  MonteCarloResult;   // WITH lead-lag
  without:  MonteCarloResult;   // WITHOUT (pure GBM)
}

// ─── API stubs — replace with real calls from lib/api.ts ─────────────────────
// These are thin wrappers so the component compiles independently.
// Move the actual implementations into lib/api.ts.

async function fetchMonteCarlo(symbol: string): Promise<MonteCarloResult> {
  const res = await fetch(`/api/montecarlo/single?symbol=${encodeURIComponent(symbol)}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function fetchPairMonteCarlo(
  leader: string, follower: string
): Promise<PairMonteCarloResult> {
  const res = await fetch(
    `/api/montecarlo/pair?leader=${encodeURIComponent(leader)}&follower=${encodeURIComponent(follower)}`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function pct(v: number) { return `${(v * 100).toFixed(1)}%`; }

function StatPill({
  label, value, color,
}: { label: string; value: string; color?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: TEXT_MUT }}>
        {label}
      </span>
      <span className="text-sm font-bold" style={{ color: color ?? TEXT_PRI }}>
        {value}
      </span>
    </div>
  );
}

// ─── Stock Searchable Dropdown (self-contained, no external dep) ───────────────

function MCStockPicker({
  label, value, onChange, stocks, color,
}: {
  label: string;
  value: string;
  onChange: (s: string) => void;
  stocks: StockSummary[];
  color: string;
}) {
  const [open,  setOpen]  = useState(false);
  const [query, setQuery] = useState("");

  const filtered = useMemo(() =>
    stocks
      .filter(s =>
        s.symbol.includes(query.toUpperCase()) ||
        s.name.toLowerCase().includes(query.toLowerCase())
      )
      .slice(0, 30),
    [stocks, query]
  );

  const current = stocks.find(s => s.symbol === value);

  return (
    <div className="relative">
      <span className="block text-[10px] font-bold uppercase tracking-widest mb-1.5"
        style={{ color }}>
        {label}
      </span>
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-semibold"
        style={{ background: CARD_H, border: `1px solid ${color}40`, color: TEXT_PRI, minWidth: 160 }}>
        <span className="flex-1 text-left truncate">
          {value}{current ? ` — ${current.name.slice(0, 10)}` : ""}
        </span>
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
          <path d="M2 4l3 3 3-3" stroke={TEXT_SEC} strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      </button>

      {open && (
        <div className="absolute top-full left-0 mt-1 w-60 rounded-lg overflow-hidden z-30"
          style={{ background: "hsl(215,25%,13%)", border: `1px solid ${BORDER}`, boxShadow: "0 8px 32px rgba(0,0,0,0.6)" }}>
          <div className="p-2 border-b" style={{ borderColor: BORDER_D }}>
            <input
              autoFocus
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Search ticker or name…"
              className="w-full px-2 py-1.5 rounded-md text-xs outline-none"
              style={{ background: "hsl(215,25%,9%)", color: TEXT_PRI }}
            />
          </div>
          <div className="max-h-52 overflow-y-auto">
            {filtered.map(s => (
              <button
                key={s.symbol}
                onClick={() => { onChange(s.symbol); setOpen(false); setQuery(""); }}
                className="w-full flex items-center gap-3 px-3 py-2.5 text-left text-xs"
                style={{
                  background: s.symbol === value ? BLUE_DIM : "transparent",
                  color: s.symbol === value ? BLUE : TEXT_PRI,
                }}>
                <span className="font-bold w-12 flex-shrink-0">{s.symbol}</span>
                <span className="truncate" style={{ color: TEXT_SEC }}>{s.name}</span>
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

// ─── Cone Chart ───────────────────────────────────────────────────────────────
// Renders a single MC cone: training line + percentile bands + actual price line.

function ConeChart({
  result, title, titleColor,
}: {
  result: MonteCarloResult;
  title: string;
  titleColor: string;
}) {
  // Merge train + band data for a single continuous x-axis.
  // Train points get price only; band points get all percentiles + optional actual.
  const data = useMemo(() => {
    const trainPoints = result.train.map((t, i) => ({
      day:    -(result.train.length - i),
      date:   t.date,
      train:  t.price,
      // placeholders so Recharts doesn't complain about missing keys
      p5: undefined as number | undefined,
      p16: undefined as number | undefined,
      p50: undefined as number | undefined,
      p84: undefined as number | undefined,
      p95: undefined as number | undefined,
      actual: undefined as number | undefined,
    }));

    const bandPoints = result.bands.map(b => ({
      day:    b.day,
      date:   b.date,
      train:  undefined as number | undefined,
      p5:     b.p5,
      p16:    b.p16,
      p50:    b.p50,
      p84:    b.p84,
      p95:    b.p95,
      actual: b.actual,
    }));

    return [...trainPoints, ...bandPoints];
  }, [result]);

  const hasCoverage = result.coverage_1s !== undefined && result.coverage_1s !== null;

  return (
    <div className="flex-1 rounded-xl p-4" style={CARD}>
      {/* Panel title */}
      <p className="text-xs font-bold uppercase tracking-widest mb-1" style={{ color: titleColor }}>
        {title}
      </p>
      <p className="text-[10px] mb-3" style={{ color: TEXT_MUT }}>
        {result.symbol} · μ={pct(result.mu_annual)} · σ={pct(result.sigma_annual)} · {result.n_sims} paths
      </p>

      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={BORDER} />
            <XAxis
              dataKey="day"
              axisLine={false}
              tickLine={false}
              tick={labelStyle}
              interval="preserveStartEnd"
              minTickGap={50}
              tickFormatter={v => v === 0 ? "Split" : v > 0 ? `+${v}d` : `${v}d`}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={labelStyle}
              width={52}
              tickFormatter={v => `$${v >= 1000 ? `${(v/1000).toFixed(1)}k` : v.toFixed(0)}`}
              domain={["auto", "auto"]}
            />
            <Tooltip
              contentStyle={{
                background: "hsl(215,25%,13%)",
                border: `1px solid ${BORDER}`,
                borderRadius: 8,
                fontSize: 11,
              }}
              labelStyle={{ color: TEXT_SEC, marginBottom: 4 }}
              formatter={(value: number | string | undefined, name: string) => {
                const labels: Record<string, string> = {
                  train:  "Historical",
                  p50:    "Median",
                  actual: "Actual",
                  p84:    "84th pct",
                  p16:    "16th pct",
                };
                const num = typeof value === "number" ? value : parseFloat(String(value ?? 0));
                return [`$${num.toFixed(2)}`, labels[name] ?? name];
              }}
            />

            {/* Split line */}
            <ReferenceLine
              x={0}
              stroke={TEXT_MUT}
              strokeDasharray="4 3"
              strokeOpacity={0.6}
              label={{ value: "Split →", fill: TEXT_MUT, fontSize: 9, position: "insideTopRight" }}
            />

            {/* 2σ outer band */}
            <Area
              type="monotone"
              dataKey="p95"
              stroke="none"
              fill={CONE_FILL_2}
              fillOpacity={1}
              isAnimationActive={false}
              connectNulls
            />
            <Area
              type="monotone"
              dataKey="p5"
              stroke="none"
              fill={BG}
              fillOpacity={1}
              isAnimationActive={false}
              connectNulls
            />

            {/* 1σ inner band */}
            <Area
              type="monotone"
              dataKey="p84"
              stroke="none"
              fill={CONE_FILL}
              fillOpacity={1}
              isAnimationActive={false}
              connectNulls
            />
            <Area
              type="monotone"
              dataKey="p16"
              stroke="none"
              fill={BG}
              fillOpacity={1}
              isAnimationActive={false}
              connectNulls
            />

            {/* Median line */}
            <Line
              type="monotone"
              dataKey="p50"
              stroke={MEDIAN_CLR}
              strokeWidth={1.8}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />

            {/* Training price line */}
            <Line
              type="monotone"
              dataKey="train"
              stroke={TRAIN_CLR}
              strokeWidth={1.4}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />

            {/* Actual price line */}
            <Line
              type="monotone"
              dataKey="actual"
              stroke={ACTUAL_CLR}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Annotations */}
      <div className="flex items-center justify-between mt-3 flex-wrap gap-2">
        <div className="flex items-center gap-4 flex-wrap">
          {[
            { color: TRAIN_CLR,  label: "Train (real)" },
            { color: MEDIAN_CLR, label: "MC Median" },
            { color: ACTUAL_CLR, label: "Actual (real)" },
            { color: CONE_FILL,  label: "1σ band", fill: true },
          ].map(({ color, label, fill }) => (
            <div key={label} className="flex items-center gap-1.5">
              {fill
                ? <div className="w-4 h-3 rounded-sm" style={{ background: color }} />
                : <div className="w-4 h-0.5 rounded" style={{ background: color }} />
              }
              <span className="text-[10px]" style={{ color: TEXT_MUT }}>{label}</span>
            </div>
          ))}
        </div>

        {hasCoverage && (
          <span
            className="text-[10px] font-semibold px-2 py-0.5 rounded"
            style={{
              background: "hsla(0,84%,60%,0.1)",
              color: ACTUAL_CLR,
              border: "1px solid hsla(0,84%,60%,0.25)",
            }}>
            1σ coverage: {pct(result.coverage_1s)}
          </span>
        )}
      </div>
    </div>
  );
}

// ─── Single Stock MC Panel ────────────────────────────────────────────────────

function SingleStockMC({ stocks }: { stocks: StockSummary[] }) {
  const [symbol,  setSymbol]  = useState("AAPL");
  const [result,  setResult]  = useState<MonteCarloResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const r = await fetchMonteCarlo(symbol);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Simulation failed.");
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  return (
    <div className="rounded-xl p-5" style={CARD}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-5 pb-4"
        style={{ borderBottom: `1px solid ${BORDER}` }}>
        <div>
          <h3 className="text-sm font-bold" style={{ color: TEXT_PRI }}>Single Stock Simulation</h3>
          <p className="text-xs mt-0.5" style={{ color: TEXT_MUT }}>
            GBM paths trained on 2015–2019 · projected over 2020–2024 test period
          </p>
        </div>

        <div className="flex items-end gap-3">
          <MCStockPicker
            label="Stock"
            value={symbol}
            onChange={setSymbol}
            stocks={stocks}
            color={BLUE}
          />
          <button
            onClick={run}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-bold disabled:opacity-40"
            style={{ background: BLUE, color: "white" }}>
            {loading
              ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
              : <TrendingUp className="w-3.5 h-3.5" />
            }
            Run
          </button>
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

      {/* Result */}
      {result && !loading && (
        <>
          {/* Stats strip */}
          <div className="flex gap-6 mb-4 px-4 py-3 rounded-lg flex-wrap"
            style={{ background: CARD_H, border: `1px solid ${BORDER_D}` }}>
            <StatPill label="Symbol"     value={result.symbol} color={BLUE} />
            <StatPill label="Annual μ"   value={pct(result.mu_annual)} />
            <StatPill label="Annual σ"   value={pct(result.sigma_annual)} />
            <StatPill label="Paths"      value={result.n_sims.toLocaleString()} />
            <StatPill
              label="1σ Coverage"
              value={pct(result.coverage_1s)}
              color={result.coverage_1s >= 0.55 && result.coverage_1s <= 0.80 ? GREEN : AMBER}
            />
          </div>

          <ConeChart
            result={result}
            title="Monte Carlo Cone — Pure GBM"
            titleColor={BLUE}
          />
        </>
      )}

      {/* Empty state */}
      {!result && !loading && !error && (
        <div className="h-48 flex flex-col items-center justify-center gap-3 rounded-xl"
          style={{ background: "hsl(215,25%,9%)" }}>
          <TrendingUp className="w-8 h-8 opacity-20" style={{ color: BLUE }} />
          <p className="text-sm" style={{ color: TEXT_MUT }}>
            Select a stock and click Run to simulate price paths
          </p>
        </div>
      )}

      {loading && (
        <div className="h-48 flex items-center justify-center">
          <Loader2 className="w-7 h-7 animate-spin" style={{ color: BLUE }} />
        </div>
      )}
    </div>
  );
}

// ─── Pair MC Panel ────────────────────────────────────────────────────────────

function PairMC({ stocks }: { stocks: StockSummary[] }) {
  const [leader,   setLeader]   = useState("WM");
  const [follower, setFollower] = useState("WMB");
  const [result,   setResult]   = useState<PairMonteCarloResult | null>(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);

  const run = useCallback(async () => {
    if (!leader.trim() || !follower.trim() || leader === follower) return;
    setLoading(true); setError(null);
    try {
      const r = await fetchPairMonteCarlo(leader, follower);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Simulation failed.");
    } finally {
      setLoading(false);
    }
  }, [leader, follower]);

  const LEADER_CLR   = BLUE;
  const FOLLOWER_CLR = GREEN;

  return (
    <div className="rounded-xl p-5" style={CARD}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-5 pb-4"
        style={{ borderBottom: `1px solid ${BORDER}` }}>
        <div>
          <h3 className="text-sm font-bold" style={{ color: TEXT_PRI }}>Lead-Lag Pair Simulation</h3>
          <p className="text-xs mt-0.5" style={{ color: TEXT_MUT }}>
            Compares follower cone WITH vs WITHOUT leader influence · OLS beta calibrated on 2015–2019 training data
          </p>
        </div>

        <div className="flex items-end gap-3 flex-wrap">
          <MCStockPicker
            label="Leader"
            value={leader}
            onChange={setLeader}
            stocks={stocks}
            color={LEADER_CLR}
          />

          {/* Arrow between pickers */}
          <div className="pb-2">
            <ArrowRight className="w-4 h-4" style={{ color: TEXT_MUT }} />
          </div>

          <MCStockPicker
            label="Follower"
            value={follower}
            onChange={setFollower}
            stocks={stocks}
            color={FOLLOWER_CLR}
          />

          <button
            onClick={run}
            disabled={loading || leader === follower}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-bold disabled:opacity-40"
            style={{ background: BLUE, color: "white" }}>
            {loading
              ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
              : <TrendingUp className="w-3.5 h-3.5" />
            }
            Run
          </button>
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

      {/* Result */}
      {result && !loading && (
        <>
          {/* Pair stats strip */}
          <div className="flex gap-6 mb-5 px-4 py-3 rounded-lg flex-wrap items-center"
            style={{ background: CARD_H, border: `1px solid ${BORDER_D}` }}>
            <div>
              <p className="text-[10px] font-bold uppercase tracking-widest mb-0.5" style={{ color: TEXT_MUT }}>
                Detected Direction
              </p>
              <p className="text-sm font-bold">
                <span style={{ color: LEADER_CLR }}>{result.leader}</span>
                <span style={{ color: TEXT_MUT }}> leads </span>
                <span style={{ color: FOLLOWER_CLR }}>{result.follower}</span>
              </p>
            </div>
            <StatPill label="Lag"         value={`${result.lag}d`}                 color={BLUE} />
            <StatPill label="OLS β"       value={result.beta.toFixed(4)}            color={result.beta >= 0 ? GREEN : RED} />
            <StatPill label="Pearson ρ"   value={result.pearson.toFixed(3)}         color={result.pearson >= 0 ? GREEN : RED} />
            <StatPill label="Direction"   value={result.beta >= 0 ? "↑ Positive" : "↓ Inverse"}
              color={result.beta >= 0 ? GREEN : AMBER} />

            {/* 1σ coverage comparison */}
            <div className="ml-auto flex items-center gap-3">
              <div className="flex flex-col gap-0.5 items-end">
                <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: TEXT_MUT }}>
                  1σ Coverage
                </span>
                <div className="flex items-center gap-3">
                  <span className="text-xs" style={{ color: TEXT_SEC }}>
                    w/o LL: <strong style={{ color: TEXT_PRI }}>{pct(result.without.coverage_1s)}</strong>
                  </span>
                  <span className="text-xs" style={{ color: TEXT_SEC }}>
                    w/ LL: <strong style={{ color: result.with_ll.coverage_1s > result.without.coverage_1s ? GREEN : AMBER }}>
                      {pct(result.with_ll.coverage_1s)}
                    </strong>
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Side-by-side cones */}
          <div className="flex flex-col lg:flex-row gap-4">
            <ConeChart
              result={result.without}
              title={`WITHOUT Lead-Lag — Pure GBM (${result.follower})`}
              titleColor={TEXT_SEC}
            />
            <ConeChart
              result={result.with_ll}
              title={`WITH Lead-Lag — VAR-MC + OLS β (${result.leader} → ${result.follower})`}
              titleColor={BLUE}
            />
          </div>

          {/* Explanation note */}
          <p className="text-[10px] mt-3 leading-relaxed" style={{ color: TEXT_MUT }}>
            <strong style={{ color: TEXT_SEC }}>How to read this:</strong>{" "}
            The blue cone is where the model thinks the follower stock could go.
            The left panel uses pure GBM with no information about the leader.
            The right panel adds the lead-lag relationship — if the leader's past
            returns carry information about the follower's future returns, the right
            cone should have a different median trajectory or width.
            The red line is the actual real price — how often it stays inside the
            1σ band tells you how well-calibrated the model is.
          </p>
        </>
      )}

      {/* Empty state */}
      {!result && !loading && !error && (
        <div className="h-48 flex flex-col items-center justify-center gap-3 rounded-xl"
          style={{ background: "hsl(215,25%,9%)" }}>
          <FlaskConical className="w-8 h-8 opacity-20" style={{ color: BLUE }} />
          <p className="text-sm" style={{ color: TEXT_MUT }}>
            Select a leader and follower stock, then click Run
          </p>
        </div>
      )}

      {loading && (
        <div className="h-48 flex items-center justify-center">
          <Loader2 className="w-7 h-7 animate-spin" style={{ color: BLUE }} />
        </div>
      )}
    </div>
  );
}

// ─── Main export ──────────────────────────────────────────────────────────────

export default function MonteCarlo({
  stocks,
  stocksLoading,
}: {
  stocks: StockSummary[];
  stocksLoading: boolean;
}) {
  const [mode, setMode] = useState<"single" | "pair">("single");

  return (
    <div className="rounded-xl p-5" style={{ background: "hsl(215, 25%, 11%)", border: `1px solid ${BORDER}` }}>
      {/* Section header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-5 pb-4"
        style={{ borderBottom: `1px solid ${BORDER}` }}>
        <div className="flex items-center gap-3">
          <FlaskConical className="w-5 h-5" style={{ color: BLUE }} />
          <div>
            <h2 className="text-base font-bold" style={{ color: TEXT_PRI }}>Monte Carlo Simulation Lab</h2>
            <p className="text-xs mt-0.5" style={{ color: TEXT_MUT }}>
              GBM price path simulation · single stock or lead-lag pair comparison
            </p>
          </div>
        </div>

        {/* Mode toggle */}
        <div className="flex items-center p-0.5 rounded-lg"
          style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER}` }}>
          {(["single", "pair"] as const).map(m => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className="px-4 py-1.5 rounded-md text-xs font-semibold"
              style={mode === m
                ? { background: BLUE, color: "white" }
                : { color: TEXT_SEC, background: "transparent" }}>
              {m === "single" ? "Single Stock" : "Lead-Lag Pair"}
            </button>
          ))}
        </div>
      </div>

      {/* Panel */}
      {stocksLoading ? (
        <div className="h-32 flex items-center justify-center">
          <Loader2 className="w-6 h-6 animate-spin" style={{ color: BLUE }} />
        </div>
      ) : mode === "single" ? (
        <SingleStockMC stocks={stocks} />
      ) : (
        <PairMC stocks={stocks} />
      )}
    </div>
  );
}
