// frontend/src/app/dashboard/model/page.tsx
"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  TrendingUp, TrendingDown, BarChart2,
  Loader2, ChevronRight, Layers, Star, Info,
} from "lucide-react";
import Sidebar from "@/components/ui/Sidebar";
import { useAuth } from "@/src/app/context/AuthContext";
import StockModal, { type Stock } from "@/components/ui/StockModal";
import { PageHelp } from "@/components/ui/PageHelp";
import TickerSearchInput from "@/components/ui/TickerSearchInput";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Leader {
  rank:        number;
  symbol:      string;
  name:        string;
  sector:      string;
  lag_days:    number;
  attn_weight: number;
  leader_ic:   number;
  price:       string;
  change:      string;
  volume:      string;
  positive:    boolean;
}

interface SectorInfo {
  count:        number;
  total_weight: number;
}

interface AnalyzeResult {
  target:           string;
  target_sector:    string;
  target_ic:        number;
  as_of_date:       string;
  leaders:          Leader[];
  sector_breakdown: Record<string, SectorInfo>;
}

interface TickerOption {
  symbol: string;
  name:   string;
  sector: string;
}

const SECTOR_COLORS: Record<string, string> = {
  "Information Technology": "hsl(217, 91%, 60%)",
  "Communication":          "hsl(270, 70%, 65%)",
  "Consumer Discretionary": "hsl(35, 90%, 55%)",
  "Consumer Staples":       "hsl(145, 60%, 45%)",
  "Energy":                 "hsl(25, 85%, 55%)",
  "Financials":             "hsl(195, 80%, 50%)",
  "Health Care":            "hsl(155, 65%, 45%)",
  "Industrials":            "hsl(45, 85%, 52%)",
  "Materials":              "hsl(15, 75%, 55%)",
  "Real Estate":            "hsl(330, 65%, 60%)",
  "Utilities":              "hsl(175, 60%, 45%)",
  // aliases
  "Technology":             "hsl(217, 91%, 60%)",
  "Communication Services": "hsl(270, 70%, 65%)",
};

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:4000";

// ─── Attention weight bar ─────────────────────────────────────────────────────
function AttnBar({ weight, max }: { weight: number; max: number }) {
  const pct = max > 0 ? (weight / max) * 100 : 0;
  return (
    <div className="flex items-center gap-2.5 w-full">
      <div className="flex-1 rounded-full overflow-hidden" style={{ height: 6, background: "hsl(215, 20%, 18%)" }}>
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: "linear-gradient(90deg, hsl(217,91%,50%), hsl(142,71%,45%))" }}
        />
      </div>
      <span
        className="text-xs font-semibold w-12 text-right"
        style={{ color: "hsl(217, 91%, 70%)", fontVariantNumeric: "tabular-nums" }}
      >
        {(weight * 100).toFixed(1)}%
      </span>
    </div>
  );
}

// ─── Sector bar row ───────────────────────────────────────────────────────────
function SectorRow({ sector, info, maxWeight }: { sector: string; info: SectorInfo; maxWeight: number }) {
  const color = SECTOR_COLORS[sector] ?? "hsl(215, 15%, 55%)";
  const pct = maxWeight > 0 ? (info.total_weight / maxWeight) * 100 : 0;
  return (
    <div className="flex items-center gap-3 py-2">
      <div className="w-48 flex-shrink-0">
        <span className="text-xs" style={{ color: "hsl(215, 15%, 60%)" }}>{sector}</span>
      </div>
      <div className="flex-1 rounded-full overflow-hidden" style={{ height: 8, background: "hsl(215, 20%, 16%)" }}>
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-xs font-semibold w-6 text-right" style={{ color, fontVariantNumeric: "tabular-nums" }}>
        {info.count}
      </span>
      <span className="text-xs w-10 text-right" style={{ color: "hsl(215, 15%, 45%)", fontVariantNumeric: "tabular-nums" }}>
        {(info.total_weight * 100).toFixed(0)}%
      </span>
    </div>
  );
}

// ─── Lag badge ────────────────────────────────────────────────────────────────
function LagBadge({ days }: { days: number }) {
  // Colour: short lag = green, longer = orange
  const hue = days <= 3 ? 142 : days <= 6 ? 45 : 25;
  return (
    <span
      className="text-xs px-1.5 py-0.5 rounded font-semibold"
      style={{
        background: `hsla(${hue}, 70%, 45%, 0.15)`,
        color:       `hsl(${hue}, 70%, 55%)`,
        border:      `1px solid hsla(${hue}, 70%, 45%, 0.3)`,
        fontVariantNumeric: "tabular-nums",
        whiteSpace: "nowrap",
      }}
    >
      {days}d lag
    </span>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────
export default function ModelPage() {
  const { user, loading: authLoading, isSaved, toggleSave } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!authLoading && !user) router.replace("/");
  }, [user, authLoading, router]);

  const [tickers,        setTickers]        = useState<TickerOption[]>([]);
  const [selectedTicker, setSelectedTicker] = useState<TickerOption | null>(null);
  const [analyzing,      setAnalyzing]      = useState(false);
  const [result,         setResult]         = useState<AnalyzeResult | null>(null);
  const [error,          setError]          = useState<string | null>(null);
  const [selectedStock,  setSelectedStock]  = useState<Stock | null>(null);
  const [lagWindow,      setLagWindow]      = useState<1 | 5 | 10>(5);

  // Load ticker list on mount
  useEffect(() => {
    fetch(`${BASE}/api/model/tickers`)
      .then(r => r.json())
      .then(d => setTickers(d.tickers ?? []))
      .catch(() => {});
  }, []);

  const selectTicker = (symbol: string) => {
    const t = tickers.find(x => x.symbol === symbol) ?? null;
    setSelectedTicker(t);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedTicker) return;
    setAnalyzing(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${BASE}/api/model/analyze`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ symbol: selectedTicker.symbol }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail ?? "Analysis failed");
      }
      setResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setAnalyzing(false);
    }
  };

  // Filter by lag window, then deduplicate by (symbol, lag_days) pair so
  // each distinct lead-lag relationship appears once, then take top 10.
  const visibleLeaders = result
    ? Object.values(
        result.leaders
          .filter(l => l.lag_days <= lagWindow)
          .reduce<Record<string, Leader>>((acc, l) => {
            if (!acc[l.symbol] || l.attn_weight > acc[l.symbol].attn_weight)
              acc[l.symbol] = l;
            return acc;
          }, {})
      ).sort((a, b) => b.attn_weight - a.attn_weight).slice(0, 10)
    : [];

  // Recompute sector breakdown for the current lag window
  const visibleSectors: Record<string, { count: number; total_weight: number }> = {};
  for (const l of visibleLeaders) {
    if (!visibleSectors[l.sector]) visibleSectors[l.sector] = { count: 0, total_weight: 0 };
    visibleSectors[l.sector].count        += 1;
    visibleSectors[l.sector].total_weight  = Math.round((visibleSectors[l.sector].total_weight + l.attn_weight) * 10000) / 10000;
  }
  const visibleSectorsSorted = Object.fromEntries(
    Object.entries(visibleSectors).sort((a, b) => b[1].total_weight - a[1].total_weight)
  );

  const maxAttn   = visibleLeaders.length ? Math.max(...visibleLeaders.map(l => l.attn_weight)) : 1;
  const maxWeight = Object.keys(visibleSectors).length
    ? Math.max(...Object.values(visibleSectors).map(s => s.total_weight))
    : 1;

  const card = { background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" };

  return (
    <div className="min-h-screen" style={{ background: "hsl(213, 27%, 7%)" }}>
      <Sidebar />

      <main className="pt-14">
        <div className="max-w-6xl mx-auto px-6 py-8">

          {/* ── Page header ── */}
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-1">
              <div
                className="w-9 h-9 rounded-xl flex items-center justify-center"
                style={{ background: "hsl(217, 91%, 60% / 0.15)", border: "1px solid hsl(217, 91%, 60% / 0.3)" }}
              >
                <BarChart2 className="w-4.5 h-4.5" style={{ color: "hsl(217, 91%, 65%)" }} />
              </div>
              <h1 className="text-2xl font-bold" style={{ color: "hsl(210, 40%, 92%)" }}>
                DeltaLag Model
              </h1>
              <div className="ml-auto">
                <PageHelp
                  title="DeltaLag Model Guide"
                  subtitle="Understand what the model does and how to interpret its results."
                  sections={[
                    {
                      title: "What is DeltaLag?",
                      body: "DeltaLag is a 2-layer GRU (Gated Recurrent Unit) neural network with cross-attention, trained on 2,000+ stocks from the Russell 1000 and Russell 2000. It learns which stocks historically precede moves in a target stock, and by exactly how many trading days.",
                    },
                    {
                      title: "How to Use",
                      body: "Search for any supported stock in the input box, select it from the autocomplete dropdown, then choose a lag window (1D, 5D, or 10D). Click Analyze to run the model. Results show the top 10 stocks that historically lead your target within that lag window.",
                      color: "hsl(142, 71%, 45%)",
                    },
                    {
                      title: "Attention Weight",
                      body: "The attention bar shows how strongly the model's cross-attention mechanism focuses on a leader stock when predicting the target's returns. A longer bar = more influential. Compare bars across leaders to understand their relative importance.",
                      color: "hsl(217, 91%, 60%)",
                    },
                    {
                      title: "Lag Days",
                      body: "The lag badge shows the number of trading days between the leader's move and the target stock's typical response. Green (≤3d) = short, actionable lag. Orange (≤6d) = medium. Red = longer lag. Shorter lags generally offer more practical trading utility.",
                      color: "hsl(38, 92%, 50%)",
                    },
                    {
                      title: "IC (Information Coefficient)",
                      body: "Shown in the results header, IC measures how well the model's predictions correlate with the target stock's actual returns. A positive IC means the model has predictive edge for this stock — higher is better. Results are as of 2024-12-30.",
                      color: "hsl(270, 70%, 65%)",
                    },
                    {
                      title: "Leaders by Sector",
                      body: "The sector breakdown panel shows which industries the top leader stocks come from, weighted by their total attention scores. A dominant sector means the target is heavily influenced by that sector's dynamics — useful context for macro risk.",
                      color: "hsl(195, 80%, 50%)",
                    },
                  ]}
                />
              </div>
            </div>
            <p className="text-sm ml-12" style={{ color: "hsl(215, 15%, 50%)" }}>
              Identify which stocks historically lead a target — using a 2-layer GRU
              with cross-attention trained on the Russell 1000 + top-1000 Russell 2000.
              Results are as of <span style={{ color: "hsl(215, 15%, 65%)" }}>2024-12-30</span>.
            </p>
          </div>

          {/* ── Input row ── */}
          <div className="flex items-end gap-4 mb-8">
            <div className="flex-1 max-w-xs">
              <TickerSearchInput
                value={selectedTicker?.symbol ?? ""}
                onChange={selectTicker}
                stocks={tickers}
                label="Target Stock"
                placeholder="e.g. AAPL, JPM, XOM…"
              />
            </div>

            {/* Lag window */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium" style={{ color: "hsl(215, 15%, 55%)" }}>Lag Window</label>
              <div className="flex items-center rounded-lg overflow-hidden" style={{ border: "1px solid hsl(215, 20%, 22%)", background: "hsl(215, 25%, 13%)" }}>
                {([1, 5, 10] as const).map((v) => (
                  <button
                    key={v}
                    onClick={() => setLagWindow(v)}
                    className="px-4 py-2.5 text-sm font-medium transition-all"
                    style={{ background: lagWindow === v ? "hsl(217, 91%, 60%)" : "transparent", color: lagWindow === v ? "white" : "hsl(215, 15%, 55%)" }}
                  >
                    {v}d
                  </button>
                ))}
              </div>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={!selectedTicker || analyzing}
              className="flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-semibold text-white transition-all hover:opacity-90 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed"
              style={{ background: "hsl(217, 91%, 60%)" }}
            >
              {analyzing
                ? <><Loader2 className="w-4 h-4 animate-spin" />Analyzing…</>
                : <><ChevronRight className="w-4 h-4" />Analyze</>}
            </button>
          </div>

          {/* ── Error ── */}
          {error && (
            <div className="mb-6 px-4 py-3 rounded-xl text-sm" style={{ background: "hsl(0, 84%, 10%)", border: "1px solid hsl(0, 84%, 20%)", color: "hsl(0, 84%, 65%)" }}>
              {error}
            </div>
          )}

          {/* ── Loading ── */}
          {analyzing && (
            <div className="rounded-xl p-5 animate-pulse" style={card}>
              <div className="h-4 w-48 rounded mb-5" style={{ background: "hsl(215,20%,18%)" }} />
              {Array.from({ length: 10 }).map((_, i) => (
                <div key={i} className="flex items-center gap-4 py-3.5" style={{ borderTop: "1px solid hsl(215,20%,16%)" }}>
                  <div className="h-3 w-6 rounded"  style={{ background: "hsl(215,20%,18%)" }} />
                  <div className="h-3 w-24 rounded" style={{ background: "hsl(215,20%,18%)" }} />
                  <div className="h-3 w-16 rounded" style={{ background: "hsl(215,20%,18%)" }} />
                  <div className="h-3 flex-1 rounded" style={{ background: "hsl(215,20%,16%)" }} />
                </div>
              ))}
            </div>
          )}

          {/* ── Results ── */}
          {result && !analyzing && (
            <div className="space-y-6">

              {/* Result header */}
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-sm" style={{ color: "hsl(215, 15%, 50%)" }}>Top 10 leaders for</span>
                  <span className="text-sm font-bold" style={{ color: "hsl(217, 91%, 65%)" }}>{result.target}</span>
                  <span
                    className="text-xs px-2 py-0.5 rounded"
                    style={{ background: `${SECTOR_COLORS[result.target_sector] ?? "hsl(215,15%,40%)"}22`, color: SECTOR_COLORS[result.target_sector] ?? "hsl(215,15%,55%)" }}
                  >
                    {result.target_sector}
                  </span>
                  <span className="text-xs" style={{ color: "hsl(215, 15%, 40%)" }}>as of {result.as_of_date}</span>
                </div>
                <div className="flex items-center gap-3">
                  <div
                    className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg"
                    style={{ background: "hsl(215,25%,14%)", border: "1px solid hsl(215,20%,20%)", color: "hsl(215,15%,50%)" }}
                  >
                    <Info className="w-3 h-3" />
                    IC: <span style={{ color: result.target_ic >= 0 ? "hsl(142,71%,50%)" : "hsl(0,84%,60%)" }}>{result.target_ic.toFixed(4)}</span>
                  </div>
                  <div className="flex items-center rounded-lg overflow-hidden" style={{ border: "1px solid hsl(215, 20%, 22%)", background: "hsl(215, 25%, 13%)" }}>
                    {([1, 5, 10] as const).map((v) => (
                      <button
                        key={v}
                        onClick={() => setLagWindow(v)}
                        className="px-3 py-1.5 text-xs font-semibold transition-all"
                        style={{ background: lagWindow === v ? "hsl(217, 91%, 60%)" : "transparent", color: lagWindow === v ? "white" : "hsl(215, 15%, 55%)" }}
                      >
                        {v}d
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* ── Leaders table ── */}
              {visibleLeaders.length === 0 && (
                <div className="rounded-xl px-6 py-8 text-sm text-center" style={{ ...card, color: "hsl(215,15%,50%)" }}>
                  No leaders found within a {lagWindow}-day lag window. Try a wider window.
                </div>
              )}
              {visibleLeaders.length > 0 && <div className="rounded-xl overflow-hidden" style={card}>
                <div
                  className="grid px-5 py-3 text-xs font-medium"
                  style={{ gridTemplateColumns: "28px 1fr 130px 60px 80px 80px 1fr", color: "hsl(215, 15%, 45%)", borderBottom: "1px solid hsl(215, 20%, 17%)" }}
                >
                  <span>#</span>
                  <span>Ticker</span>
                  <span>Sector</span>
                  <span className="text-center">Lag</span>
                  <span className="text-right">Price</span>
                  <span className="text-right">Change</span>
                  <span className="pl-3">Attention</span>
                </div>

                {visibleLeaders.map((leader, i) => {
                  const sectorColor = SECTOR_COLORS[leader.sector] ?? "hsl(215,15%,50%)";
                  const saved = isSaved(leader.symbol);
                  return (
                    <div
                      key={`${leader.symbol}-${leader.lag_days}`}
                      onClick={() => setSelectedStock({
                        symbol:   leader.symbol,
                        name:     leader.name,
                        price:    leader.price,
                        change:   leader.change,
                        volume:   leader.volume,
                        positive: leader.positive,
                      })}
                      className="grid px-5 py-3.5 items-center cursor-pointer transition-colors"
                      style={{ gridTemplateColumns: "28px 1fr 130px 60px 80px 80px 1fr", borderTop: "1px solid hsl(215, 20%, 16%)" }}
                      onMouseEnter={(e) => (e.currentTarget.style.background = "hsl(215, 25%, 14%)")}
                      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                    >
                      {/* Rank */}
                      <span className="text-sm font-bold" style={{ color: i < 3 ? "hsl(217, 91%, 65%)" : "hsl(215, 15%, 38%)" }}>
                        {leader.rank}
                      </span>

                      {/* Ticker + name + star */}
                      <div className="flex flex-col min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className="text-sm font-bold" style={{ color: "hsl(210, 40%, 92%)" }}>{leader.symbol}</span>
                          <button
                            onClick={(e) => { e.stopPropagation(); toggleSave({ symbol: leader.symbol, name: leader.name }); }}
                            className="transition-all hover:scale-110"
                            aria-label={saved ? "Unsave" : "Save"}
                          >
                            <Star className="w-3 h-3" style={saved
                              ? { fill: "hsl(48,96%,53%)", color: "hsl(48,96%,53%)" }
                              : { fill: "transparent",      color: "hsl(215,15%,38%)" }} />
                          </button>
                        </div>
                        <span className="text-xs truncate" style={{ color: "hsl(215, 15%, 50%)" }}>{leader.name}</span>
                      </div>

                      {/* Sector pill */}
                      <div>
                        <span className="text-xs px-2 py-0.5 rounded-md font-medium" style={{ background: `${sectorColor}1a`, color: sectorColor, border: `1px solid ${sectorColor}33` }}>
                          {leader.sector}
                        </span>
                      </div>

                      {/* Lag badge */}
                      <div className="flex justify-center">
                        <LagBadge days={leader.lag_days} />
                      </div>

                      {/* Price */}
                      <span className="text-sm font-medium text-right" style={{ color: "hsl(210, 40%, 88%)" }}>{leader.price}</span>

                      {/* Change */}
                      <span className="text-sm font-semibold text-right" style={{ color: leader.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)" }}>
                        {leader.change}
                      </span>

                      {/* Attention weight bar */}
                      <div className="pl-3">
                        <AttnBar weight={leader.attn_weight} max={maxAttn} />
                      </div>
                    </div>
                  );
                })}
              </div>

              }
              {/* ── Sector breakdown ── */}
              <div className="rounded-xl p-6" style={card}>
                <div className="flex items-center gap-2 mb-5">
                  <Layers className="w-4 h-4" style={{ color: "hsl(217, 91%, 60%)" }} />
                  <h2 className="text-base font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>Leaders by Sector</h2>
                  <span className="text-xs" style={{ color: "hsl(215, 15%, 45%)" }}>— weighted by attention score</span>
                </div>

                <div className="space-y-0.5">
                  {Object.entries(visibleSectorsSorted).map(([sector, info]) => (
                    <SectorRow key={sector} sector={sector} info={info} maxWeight={maxWeight} />
                  ))}
                </div>

                {/* Summary pills */}
                <div className="flex flex-wrap gap-2 mt-5 pt-4" style={{ borderTop: "1px solid hsl(215,20%,17%)" }}>
                  {Object.entries(visibleSectorsSorted).map(([sector, info]) => {
                    const color = SECTOR_COLORS[sector] ?? "hsl(215,15%,50%)";
                    return (
                      <div key={sector} className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium" style={{ background: `${color}15`, border: `1px solid ${color}30`, color }}>
                        <span>{sector}</span>
                        <span className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold" style={{ background: `${color}30` }}>
                          {info.count}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

            </div>
          )}

          {/* ── Empty state ── */}
          {!result && !analyzing && !error && (
            <div className="rounded-2xl p-16 flex flex-col items-center text-center" style={card}>
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4" style={{ background: "hsl(217, 91%, 60% / 0.1)", border: "1px solid hsl(217, 91%, 60% / 0.2)" }}>
                <BarChart2 className="w-8 h-8" style={{ color: "hsl(217, 91%, 60%)" }} />
              </div>
              <h3 className="text-lg font-semibold mb-2" style={{ color: "hsl(210, 40%, 85%)" }}>Ready to analyze</h3>
              <p className="text-sm max-w-sm" style={{ color: "hsl(215, 15%, 45%)" }}>
                Search for any of the 1,164 supported stocks and click Analyze
                to discover which stocks historically lead it, along with the exact lag window.
              </p>
            </div>
          )}

          {/* ── Disclaimer ── */}
          <div
            className="mt-8 rounded-xl p-5 text-xs space-y-3"
            style={{ background: "hsl(215, 25%, 10%)", border: "1px solid hsl(215, 20%, 17%)", color: "hsl(215, 15%, 48%)" }}
          >
            <p>
              <span style={{ color: "hsl(215, 15%, 62%)" }}>About this model: </span>
              This tool is based on <span style={{ color: "hsl(215, 15%, 65%)" }}>DeltaLag (Zhou et al., ICAIF 2025)</span>, an academic model for detecting lead-lag relationships in equity markets. Our implementation matches the paper&apos;s reported metrics; Information Coefficient of{" "}
              <span style={{ color: "hsl(142, 71%, 45%)" }}>0.026</span>, Annualized Return, and Sharpe Ratio, all on a universe of approximately 2,000 U.S. equities.
            </p>
            <p>
              <span style={{ color: "hsl(215, 15%, 62%)" }}>Known limitation: </span>
              The model&apos;s attention mechanism tends to concentrate on a recurring pool of 5–10 high-influence stocks rather than identifying unique leader-lagger pairs for each target stock. This is a consequence of training on a universe where broad market factors drive much of the cross-sectional return variation, making it difficult for the model to isolate stock-specific lead-lag structure. The leaders displayed are statistically meaningful at the portfolio level but may not represent a distinct predictive relationship for your specific stock.
            </p>
            <p style={{ color: "hsl(215, 15%, 40%)" }}>
              This tool is intended for research and educational purposes only and does not constitute investment advice.
            </p>
          </div>

        </div>
      </main>

      {selectedStock && (
        <StockModal stock={selectedStock} onClose={() => setSelectedStock(null)} />
      )}
    </div>
  );
}