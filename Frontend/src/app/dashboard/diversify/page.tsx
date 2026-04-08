// Frontend/src/app/dashboard/diversify/page.tsx
"use client";
import { useState, useRef, useEffect } from "react";
import Sidebar from "@/components/ui/Sidebar";
import {
  analyzePortfolio,
  type OverlapResult,
  type Recommendation,
} from "@/src/app/lib/api";
import {
  AlertTriangle, TrendingUp, TrendingDown, Plus, X,
  ChevronDown, ChevronUp, Loader2, Sparkles, ArrowRight,
  ShieldAlert, BarChart3,
} from "lucide-react";

// ── Design tokens (match rest of app) ────────────────────────────────────────
const BG        = "hsl(213, 27%, 7%)";
const CARD      = { background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" };
const CARD_HOVER= "hsl(215, 25%, 13%)";
const TEXT_PRI  = "hsl(210, 40%, 92%)";
const TEXT_SEC  = "hsl(215, 15%, 55%)";
const TEXT_MUT  = "hsl(215, 15%, 40%)";
const BLUE      = "hsl(217, 91%, 60%)";
const BLUE_DIM  = "hsla(217, 91%, 60%, 0.15)";
const GREEN     = "hsl(142, 71%, 45%)";
const AMBER     = "hsl(38, 92%, 50%)";
const RED       = "hsl(0, 84%, 60%)";
const BORDER    = "hsl(215, 20%, 18%)";
const BORDER_DIM= "hsl(215, 20%, 16%)";

const SECTOR_COLORS: Record<string, string> = {
  Technology:  "hsl(217, 91%, 60%)",
  Financials:  "hsl(142, 71%, 45%)",
  Healthcare:  "hsl(280, 70%, 65%)",
  Energy:      "hsl(38, 92%, 50%)",
  Consumer:    "hsl(15, 90%, 58%)",
  Industrials: "hsl(195, 80%, 50%)",
  Utilities:   "hsl(60, 70%, 50%)",
};

function sectorColor(sector: string) {
  return SECTOR_COLORS[sector] ?? "hsl(215, 15%, 55%)";
}

function signalBar(score: number) {
  const pct  = Math.min(100, Math.max(0, score));
  const color = pct >= 80 ? GREEN : pct >= 65 ? BLUE : AMBER;
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: "hsl(215,20%,18%)" }}>
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-xs font-semibold w-8 text-right" style={{ color }}>{Math.round(pct)}</span>
    </div>
  );
}

function directionLabel(direction: string) {
  if (direction === "leads_your_holdings")
    return <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: "hsla(142,71%,45%,0.15)", color: GREEN }}>Leads your stocks</span>;
  if (direction === "follows_your_holdings")
    return <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: "hsla(217,91%,60%,0.15)", color: BLUE }}>Follows your stocks</span>;
  return <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: "hsla(38,92%,50%,0.15)", color: AMBER }}>Bidirectional</span>;
}

// ── Score breakdown tooltip ───────────────────────────────────────────────────
function ScoreBreakdown({ rec }: { rec: Recommendation }) {
  return (
    <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2">
      {[
        { label: "Signal Strength",  val: rec.signal_score },
        { label: "Market Centrality",val: rec.centrality_score },
        { label: "Portfolio Coverage",val: rec.coverage_score },
        { label: "Sector Diversity", val: rec.sector_diversity_score },
      ].map(({ label, val }) => (
        <div key={label}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs" style={{ color: TEXT_SEC }}>{label}</span>
          </div>
          {signalBar(val)}
        </div>
      ))}
    </div>
  );
}

// ── Overlap card ──────────────────────────────────────────────────────────────
function OverlapCard({ overlap }: { overlap: OverlapResult }) {
  const [expanded, setExpanded] = useState(false);
  const sameSector = overlap.sector_leader === overlap.sector_follower;

  return (
    <div className="rounded-xl overflow-hidden" style={CARD}>
      <div
        className="px-5 py-4 cursor-pointer"
        onClick={() => setExpanded(e => !e)}
        onMouseEnter={e => (e.currentTarget.style.background = CARD_HOVER)}
        onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
      >
        <div className="flex items-center gap-3">
          {/* Leader → Follower */}
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <span className="text-sm font-bold" style={{ color: TEXT_PRI }}>{overlap.ticker_leader}</span>
            <ArrowRight className="w-3.5 h-3.5 flex-shrink-0" style={{ color: AMBER }} />
            <span className="text-sm font-bold" style={{ color: TEXT_PRI }}>{overlap.ticker_follower}</span>
            <span className="text-xs ml-1" style={{ color: TEXT_MUT }}>lag {overlap.best_lag}d</span>
            {sameSector && (
              <span className="text-xs px-2 py-0.5 rounded-full ml-1"
                style={{ background: "hsla(38,92%,50%,0.12)", color: AMBER }}>
                Same sector
              </span>
            )}
          </div>

          {/* Signal strength */}
          <div className="w-28 flex-shrink-0">
            {signalBar(overlap.signal_strength)}
          </div>

          {expanded
            ? <ChevronUp className="w-4 h-4 flex-shrink-0" style={{ color: TEXT_MUT }} />
            : <ChevronDown className="w-4 h-4 flex-shrink-0" style={{ color: TEXT_MUT }} />}
        </div>

        {/* Sector tags */}
        <div className="flex items-center gap-2 mt-2">
          <span className="text-xs px-2 py-0.5 rounded-full"
            style={{ background: `${sectorColor(overlap.sector_leader)}20`, color: sectorColor(overlap.sector_leader) }}>
            {overlap.sector_leader}
          </span>
          <ArrowRight className="w-3 h-3" style={{ color: TEXT_MUT }} />
          <span className="text-xs px-2 py-0.5 rounded-full"
            style={{ background: `${sectorColor(overlap.sector_follower)}20`, color: sectorColor(overlap.sector_follower) }}>
            {overlap.sector_follower}
          </span>
        </div>
      </div>

      {expanded && (
        <div className="px-5 pb-4" style={{ borderTop: `1px solid ${BORDER_DIM}` }}>
          <p className="text-sm mt-3 leading-relaxed" style={{ color: TEXT_SEC }}>
            {overlap.interpretation}
          </p>
          <div className="grid grid-cols-3 gap-3 mt-4">
            {[
              { label: "dCor",        val: overlap.mean_dcor.toFixed(3) },
              { label: "OOS Sharpe",  val: overlap.oos_sharpe_net.toFixed(2) },
              { label: "Half-life",   val: `${Math.round(overlap.half_life)}d` },
            ].map(({ label, val }) => (
              <div key={label} className="rounded-lg px-3 py-2 text-center"
                style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER_DIM}` }}>
                <p className="text-xs mb-1" style={{ color: TEXT_MUT }}>{label}</p>
                <p className="text-sm font-semibold" style={{ color: TEXT_PRI }}>{val}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Recommendation card ───────────────────────────────────────────────────────
function RecCard({ rec, rank }: { rec: Recommendation; rank: number }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="rounded-xl overflow-hidden" style={CARD}>
      <div
        className="px-5 py-4 cursor-pointer"
        onClick={() => setExpanded(e => !e)}
        onMouseEnter={e => (e.currentTarget.style.background = CARD_HOVER)}
        onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
      >
        <div className="flex items-center gap-3">
          {/* Rank badge */}
          <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold"
            style={{ background: rank <= 3 ? BLUE_DIM : "hsl(215,25%,16%)", color: rank <= 3 ? BLUE : TEXT_SEC }}>
            {rank}
          </div>

          {/* Ticker + sector */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold" style={{ color: TEXT_PRI }}>{rec.ticker}</span>
              <span className="text-xs px-2 py-0.5 rounded-full"
                style={{ background: `${sectorColor(rec.sector)}20`, color: sectorColor(rec.sector) }}>
                {rec.sector}
              </span>
              {directionLabel(rec.direction)}
            </div>
            <p className="text-xs mt-0.5 truncate" style={{ color: TEXT_SEC }}>
              Relates to: {rec.related_holdings.join(", ")}
            </p>
          </div>

          {/* Composite score */}
          <div className="w-24 flex-shrink-0">
            {signalBar(rec.composite_score)}
          </div>

          {expanded
            ? <ChevronUp className="w-4 h-4 flex-shrink-0" style={{ color: TEXT_MUT }} />
            : <ChevronDown className="w-4 h-4 flex-shrink-0" style={{ color: TEXT_MUT }} />}
        </div>
      </div>

      {expanded && (
        <div className="px-5 pb-4" style={{ borderTop: `1px solid ${BORDER_DIM}` }}>
          <p className="text-sm mt-3 leading-relaxed" style={{ color: TEXT_SEC }}>{rec.reasoning}</p>
          <ScoreBreakdown rec={rec} />
        </div>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function DiversifyPage() {
  const [inputVal,   setInputVal]   = useState("");
  const [tickers,    setTickers]    = useState<string[]>([]);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState<string | null>(null);
  const [result,     setResult]     = useState<{
    tickers_analyzed: string[];
    unknown_tickers:  string[];
    overlaps:         OverlapResult[];
    recommendations:  Recommendation[];
  } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus input on mount
  useEffect(() => { inputRef.current?.focus(); }, []);

  const addTicker = (raw: string) => {
    const parts = raw.toUpperCase().split(/[\s,]+/).filter(Boolean);
    setTickers(prev => {
      const next = [...prev];
      for (const t of parts) {
        if (t.length <= 10 && !next.includes(t)) next.push(t);
      }
      return next;
    });
    setInputVal("");
  };

  const removeTicker = (t: string) => setTickers(prev => prev.filter(x => x !== t));

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (["Enter", ",", " ", "Tab"].includes(e.key)) {
      e.preventDefault();
      if (inputVal.trim()) addTicker(inputVal.trim());
    }
    if (e.key === "Backspace" && !inputVal && tickers.length) {
      setTickers(prev => prev.slice(0, -1));
    }
  };

  const handleAnalyze = async () => {
    if (inputVal.trim()) addTicker(inputVal.trim());
    const toAnalyze = inputVal.trim()
      ? [...tickers, ...inputVal.trim().toUpperCase().split(/[\s,]+/).filter(Boolean)]
      : tickers;
    if (toAnalyze.length === 0) return;

    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await analyzePortfolio(toAnalyze);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setTickers([]);
    setInputVal("");
    setResult(null);
    setError(null);
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  const hasResult = result !== null;

  return (
    <div className="min-h-screen" style={{ background: BG }}>
      <Sidebar />
      <main className="pt-14">
        <div className="max-w-4xl mx-auto px-6 py-8">

          {/* ── Header ── */}
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                style={{ background: BLUE_DIM }}>
                <Sparkles className="w-4 h-4" style={{ color: BLUE }} />
              </div>
              <h1 className="text-xl font-bold" style={{ color: TEXT_PRI }}>Portfolio Diversifier</h1>
            </div>
            <p className="text-sm leading-relaxed" style={{ color: TEXT_SEC }}>
              Enter your holdings to uncover hidden lead-lag relationships between your stocks
              and get signal-backed recommendations to reduce concentration risk.
            </p>
          </div>

          {/* ── Ticker input card ── */}
          <div className="rounded-xl p-5 mb-6" style={CARD}>
            <p className="text-xs font-medium mb-3" style={{ color: TEXT_SEC }}>
              YOUR HOLDINGS — type a ticker and press Enter, Space, or comma
            </p>

            {/* Tag input */}
            <div
              className="flex flex-wrap gap-2 min-h-12 p-2 rounded-lg cursor-text"
              style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER}` }}
              onClick={() => inputRef.current?.focus()}
            >
              {tickers.map(t => (
                <span key={t}
                  className="flex items-center gap-1 px-2.5 py-1 rounded-md text-sm font-semibold"
                  style={{ background: BLUE_DIM, color: BLUE }}>
                  {t}
                  <button onClick={(e) => { e.stopPropagation(); removeTicker(t); }}
                    className="hover:opacity-60 transition-opacity ml-0.5">
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
              <input
                ref={inputRef}
                value={inputVal}
                onChange={e => setInputVal(e.target.value.toUpperCase())}
                onKeyDown={handleKeyDown}
                onBlur={() => { if (inputVal.trim()) addTicker(inputVal.trim()); }}
                placeholder={tickers.length === 0 ? "AAPL, MSFT, NVDA..." : ""}
                className="flex-1 min-w-24 bg-transparent outline-none text-sm"
                style={{ color: TEXT_PRI }}
              />
            </div>

            {/* Quick-add examples */}
            {tickers.length === 0 && !result && (
              <div className="flex items-center gap-2 mt-3 flex-wrap">
                <span className="text-xs" style={{ color: TEXT_MUT }}>Try:</span>
                {[
                  ["NVDA","AMD","AAPL","MSFT","GOOGL"],
                  ["JPM","XOM","JNJ","AMZN"],
                ].map((ex, i) => (
                  <button key={i}
                    onClick={() => setTickers(ex)}
                    className="text-xs px-2.5 py-1 rounded-md transition-colors"
                    style={{ background: "hsl(215,25%,14%)", border: `1px solid ${BORDER}`, color: TEXT_SEC }}
                    onMouseEnter={e => (e.currentTarget.style.background = "hsl(215,25%,18%)")}
                    onMouseLeave={e => (e.currentTarget.style.background = "hsl(215,25%,14%)")}>
                    {ex.join(", ")}
                  </button>
                ))}
              </div>
            )}

            {/* Actions */}
            <div className="flex items-center gap-3 mt-4">
              <button
                onClick={handleAnalyze}
                disabled={loading || (tickers.length === 0 && !inputVal.trim())}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold transition-opacity disabled:opacity-40"
                style={{ background: BLUE, color: "white" }}
              >
                {loading
                  ? <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing...</>
                  : <><BarChart3 className="w-4 h-4" /> Analyze Portfolio</>}
              </button>
              {(tickers.length > 0 || hasResult) && (
                <button onClick={reset}
                  className="px-4 py-2.5 rounded-lg text-sm transition-colors"
                  style={{ background: "hsl(215,25%,14%)", border: `1px solid ${BORDER}`, color: TEXT_SEC }}
                  onMouseEnter={e => (e.currentTarget.style.background = "hsl(215,25%,18%)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "hsl(215,25%,14%)")}>
                  Reset
                </button>
              )}
            </div>
          </div>

          {/* ── Error ── */}
          {error && (
            <div className="rounded-xl px-5 py-4 mb-6 flex items-center gap-3"
              style={{ background: "hsla(0,84%,60%,0.1)", border: "1px solid hsla(0,84%,60%,0.3)" }}>
              <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: RED }} />
              <p className="text-sm" style={{ color: RED }}>{error}</p>
            </div>
          )}

          {/* ── Results ── */}
          {hasResult && result && (
            <>
              {/* Unknown tickers warning */}
              {result.unknown_tickers.length > 0 && (
                <div className="rounded-xl px-5 py-4 mb-5 flex items-center gap-3"
                  style={{ background: "hsla(38,92%,50%,0.08)", border: "1px solid hsla(38,92%,50%,0.25)" }}>
                  <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: AMBER }} />
                  <p className="text-sm" style={{ color: AMBER }}>
                    <span className="font-semibold">{result.unknown_tickers.join(", ")}</span>
                    {" "}not found in our signal universe — results based on remaining tickers.
                  </p>
                </div>
              )}

              {/* Summary row */}
              <div className="grid grid-cols-3 gap-4 mb-6">
                {[
                  {
                    label: "Stocks Analyzed",
                    value: result.tickers_analyzed.length.toString(),
                    color: BLUE,
                    icon: <BarChart3 className="w-4 h-4" />,
                  },
                  {
                    label: "Hidden Overlaps",
                    value: result.overlaps.length.toString(),
                    color: result.overlaps.length > 0 ? AMBER : GREEN,
                    icon: <ShieldAlert className="w-4 h-4" />,
                  },
                  {
                    label: "Recommendations",
                    value: result.recommendations.length.toString(),
                    color: GREEN,
                    icon: <Sparkles className="w-4 h-4" />,
                  },
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

              {/* ── Concentration Risk ── */}
              <section className="mb-8">
                <div className="flex items-center gap-2 mb-4">
                  <ShieldAlert className="w-4 h-4" style={{ color: AMBER }} />
                  <h2 className="text-base font-semibold" style={{ color: TEXT_PRI }}>
                    Hidden Concentration Risk
                  </h2>
                  <span className="text-xs px-2 py-0.5 rounded-full ml-1"
                    style={{ background: "hsla(38,92%,50%,0.15)", color: AMBER }}>
                    {result.overlaps.length} pair{result.overlaps.length !== 1 ? "s" : ""}
                  </span>
                </div>

                {result.overlaps.length === 0 ? (
                  <div className="rounded-xl px-6 py-8 text-center" style={CARD}>
                    <div className="w-10 h-10 rounded-full flex items-center justify-center mx-auto mb-3"
                      style={{ background: "hsla(142,71%,45%,0.15)" }}>
                      <TrendingUp className="w-5 h-5" style={{ color: GREEN }} />
                    </div>
                    <p className="text-sm font-medium mb-1" style={{ color: TEXT_PRI }}>
                      No significant overlaps found
                    </p>
                    <p className="text-xs" style={{ color: TEXT_SEC }}>
                      Your holdings don't show strong lead-lag dependencies — looking well diversified.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {result.overlaps.map((o, i) => (
                      <OverlapCard key={i} overlap={o} />
                    ))}
                  </div>
                )}
              </section>

              {/* ── Recommendations ── */}
              <section>
                <div className="flex items-center gap-2 mb-4">
                  <Sparkles className="w-4 h-4" style={{ color: BLUE }} />
                  <h2 className="text-base font-semibold" style={{ color: TEXT_PRI }}>
                    Recommended Additions
                  </h2>
                  <span className="text-xs px-2 py-0.5 rounded-full ml-1"
                    style={{ background: BLUE_DIM, color: BLUE }}>
                    {result.recommendations.length} stocks
                  </span>
                </div>

                <p className="text-xs mb-4 leading-relaxed" style={{ color: TEXT_MUT }}>
                  Scored on signal strength of lead-lag connections to your holdings, market
                  centrality, sector diversity, and portfolio coverage. Based on 15 years of
                  factor-adjusted return analysis.
                </p>

                {result.recommendations.length === 0 ? (
                  <div className="rounded-xl px-6 py-8 text-center" style={CARD}>
                    <TrendingDown className="w-8 h-8 mx-auto mb-3" style={{ color: TEXT_MUT }} />
                    <p className="text-sm" style={{ color: TEXT_SEC }}>
                      No recommendations found — try adding more tickers or lowering the signal threshold.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {result.recommendations.map((rec, i) => (
                      <RecCard key={rec.ticker} rec={rec} rank={i + 1} />
                    ))}
                  </div>
                )}
              </section>
            </>
          )}

          {/* ── Empty state ── */}
          {!hasResult && !loading && tickers.length === 0 && (
            <div className="rounded-xl px-6 py-12 text-center" style={CARD}>
              <div className="w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-4"
                style={{ background: BLUE_DIM }}>
                <Plus className="w-6 h-6" style={{ color: BLUE }} />
              </div>
              <p className="text-sm font-medium mb-1" style={{ color: TEXT_PRI }}>
                Add your stock holdings above
              </p>
              <p className="text-xs leading-relaxed max-w-xs mx-auto" style={{ color: TEXT_SEC }}>
                Enter tickers like AAPL, MSFT, JPM — we'll identify hidden relationships
                and suggest stocks that genuinely diversify your exposure.
              </p>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}
