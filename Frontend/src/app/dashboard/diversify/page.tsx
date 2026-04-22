// Frontend/src/app/dashboard/diversify/page.tsx
"use client";
import { useState, useRef, useEffect } from "react";
import Sidebar from "@/components/ui/Sidebar";
import SpiderChart from "@/components/ui/SpiderChart";
import SectorDonut from "@/components/ui/SectorDonut";
import StockModal, { type Stock } from "@/components/ui/StockModal";
import {
  analyzePortfolio,
  fetchStockSummaries,
  type OverlapResult,
  type Recommendation,
  type IndependentRecommendation,
} from "@/src/app/lib/api";
import {
  AlertTriangle, TrendingUp, Plus, X,
  ChevronDown, ChevronUp, Loader2, Sparkles, ArrowRight,
  ShieldAlert, BarChart3, Unlink, Info, Zap, Globe, Link2, Layers,
} from "lucide-react";

// ── Design tokens ─────────────────────────────────────────────────────────────
const BG         = "hsl(213, 27%, 7%)";
const CARD       = { background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" };
const CARD_H     = "hsl(215, 25%, 13%)";
const TEXT_PRI   = "hsl(210, 40%, 92%)";
const TEXT_SEC   = "hsl(215, 15%, 55%)";
const TEXT_MUT   = "hsl(215, 15%, 40%)";
const BLUE       = "hsl(217, 91%, 60%)";
const BLUE_DIM   = "hsla(217, 91%, 60%, 0.15)";
const GREEN      = "hsl(142, 71%, 45%)";
const GREEN_DIM  = "hsla(142, 71%, 45%, 0.15)";
const AMBER      = "hsl(38, 92%, 50%)";
const RED        = "hsl(0, 84%, 60%)";
const BORDER     = "hsl(215, 20%, 18%)";
const BORDER_D   = "hsl(215, 20%, 16%)";
const PURPLE     = "hsl(270, 70%, 65%)";
const PURPLE_DIM = "hsla(270, 70%, 65%, 0.15)";

const RANK_COLORS = [
  BLUE, GREEN, AMBER, PURPLE, "hsl(195,80%,50%)",
  "hsl(15,90%,58%)", "hsl(330,70%,60%)", "hsl(90,60%,50%)",
  "hsl(260,60%,65%)", "hsl(50,80%,55%)",
];

const SECTOR_COLORS: Record<string, string> = {
  Technology: BLUE, Financials: GREEN, Healthcare: PURPLE,
  Energy: AMBER, Consumer: "hsl(15, 90%, 58%)",
  Industrials: "hsl(195, 80%, 50%)", Utilities: "hsl(60, 70%, 50%)",
};
const sc = (s: string) => SECTOR_COLORS[s] ?? TEXT_SEC;

// ── Factor explanations ───────────────────────────────────────────────────────
const FACTOR_EXPLANATIONS = [
  {
    label: "Signal Strength", weight: "45%", color: BLUE, Icon: Zap,
    what: "How statistically robust and economically meaningful the lead-lag relationship is between the candidate stock and your holdings.",
    how: "Combines distance correlation (dCor) at the optimal lag, the out-of-sample Sharpe ratio of a trading strategy built on the signal, and how consistently the relationship appeared across 15 years of rolling windows.",
    why: "A high signal score means the relationship is not a statistical fluke — it appeared reliably over time and was historically exploitable.",
  },
  {
    label: "Market Centrality", weight: "20%", color: GREEN, Icon: Globe,
    what: "How connected this stock is across the entire 2000-stock market network, not just to your holdings.",
    how: "Eigenvector centrality from the directed lead-lag network — stocks that lead many others across multiple sectors score higher.",
    why: "A high-centrality stock acts as a market barometer. Its movements carry predictive information about many other assets, making it more informationally valuable to hold.",
  },
  {
    label: "Portfolio Coverage", weight: "15%", color: AMBER, Icon: Link2,
    what: "How many of your existing holdings this stock has a detected relationship with.",
    how: "Count of your holdings that appear in a significant lead-lag pair with this candidate, divided by the maximum coverage across all candidates.",
    why: "A stock that connects to four of your holdings is more efficient than one connecting to one — you gain informational coverage across more positions with a single addition.",
  },
  {
    label: "Sector Diversity", weight: "20%", color: PURPLE, Icon: Layers,
    what: "How much new sector exposure this stock adds relative to what you already own.",
    how: "A full bonus (100) if your portfolio has zero stocks in this sector. The bonus scales down proportionally as your sector concentration increases.",
    why: "Different sectors respond to different economic drivers. Adding an underrepresented sector reduces the risk that a single macro event affects your entire portfolio.",
  },
];

// ── Small helpers ─────────────────────────────────────────────────────────────
function SignalBar({ score }: { score: number }) {
  const pct   = Math.min(100, Math.max(0, score));
  const color = pct >= 80 ? GREEN : pct >= 65 ? BLUE : AMBER;
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: BORDER }}>
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-xs font-semibold w-8 text-right" style={{ color }}>{Math.round(pct)}</span>
    </div>
  );
}

function SectorTag({ sector }: { sector: string }) {
  return (
    <span className="text-xs px-2 py-0.5 rounded-full"
      style={{ background: `${sc(sector)}20`, color: sc(sector) }}>
      {sector}
    </span>
  );
}

function DirectionTag({ direction }: { direction: string }) {
  if (direction === "leads_your_holdings")
    return <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: GREEN_DIM, color: GREEN }}>Leads your stocks</span>;
  if (direction === "follows_your_holdings")
    return <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: BLUE_DIM, color: BLUE }}>Follows your stocks</span>;
  return <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: "hsla(38,92%,50%,0.15)", color: AMBER }}>Bidirectional</span>;
}

function SectionHeader({ icon, label, count, color, dimColor, subtitle }: {
  icon: React.ReactNode; label: string; count: number;
  color: string; dimColor: string; subtitle: string;
}) {
  return (
    <div className="mb-4">
      <div className="flex items-center gap-2 mb-1">
        <span style={{ color }}>{icon}</span>
        <h2 className="text-base font-semibold" style={{ color: TEXT_PRI }}>{label}</h2>
        <span className="text-xs px-2 py-0.5 rounded-full ml-1" style={{ background: dimColor, color }}>
          {count} stock{count !== 1 ? "s" : ""}
        </span>
      </div>
      <p className="text-xs" style={{ color: TEXT_MUT }}>{subtitle}</p>
    </div>
  );
}

// ── Factor explanations panel ─────────────────────────────────────────────────
function FactorExplanations() {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl overflow-hidden mb-6" style={CARD}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full px-5 py-4 flex items-center justify-between text-left"
        style={{ background: open ? CARD_H : "transparent" }}
      >
        <div className="flex items-center gap-2">
          <Info className="w-4 h-4 flex-shrink-0" style={{ color: BLUE }} />
          <span className="text-sm font-semibold" style={{ color: TEXT_PRI }}>
            How we score recommendations
          </span>
          <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: BLUE_DIM, color: BLUE }}>
            4 factors
          </span>
        </div>
        {open
          ? <ChevronUp className="w-4 h-4" style={{ color: TEXT_MUT }} />
          : <ChevronDown className="w-4 h-4" style={{ color: TEXT_MUT }} />}
      </button>
      {open && (
        <div className="px-5 pb-5" style={{ borderTop: `1px solid ${BORDER_D}` }}>
          <p className="text-xs mt-4 mb-5 leading-relaxed" style={{ color: TEXT_SEC }}>
            Each recommended stock receives a composite score from 0–100 built from four independent
            dimensions. The score tells you not just <em>what</em> to consider adding, but{" "}
            <em>why</em> — and which aspect of diversification it addresses.
          </p>
          <div className="grid grid-cols-2 gap-4">
            {FACTOR_EXPLANATIONS.map(f => (
              <div key={f.label} className="rounded-lg p-4"
                style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER_D}` }}>
                <div className="flex items-center gap-2 mb-2">
                  <f.Icon className="w-4 h-4 flex-shrink-0" style={{ color: f.color }} />
                  <span className="text-xs font-bold" style={{ color: f.color }}>{f.label}</span>
                  <span className="ml-auto text-xs px-1.5 py-0.5 rounded"
                    style={{ background: `${f.color}20`, color: f.color }}>
                    {f.weight}
                  </span>
                </div>
                <p className="text-xs mb-2 leading-relaxed font-medium" style={{ color: TEXT_PRI }}>{f.what}</p>
                <p className="text-xs mb-1.5 leading-relaxed" style={{ color: TEXT_SEC }}>
                  <span className="font-medium" style={{ color: TEXT_MUT }}>How: </span>{f.how}
                </p>
                <p className="text-xs leading-relaxed" style={{ color: TEXT_SEC }}>
                  <span className="font-medium" style={{ color: TEXT_MUT }}>Why: </span>{f.why}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Overlap card ──────────────────────────────────────────────────────────────
function OverlapCard({ overlap, companyNames, onTickerClick }: {
  overlap: OverlapResult;
  companyNames: Record<string, string>;
  onTickerClick: (ticker: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const same = overlap.sector_leader === overlap.sector_follower;
  return (
    <div className="rounded-xl overflow-hidden" style={CARD}>
      <div className="px-5 py-4 cursor-pointer" onClick={() => setOpen(o => !o)}
        onMouseEnter={e => (e.currentTarget.style.background = CARD_H)}
        onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <button className="text-sm font-bold hover:underline"
              style={{ color: TEXT_PRI }}
              onClick={e => { e.stopPropagation(); onTickerClick(overlap.ticker_leader); }}>
              {overlap.ticker_leader}
            </button>
            {companyNames[overlap.ticker_leader] && (
              <span className="text-xs" style={{ color: TEXT_MUT }}>{companyNames[overlap.ticker_leader]}</span>
            )}
            <ArrowRight className="w-3.5 h-3.5 flex-shrink-0" style={{ color: AMBER }} />
            <button className="text-sm font-bold hover:underline"
              style={{ color: TEXT_PRI }}
              onClick={e => { e.stopPropagation(); onTickerClick(overlap.ticker_follower); }}>
              {overlap.ticker_follower}
            </button>
            {companyNames[overlap.ticker_follower] && (
              <span className="text-xs" style={{ color: TEXT_MUT }}>{companyNames[overlap.ticker_follower]}</span>
            )}
            <span className="text-xs ml-1" style={{ color: TEXT_MUT }}>lag {overlap.best_lag}d</span>
            {same && (
              <span className="text-xs px-2 py-0.5 rounded-full"
                style={{ background: "hsla(38,92%,50%,0.12)", color: AMBER }}>Same sector</span>
            )}
          </div>
          <div className="w-28 flex-shrink-0"><SignalBar score={overlap.signal_strength} /></div>
          {open ? <ChevronUp className="w-4 h-4" style={{ color: TEXT_MUT }} /> : <ChevronDown className="w-4 h-4" style={{ color: TEXT_MUT }} />}
        </div>
        <div className="flex items-center gap-2 mt-2">
          <SectorTag sector={overlap.sector_leader} />
          <ArrowRight className="w-3 h-3" style={{ color: TEXT_MUT }} />
          <SectorTag sector={overlap.sector_follower} />
        </div>
      </div>
      {open && (
        <div className="px-5 pb-4" style={{ borderTop: `1px solid ${BORDER_D}` }}>
          <p className="text-sm mt-3 leading-relaxed" style={{ color: TEXT_SEC }}>{overlap.interpretation}</p>
          <div className="grid grid-cols-3 gap-3 mt-4">
            {[
              { label: "dCor",       val: overlap.mean_dcor.toFixed(3) },
              { label: "OOS Sharpe", val: overlap.oos_sharpe_net.toFixed(2) },
              { label: "Half-life",  val: `${Math.round(overlap.half_life)}d` },
            ].map(({ label, val }) => (
              <div key={label} className="rounded-lg px-3 py-2 text-center"
                style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER_D}` }}>
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


// ── Independent rec card (Group B) ───────────────────────────────────────────
function IndependentRecCard({ rec, rank, companyNames, onTickerClick }: {
  rec: IndependentRecommendation; rank: number;
  companyNames: Record<string, string>;
  onTickerClick: (ticker: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const pctCent = Math.round(rec.centrality_score);
  const pctGap  = Math.round(rec.sector_gap_score);
  return (
    <div className="rounded-xl overflow-hidden" style={CARD}>
      <div className="px-5 py-4 cursor-pointer" onClick={() => setOpen(o => !o)}
        onMouseEnter={e => (e.currentTarget.style.background = CARD_H)}
        onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold"
            style={{ background: rank <= 3 ? PURPLE_DIM : "hsl(215,25%,16%)", color: rank <= 3 ? PURPLE : TEXT_SEC }}>
            {rank}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <button className="text-sm font-bold hover:underline"
                style={{ color: TEXT_PRI }}
                onClick={e => { e.stopPropagation(); onTickerClick(rec.ticker); }}>
                {rec.ticker}
              </button>
              {companyNames[rec.ticker] && (
                <span className="text-xs" style={{ color: TEXT_MUT }}>{companyNames[rec.ticker]}</span>
              )}
              <SectorTag sector={rec.sector} />
              <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: PURPLE_DIM, color: PURPLE }}>
                No overlap detected
              </span>
            </div>
            {/* Two sub-scores inline */}
            <div className="flex items-center gap-4 mt-1.5">
              <span className="text-xs" style={{ color: TEXT_MUT }}>
                <Layers className="w-3 h-3 inline mr-1" style={{ color: PURPLE }} />Sector gap <span className="font-semibold" style={{ color: PURPLE }}>{pctGap}</span>
              </span>
              <span className="text-xs" style={{ color: TEXT_MUT }}>
                <Globe className="w-3 h-3 inline mr-1" style={{ color: TEXT_SEC }} />Centrality <span className="font-semibold" style={{ color: TEXT_SEC }}>{pctCent}</span>
              </span>
              <span className="text-xs" style={{ color: TEXT_MUT }}>
                Score <span className="font-semibold" style={{ color: PURPLE }}>{rec.composite_score.toFixed(0)}</span>
              </span>
            </div>
          </div>
          {open ? <ChevronUp className="w-4 h-4" style={{ color: TEXT_MUT }} /> : <ChevronDown className="w-4 h-4" style={{ color: TEXT_MUT }} />}
        </div>
      </div>
      {open && (
        <div className="px-5 pb-4" style={{ borderTop: `1px solid ${BORDER_D}` }}>
          <p className="text-sm mt-3 leading-relaxed" style={{ color: TEXT_SEC }}>{rec.reasoning}</p>
          <div className="grid grid-cols-2 gap-3 mt-4">
            {[
              { label: "Sector Gap",     val: rec.sector_gap_score,  desc: "How underrepresented this sector is in your portfolio (70% of score)" },
              { label: "Market Centrality",val: rec.centrality_score, desc: "How central this stock is in the market network (30% of score)" },
            ].map(({ label, val, desc }) => (
              <div key={label}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-semibold" style={{ color: TEXT_PRI }}>{label}</span>
                  <span className="text-xs font-bold" style={{ color: PURPLE }}>{Math.round(val)}/100</span>
                </div>
                <div className="h-1.5 rounded-full overflow-hidden mb-1" style={{ background: BORDER_D }}>
                  <div className="h-full rounded-full" style={{ width: `${val}%`, background: PURPLE }} />
                </div>
                <p className="text-xs" style={{ color: TEXT_MUT }}>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function DiversifyPage() {
  const [inputVal,      setInputVal]      = useState("");
  const [tickers,       setTickers]       = useState<string[]>([]);
  const [loading,       setLoading]       = useState(false);
  const [error,         setError]         = useState<string | null>(null);
  const [result,        setResult]        = useState<{
    tickers_analyzed:            string[];
    unknown_tickers:             string[];
    overlaps:                    OverlapResult[];
    signal_recommendations:      Recommendation[];
    independent_recommendations: IndependentRecommendation[];
  } | null>(null);
  // activeSpiderIdx is lifted from SpiderChart so it drives the sector donut preview.
  const [activeSpiderIdx, setActiveSpiderIdx] = useState<number | null>(0);
  // Company names fetched after analysis for display alongside tickers
  const [companyNames, setCompanyNames] = useState<Record<string, string>>({});
  // OHLCV modal state
  const [modalStock, setModalStock] = useState<Stock | null>(null);

  // Derived: which ticker is selected in the spider chart
  const hoveredTicker = activeSpiderIdx !== null
    ? (result?.signal_recommendations[activeSpiderIdx]?.ticker ?? null)
    : null;
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  // Derive sector distribution from overlaps (both sides are user holdings with sector info)
  const currentSectors = (() => {
    if (!result) return {};
    const dist: Record<string, number> = {};
    const seen = new Set<string>();
    result.overlaps.forEach(o => {
      if (!seen.has(o.ticker_leader))   { dist[o.sector_leader]  = (dist[o.sector_leader]  ?? 0) + 1; seen.add(o.ticker_leader); }
      if (!seen.has(o.ticker_follower)) { dist[o.sector_follower] = (dist[o.sector_follower] ?? 0) + 1; seen.add(o.ticker_follower); }
    });
    return dist;
  })();

  const hoveredRec = result?.signal_recommendations.find(r => r.ticker === hoveredTicker) ?? null;

  const addTicker = (raw: string) => {
    const parts = raw.toUpperCase().split(/[\s,]+/).filter(Boolean);
    setTickers(prev => {
      const next = [...prev];
      for (const t of parts) if (t.length <= 10 && !next.includes(t)) next.push(t);
      return next;
    });
    setInputVal("");
  };

  const removeTicker = (t: string) => setTickers(prev => prev.filter(x => x !== t));

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (["Enter", ",", " ", "Tab"].includes(e.key)) { e.preventDefault(); if (inputVal.trim()) addTicker(inputVal.trim()); }
    if (e.key === "Backspace" && !inputVal && tickers.length) setTickers(prev => prev.slice(0, -1));
  };

  const handleAnalyze = async () => {
    if (inputVal.trim()) addTicker(inputVal.trim());
    const toAnalyze = inputVal.trim()
      ? [...tickers, ...inputVal.trim().toUpperCase().split(/[\s,]+/).filter(Boolean)]
      : tickers;
    if (!toAnalyze.length) return;
    setLoading(true); setError(null); setResult(null); setActiveSpiderIdx(0);
    try {
      const data = await analyzePortfolio(toAnalyze);
      setResult(data);
      // Fetch company names for all tickers involved (holdings + recommendations)
      const allTickers = [
        ...data.tickers_analyzed,
        ...data.signal_recommendations.map(r => r.ticker),
        ...data.independent_recommendations.map(r => r.ticker),
      ];
      const uniqueTickers = [...new Set(allTickers)];
      if (uniqueTickers.length > 0) {
        fetchStockSummaries(uniqueTickers)
          .then(summaries => {
            const names: Record<string, string> = {};
            summaries.forEach(s => { names[s.symbol] = s.name; });
            setCompanyNames(names);
          })
          .catch(() => {}); // non-critical — names just won't show
      }
    }
    catch (err) { setError(err instanceof Error ? err.message : "Something went wrong."); }
    finally { setLoading(false); }
  };

  const reset = () => {
    setTickers([]); setInputVal(""); setResult(null); setError(null); setActiveSpiderIdx(null); setCompanyNames({}); setModalStock(null);
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  const hasResult  = result !== null;

  // Opens OHLCV modal for a ticker — uses live price data if available
  const handleTickerClick = (ticker: string) => {
    setModalStock({
      symbol:   ticker,
      name:     companyNames[ticker] ?? ticker,
      price:    "—",
      change:   "—",
      volume:   "—",
      positive: true,
    });
  };
  const signalRecs = result?.signal_recommendations ?? [];

  return (
    <div className="min-h-screen" style={{ background: BG }}>
      <Sidebar />
      <main className="pt-14">
        <div className="max-w-5xl mx-auto px-6 py-8">

          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: BLUE_DIM }}>
                <Sparkles className="w-4 h-4" style={{ color: BLUE }} />
              </div>
              <h1 className="text-xl font-bold" style={{ color: TEXT_PRI }}>Portfolio Diversifier</h1>
            </div>
            <p className="text-sm leading-relaxed" style={{ color: TEXT_SEC }}>
              Enter your holdings to uncover hidden lead-lag relationships and receive
              signal-backed diversification recommendations.
            </p>
          </div>

          {/* Input */}
          <div className="rounded-xl p-5 mb-6" style={CARD}>
            <p className="text-xs font-medium mb-3" style={{ color: TEXT_SEC }}>
              YOUR HOLDINGS — type a ticker and press Enter, Space, or comma
            </p>
            <div className="flex flex-wrap gap-2 min-h-12 p-2 rounded-lg cursor-text"
              style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER}` }}
              onClick={() => inputRef.current?.focus()}>
              {tickers.map(t => (
                <span key={t} className="flex items-center gap-1 px-2.5 py-1 rounded-md text-sm font-semibold"
                  style={{ background: BLUE_DIM, color: BLUE }}>
                  {t}
                  <button onClick={e => { e.stopPropagation(); removeTicker(t); }} className="hover:opacity-60 ml-0.5">
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
              <input ref={inputRef} value={inputVal}
                onChange={e => setInputVal(e.target.value.toUpperCase())}
                onKeyDown={handleKeyDown}
                onBlur={() => { if (inputVal.trim()) addTicker(inputVal.trim()); }}
                placeholder={tickers.length === 0 ? "AAPL, MSFT, NVDA..." : ""}
                className="flex-1 min-w-24 bg-transparent outline-none text-sm"
                style={{ color: TEXT_PRI }} />
            </div>

            {tickers.length === 0 && !result && (
              <div className="flex items-center gap-2 mt-3 flex-wrap">
                <span className="text-xs" style={{ color: TEXT_MUT }}>Try:</span>
                {[["NVDA","AMD","AAPL","MSFT","GOOGL"], ["JPM","XOM","JNJ","AMZN"]].map((ex, i) => (
                  <button key={i} onClick={() => setTickers(ex)}
                    className="text-xs px-2.5 py-1 rounded-md"
                    style={{ background: "hsl(215,25%,14%)", border: `1px solid ${BORDER}`, color: TEXT_SEC }}
                    onMouseEnter={e => (e.currentTarget.style.background = "hsl(215,25%,18%)")}
                    onMouseLeave={e => (e.currentTarget.style.background = "hsl(215,25%,14%)")}>
                    {ex.join(", ")}
                  </button>
                ))}
              </div>
            )}

            <div className="flex items-center gap-3 mt-4">
              <button onClick={handleAnalyze}
                disabled={loading || (tickers.length === 0 && !inputVal.trim())}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold disabled:opacity-40"
                style={{ background: BLUE, color: "white" }}>
                {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Analyzing...</> : <><BarChart3 className="w-4 h-4" />Analyze Portfolio</>}
              </button>
              {(tickers.length > 0 || hasResult) && (
                <button onClick={reset} className="px-4 py-2.5 rounded-lg text-sm"
                  style={{ background: "hsl(215,25%,14%)", border: `1px solid ${BORDER}`, color: TEXT_SEC }}
                  onMouseEnter={e => (e.currentTarget.style.background = "hsl(215,25%,18%)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "hsl(215,25%,14%)")}>
                  Reset
                </button>
              )}
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="rounded-xl px-5 py-4 mb-6 flex items-center gap-3"
              style={{ background: "hsla(0,84%,60%,0.1)", border: "1px solid hsla(0,84%,60%,0.3)" }}>
              <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: RED }} />
              <p className="text-sm" style={{ color: RED }}>{error}</p>
            </div>
          )}

          {/* Results */}
          {hasResult && result && (
            <>
              {result.unknown_tickers.length > 0 && (
                <div className="rounded-xl px-5 py-4 mb-5 flex items-center gap-3"
                  style={{ background: "hsla(38,92%,50%,0.08)", border: "1px solid hsla(38,92%,50%,0.25)" }}>
                  <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: AMBER }} />
                  <p className="text-sm" style={{ color: AMBER }}>
                    <span className="font-semibold">{result.unknown_tickers.join(", ")}</span>{" "}
                    not found in our signal universe.
                  </p>
                </div>
              )}

              {/* Stats */}
              <div className="grid grid-cols-4 gap-3 mb-6">
                {[
                  { label: "Analyzed",     value: result.tickers_analyzed.length,           color: BLUE,   icon: <BarChart3 className="w-4 h-4" /> },
                  { label: "Overlaps",     value: result.overlaps.length,                    color: result.overlaps.length > 0 ? AMBER : GREEN, icon: <ShieldAlert className="w-4 h-4" /> },
                  { label: "Signal Picks", value: result.signal_recommendations.length,      color: BLUE,   icon: <Sparkles className="w-4 h-4" /> },
                  { label: "Pure Picks",   value: result.independent_recommendations.length, color: PURPLE, icon: <Unlink className="w-4 h-4" /> },
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

              {/* Factor explanations */}
              <FactorExplanations />

              {/* Concentration Risk */}
              <section className="mb-8">
                <SectionHeader
                  icon={<ShieldAlert className="w-4 h-4" />}
                  label="Hidden Concentration Risk" count={result.overlaps.length}
                  color={AMBER} dimColor="hsla(38,92%,50%,0.15)"
                  subtitle="Lead-lag relationships detected within your existing holdings — you may be less diversified than you think"
                />
                {result.overlaps.length === 0 ? (
                  <div className="rounded-xl px-6 py-8 text-center" style={CARD}>
                    <TrendingUp className="w-8 h-8 mx-auto mb-3" style={{ color: GREEN }} />
                    <p className="text-sm font-medium mb-1" style={{ color: TEXT_PRI }}>No significant overlaps found</p>
                    <p className="text-xs" style={{ color: TEXT_SEC }}>Your holdings don't show strong lead-lag dependencies.</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {result.overlaps.map((o, i) => <OverlapCard key={i} overlap={o} companyNames={companyNames} onTickerClick={handleTickerClick} />)}
                  </div>
                )}
              </section>

              {/* Signal-connected recommendations */}
              {signalRecs.length > 0 && (
                <section className="mb-8">
                  <SectionHeader
                    icon={<Sparkles className="w-4 h-4" />}
                    label="Signal-Connected Recommendations" count={signalRecs.length}
                    color={BLUE} dimColor={BLUE_DIM}
                    subtitle="Stocks with detected lead-lag relationships to your holdings — click a ticker on the chart or in the list to explore its profile"
                  />

                  {/* ── Row 1: Spider chart — full width, activeIdx lifted ── */}
                  <div className="mb-4">
                    <SpiderChart
                      recommendations={signalRecs}
                      activeIdx={activeSpiderIdx}
                      onActiveChange={setActiveSpiderIdx}
                      onTickerClick={handleTickerClick}
                      companyNames={companyNames}
                    />
                  </div>

                  {/* ── Row 2: Sector donut — driven by spider chart selection ── */}
                  <div className="mb-6">
                    <SectorDonut
                      currentSectors={currentSectors}
                      previewSector={hoveredRec?.sector ?? null}
                      previewTicker={hoveredRec?.ticker ?? null}
                    />
                  </div>
                </section>
              )}

              {/* Independent recommendations */}
              <section className="mb-8">
                <SectionHeader
                  icon={<Unlink className="w-4 h-4" />}
                  label="Independent Recommendations" count={result.independent_recommendations.length}
                  color={PURPLE} dimColor={PURPLE_DIM}
                  subtitle="Stocks with zero detected lead-lag relationships to your holdings — ranked by sector gap (70%) and market centrality (30%)"
                />

                {/* One-line explanation of why no spider chart */}
                {result.independent_recommendations.length > 0 && (
                  <p className="text-xs mb-4 px-1 leading-relaxed" style={{ color: TEXT_MUT }}>
                    These stocks have no detected relationship to your holdings, so signal strength
                    and portfolio coverage are not applicable. They are ranked purely by how much
                    new sector exposure they add and how central they are in the market network.
                  </p>
                )}

                {result.independent_recommendations.length === 0 ? (
                  <div className="rounded-xl px-6 py-8 text-center" style={CARD}>
                    <p className="text-sm" style={{ color: TEXT_SEC }}>No independent stocks found.</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {result.independent_recommendations.map((rec, i) => (
                      <IndependentRecCard key={rec.ticker} rec={rec} rank={i + 1} companyNames={companyNames} onTickerClick={handleTickerClick} />
                    ))}
                  </div>
                )}
              </section>
            </>
          )}

          {/* Empty state */}
          {!hasResult && !loading && tickers.length === 0 && (
            <div className="rounded-xl px-6 py-12 text-center" style={CARD}>
              <div className="w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-4" style={{ background: BLUE_DIM }}>
                <Plus className="w-6 h-6" style={{ color: BLUE }} />
              </div>
              <p className="text-sm font-medium mb-1" style={{ color: TEXT_PRI }}>Add your stock holdings above</p>
              <p className="text-xs leading-relaxed max-w-xs mx-auto" style={{ color: TEXT_SEC }}>
                We'll identify hidden relationships between your stocks and recommend two types of
                additions: signal-connected picks for early warning, and truly independent picks
                for pure diversification.
              </p>
            </div>
          )}

        </div>
      </main>

      {/* OHLCV modal — opened when user clicks a ticker name */}
      {modalStock && (
        <StockModal stock={modalStock} onClose={() => setModalStock(null)} />
      )}
    </div>
  );
}
