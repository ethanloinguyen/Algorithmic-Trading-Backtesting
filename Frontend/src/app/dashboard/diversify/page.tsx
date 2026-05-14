// Frontend/src/app/dashboard/diversify/page.tsx
"use client";
import { useState, useRef, useEffect } from "react";
import { useAuth } from "@/src/app/context/AuthContext";
import Sidebar from "@/components/ui/Sidebar";
import SpiderChart from "@/components/ui/SpiderChart";
import SectorDonut from "@/components/ui/SectorDonut";
import StockModal, { type Stock } from "@/components/ui/StockModal";
import {
  analyzePortfolio,
  fetchStockSummaries,
  runRiskPipeline,
  type OverlapResult,
  type Recommendation,
  type IndependentRecommendation,
  type QualityRecommendation,
  type AnalysisMode,
  type RiskPipelineResult,
  type ClusteringPick,
} from "@/src/app/lib/api";
import {
  AlertTriangle, TrendingUp, Plus, X,
  ChevronDown, ChevronUp, Loader2, Sparkles, ArrowRight,
  ShieldAlert, BarChart3, Unlink, Info, Zap, Globe, Link2, Layers, Timer,
  Activity,
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

// ── Preset portfolios (from new version) ──────────────────────────────────────
const PRESETS = [
  { label: "Big Tech",   tickers: ["NVDA","AMD","AAPL","MSFT","GOOGL"] },
  { label: "Financials", tickers: ["JPM","GS","MS","BAC","V"] },
  { label: "Energy",     tickers: ["XOM","CVX","COP","SLB"] },
  { label: "Healthcare", tickers: ["JNJ","UNH","MRK","PFE","ABBV"] },
];

// ── Factor explanations — 5 quality factors ──────────────────────────────────
const FACTOR_EXPLANATIONS = [
  {
    label: "Momentum", weight: "35%", color: BLUE, Icon: TrendingUp,
    what: "6-month price return, rank-normalised across the filtered universe of ~800 quality stocks.",
    how: "Latest close divided by the closing price closest to 6 months ago (±7-day window), then percentile-ranked 0–100 across all stocks.",
    why: "Momentum is one of the most empirically validated equity factors. A high score means the stock has been outperforming peers recently — a signal of positive market sentiment and trend continuation.",
  },
  {
    label: "Fundamental Quality", weight: "25%", color: GREEN, Icon: BarChart3,
    what: "A blend of valuation (P/E ratio vs. sector peers) and size (log market cap), capturing 'quality at a reasonable price'.",
    how: "50% from within-sector P/E rank (lower P/E = higher score, capped 0–100) and 50% from log market-cap rank. Missing P/E defaults to the sector median.",
    why: "Fundamentally strong, reasonably valued companies have historically shown better long-term risk-adjusted returns than expensive or low-quality peers.",
  },
  {
    label: "Sector Diversity", weight: "20%", color: AMBER, Icon: Layers,
    what: "How much new sector exposure this stock adds relative to your current portfolio sector distribution.",
    how: "Inverted portfolio sector weight: a stock in a sector you have 0% exposure to scores 100; a stock in your most concentrated sector scores proportionally lower.",
    why: "Different sectors are driven by different macro factors (rates, oil, consumer spending). Adding an underrepresented sector reduces the chance a single economic event affects your whole portfolio.",
  },
  {
    label: "Volatility Fit", weight: "10%", color: "hsl(195,80%,50%)", Icon: Activity,
    what: "How closely this stock's annualised volatility matches the average volatility profile of your existing portfolio.",
    how: "Absolute difference between candidate annualised vol and portfolio average vol, rank-normalised — smallest difference scores 100.",
    why: "Adding a stock whose risk profile is compatible with your existing holdings avoids unintended volatility spikes. It's not about avoiding risk — it's about adding the type of risk you already understand.",
  },
  {
    label: "Market Centrality", weight: "10%", color: PURPLE, Icon: Globe,
    what: "How connected this stock is across the entire 2000-stock market network, not just to your holdings.",
    how: "Eigenvector centrality from the full lead-lag network, rank-normalised 0–100. Gracefully zeroed if centrality data is unavailable.",
    why: "Central stocks transmit and receive information efficiently across the market. They tend to be more liquid, more widely followed, and more reliably priced.",
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
            5 factors
          </span>
        </div>
        {open
          ? <ChevronUp className="w-4 h-4" style={{ color: TEXT_MUT }} />
          : <ChevronDown className="w-4 h-4" style={{ color: TEXT_MUT }} />}
      </button>
      {open && (
        <div className="px-5 pb-5" style={{ borderTop: `1px solid ${BORDER_D}` }}>
          <p className="text-xs mt-4 mb-5 leading-relaxed" style={{ color: TEXT_SEC }}>
            Each recommended stock receives a composite score from 0–100 built from five independent
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

// ── Overlap card — original design + expanded stats from new version ───────────
function OverlapCard({ overlap, companyNames, onTickerClick }: {
  overlap: OverlapResult;
  companyNames: Record<string, string>;
  onTickerClick: (ticker: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const same = overlap.sector_leader === overlap.sector_follower;
  return (
    <div className="rounded-xl overflow-hidden" style={CARD}>
      {/* Header row — original layout */}
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
              <span className="text-xs truncate" style={{ color: TEXT_MUT }}>
                {companyNames[overlap.ticker_leader]}
              </span>
            )}
            <ArrowRight className="w-3.5 h-3.5 flex-shrink-0" style={{ color: AMBER }} />
            <button className="text-sm font-bold hover:underline"
              style={{ color: TEXT_PRI }}
              onClick={e => { e.stopPropagation(); onTickerClick(overlap.ticker_follower); }}>
              {overlap.ticker_follower}
            </button>
            {companyNames[overlap.ticker_follower] && (
              <span className="text-xs truncate" style={{ color: TEXT_MUT }}>
                {companyNames[overlap.ticker_follower]}
              </span>
            )}
            <span className="text-xs ml-1 flex-shrink-0" style={{ color: TEXT_MUT }}>
              lag {overlap.best_lag}d
            </span>
            {same && (
              <span className="text-xs px-2 py-0.5 rounded-full flex-shrink-0"
                style={{ background: "hsla(38,92%,50%,0.12)", color: AMBER }}>Same sector</span>
            )}
          </div>
          <div className="w-28 flex-shrink-0"><SignalBar score={overlap.signal_strength} /></div>
          {open
            ? <ChevronUp className="w-4 h-4 flex-shrink-0" style={{ color: TEXT_MUT }} />
            : <ChevronDown className="w-4 h-4 flex-shrink-0" style={{ color: TEXT_MUT }} />}
        </div>
        {/* Sector tags row — original */}
        <div className="flex items-center gap-2 mt-2">
          <SectorTag sector={overlap.sector_leader} />
          <ArrowRight className="w-3 h-3" style={{ color: TEXT_MUT }} />
          <SectorTag sector={overlap.sector_follower} />
        </div>
      </div>
      {/* Expanded — 6 stats including new ones from new version */}
      {open && (
        <div className="px-5 pb-4" style={{ borderTop: `1px solid ${BORDER_D}` }}>
          <p className="text-sm mt-3 leading-relaxed" style={{ color: TEXT_SEC }}>
            {overlap.interpretation}
          </p>
          <div className="grid grid-cols-3 gap-3 mt-4">
            {[
              { label: "dCor",       val: overlap.mean_dcor.toFixed(3),          desc: "Distance correlation at best lag" },
              { label: "OOS Sharpe", val: overlap.oos_sharpe_net.toFixed(2),     desc: "Net out-of-sample Sharpe ratio" },
              { label: "Half-life",  val: `${Math.round(overlap.half_life)}d`,   desc: "Days until signal decays 50%" },
              { label: "Frequency",  val: `${Math.round(overlap.frequency * 100)}%`, desc: "% of 15-yr windows significant" },
              { label: "Sharpness",  val: overlap.sharpness.toFixed(2),          desc: "Signal concentration at one lag" },
              { label: "Best Lag",   val: `${overlap.best_lag}d`,                desc: "Lead-lag in trading days" },
            ].map(({ label, val, desc }) => (
              <div key={label} className="rounded-lg px-3 py-2 text-center"
                style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER_D}` }}>
                <p className="text-xs mb-0.5" style={{ color: TEXT_MUT }}>{label}</p>
                <p className="text-sm font-semibold" style={{ color: TEXT_PRI }}>{val}</p>
                <p className="text-xs mt-0.5" style={{ color: TEXT_MUT, fontSize: "10px" }}>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Independent rec card — original color + labelling fully restored ──────────
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
          {/* Colored rank badge — purple for top 3 */}
          <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold"
            style={{
              background: rank <= 3 ? PURPLE_DIM : "hsl(215,25%,16%)",
              color:      rank <= 3 ? PURPLE     : TEXT_SEC,
            }}>
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
              {/* Sector tag with sector color */}
              <SectorTag sector={rec.sector} />
              {/* "No overlap detected" pill — purple */}
              <span className="text-xs px-2 py-0.5 rounded-full"
                style={{ background: PURPLE_DIM, color: PURPLE }}>
                No overlap detected
              </span>
            </div>
            {/* Inline sub-scores row with icons — from original */}
            <div className="flex items-center gap-4 mt-1.5 flex-wrap">
              <span className="text-xs" style={{ color: TEXT_MUT }}>
                <Layers className="w-3 h-3 inline mr-1" style={{ color: PURPLE }} />
                Sector gap{" "}
                <span className="font-semibold" style={{ color: PURPLE }}>{pctGap}</span>
              </span>
              <span className="text-xs" style={{ color: TEXT_MUT }}>
                <Globe className="w-3 h-3 inline mr-1" style={{ color: TEXT_SEC }} />
                Centrality{" "}
                <span className="font-semibold" style={{ color: TEXT_SEC }}>{pctCent}</span>
              </span>
              <span className="text-xs" style={{ color: TEXT_MUT }}>
                Score{" "}
                <span className="font-semibold" style={{ color: PURPLE }}>
                  {rec.composite_score.toFixed(0)}
                </span>
              </span>
            </div>
          </div>
          {open
            ? <ChevronUp className="w-4 h-4 flex-shrink-0" style={{ color: TEXT_MUT }} />
            : <ChevronDown className="w-4 h-4 flex-shrink-0" style={{ color: TEXT_MUT }} />}
        </div>
      </div>
      {open && (
        <div className="px-5 pb-4" style={{ borderTop: `1px solid ${BORDER_D}` }}>
          <p className="text-sm mt-3 leading-relaxed" style={{ color: TEXT_SEC }}>{rec.reasoning}</p>
          <div className="grid grid-cols-2 gap-3 mt-4">
            {[
              { label: "Sector Gap",       val: rec.sector_gap_score,  desc: "How underrepresented this sector is in your portfolio (70% of score)", color: PURPLE },
              { label: "Market Centrality", val: rec.centrality_score, desc: "How central this stock is in the market network (30% of score)",        color: TEXT_SEC },
            ].map(({ label, val, desc, color }) => (
              <div key={label}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-semibold" style={{ color: TEXT_PRI }}>{label}</span>
                  <span className="text-xs font-bold" style={{ color }}>{Math.round(val)}/100</span>
                </div>
                <div className="h-1.5 rounded-full overflow-hidden mb-1" style={{ background: BORDER_D }}>
                  <div className="h-full rounded-full" style={{ width: `${val}%`, background: color }} />
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

// ── Risk Assessment panel ─────────────────────────────────────────────────────
function RiskResultPanel({ result, onTickerClick }: {
  result: RiskPipelineResult;
  onTickerClick: (ticker: string) => void;
}) {
  const { recommendations, risk } = result;
  const p = risk.portfolio;
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const lossColor = RED;
  const benefitColor = (v: number) => v > 0 ? GREEN : RED;

  const portfolioMetrics = [
    { label: "VaR 95%",             value: pct(p.var_95),                       color: lossColor,              hint: "Worst loss 95% of the time" },
    { label: "CVaR 95%",            value: pct(p.cvar_95),                      color: lossColor,              hint: "Avg loss in worst 5% of scenarios" },
    { label: "Expected Drawdown",   value: pct(p.expected_max_drawdown),        color: lossColor,              hint: "Average peak-to-trough decline" },
    { label: "Prob of Loss",        value: pct(p.prob_loss),                    color: p.prob_loss > 0.5 ? RED : AMBER, hint: "Probability of any loss at horizon" },
    { label: "Diversif. Benefit",   value: pct(p.diversification_benefit_95),   color: benefitColor(p.diversification_benefit_95), hint: "VaR improvement vs. equal-weight avg" },
    { label: "Worst-Case DD p95",   value: pct(p.worst_case_max_drawdown_p95),  color: lossColor,              hint: "Drawdown exceeded only 5% of paths" },
  ];

  return (
    <>
      {/* Portfolio risk metrics */}
      <div className="rounded-xl p-5 mb-4" style={CARD}>
        <p className="text-xs font-semibold mb-4 tracking-wide" style={{ color: TEXT_SEC }}>
          PORTFOLIO METRICS — {risk.horizon_days}-DAY HORIZON · {risk.n_simulations.toLocaleString()} PATHS
        </p>
        <div className="grid grid-cols-3 gap-3">
          {portfolioMetrics.map(({ label, value, color, hint }) => (
            <div key={label} className="rounded-lg p-3"
              style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER_D}` }}>
              <p className="text-xs mb-1" style={{ color: TEXT_MUT }}>{label}</p>
              <p className="text-xl font-bold" style={{ color }}>{value}</p>
              <p className="text-xs mt-1 leading-tight" style={{ color: TEXT_MUT }}>{hint}</p>
            </div>
          ))}
        </div>
      </div>

    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function DiversifyPage() {
  const { customPortfolios } = useAuth();
  const [analysisMode,  setAnalysisMode]  = useState<AnalysisMode>("broad_market");
  const [resultMode,    setResultMode]    = useState<AnalysisMode>("broad_market");
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
    quality_picks:               QualityRecommendation[];
    holdings_sectors:            Record<string, string>;
  } | null>(null);
  const [activeSpiderIdx, setActiveSpiderIdx] = useState<number | null>(0);
  const [companyNames,    setCompanyNames]    = useState<Record<string, string>>({});
  const [modalStock,      setModalStock]      = useState<Stock | null>(null);
  const [riskResult,      setRiskResult]      = useState<RiskPipelineResult | null>(null);
  const [riskLoading,     setRiskLoading]     = useState(false);
  const [riskError,       setRiskError]       = useState<string | null>(null);

  const hoveredTicker = activeSpiderIdx !== null
    ? (result?.quality_picks[activeSpiderIdx]?.ticker ?? null)
    : null;
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  // currentSectors from holdings_sectors — always populated, fixes blank donut
  const currentSectors = (() => {
    if (!result?.holdings_sectors) return {};
    const dist: Record<string, number> = {};
    Object.values(result.holdings_sectors).forEach(sector => {
      dist[sector] = (dist[sector] ?? 0) + 1;
    });
    return dist;
  })();

  const hoveredRec = result?.quality_picks.find(r => r.ticker === hoveredTicker) ?? null;

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
    setRiskResult(null); setRiskLoading(true); setRiskError(null);
    try {
      const data = await analyzePortfolio(toAnalyze, analysisMode);
      setResult(data);
      setResultMode(analysisMode);
      const allTickers = [
        ...data.tickers_analyzed,
        ...data.signal_recommendations.map(r => r.ticker),
        ...data.independent_recommendations.map(r => r.ticker),
        ...data.quality_picks.map(r => r.ticker),
      ];
      const uniqueTickers = [...new Set(allTickers)];
      if (uniqueTickers.length > 0) {
        fetchStockSummaries(uniqueTickers)
          .then(summaries => {
            const names: Record<string, string> = {};
            summaries.forEach(s => { names[s.symbol] = s.name; });
            setCompanyNames(names);
          })
          .catch(() => {});
      }
      // Fire risk pipeline in background — portfolio analysis results show immediately
      runRiskPipeline(toAnalyze)
        .then(riskData => setRiskResult(riskData))
        .catch(err => setRiskError(err instanceof Error ? err.message : "Risk assessment failed."))
        .finally(() => setRiskLoading(false));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setRiskLoading(false);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setTickers([]); setInputVal(""); setResult(null); setError(null);
    setActiveSpiderIdx(null); setCompanyNames({}); setModalStock(null);
    setRiskResult(null); setRiskLoading(false); setRiskError(null);
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  const hasResult = result !== null;

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

  const qualityPicks = result?.quality_picks ?? [];

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

          {/* ── Input — always visible, even after analysis ── */}
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

            {/* Preset buttons — always visible, new version style */}
            <div className="flex items-center gap-2 mt-3 flex-wrap">
              <span className="text-xs" style={{ color: TEXT_MUT }}>Presets:</span>
              {PRESETS.map(p => (
                <button key={p.label} onClick={() => { setTickers(p.tickers); setResult(null); }}
                  className="text-xs px-2.5 py-1 rounded-md transition-colors"
                  style={{ background: "hsl(215,25%,14%)", border: `1px solid ${BORDER}`, color: TEXT_SEC }}
                  onMouseEnter={e => (e.currentTarget.style.background = "hsl(215,25%,18%)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "hsl(215,25%,14%)")}>
                  {p.label}
                </button>
              ))}
              {customPortfolios.length > 0 && (
                <>
                  <span className="text-xs" style={{ color: "hsl(215,20%,28%)" }}>|</span>
                  <span className="text-xs" style={{ color: TEXT_MUT }}>My portfolios:</span>
                  {customPortfolios.map(p => (
                    <button
                      key={p.id}
                      onClick={() => { setTickers(p.tickers); setResult(null); }}
                      className="text-xs px-2.5 py-1 rounded-md transition-colors"
                      style={{ background: "hsla(217,91%,60%,0.1)", border: "1px solid hsla(217,91%,60%,0.3)", color: "hsl(217,91%,70%)" }}
                      onMouseEnter={e => (e.currentTarget.style.background = "hsla(217,91%,60%,0.2)")}
                      onMouseLeave={e => (e.currentTarget.style.background = "hsla(217,91%,60%,0.1)")}>
                      {p.name}
                    </button>
                  ))}
                </>
              )}
            </div>

            <div className="flex items-center gap-3 mt-4">
              <button onClick={handleAnalyze}
                disabled={loading || (tickers.length === 0 && !inputVal.trim())}
                className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold disabled:opacity-40"
                style={{ background: BLUE, color: "white" }}>
                {loading
                  ? <><Loader2 className="w-4 h-4 animate-spin" />Analyzing…</>
                  : <><Sparkles className="w-4 h-4" />Analyze Portfolio</>}
              </button>
              {hasResult && (
                <button onClick={reset}
                  className="text-xs px-3 py-2 rounded-lg"
                  style={{ background: "hsl(215,25%,16%)", border: `1px solid ${BORDER}`, color: TEXT_SEC }}
                  onMouseEnter={e => (e.currentTarget.style.background = "hsl(215,25%,20%)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "hsl(215,25%,16%)")}>
                  Clear results
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
            <div>
              {result.unknown_tickers.length > 0 && (
                <div className="rounded-xl px-5 py-3 mb-4 flex items-center gap-2"
                  style={{ background: "hsla(38,92%,50%,0.1)", border: "1px solid hsla(38,92%,50%,0.3)" }}>
                  <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: AMBER }} />
                  <p className="text-xs" style={{ color: AMBER }}>
                    Not found in our universe: <strong>{result.unknown_tickers.join(", ")}</strong>
                  </p>
                </div>
              )}

              {/* ── Stat cards — original design with icons fully restored ── */}
              <div className="grid grid-cols-4 gap-3 mb-6">
                {[
                  { label: "Analyzed",     value: result.tickers_analyzed.length,            color: BLUE,   icon: <BarChart3 className="w-4 h-4" /> },
                  { label: "Overlaps",     value: result.overlaps.length,                    color: result.overlaps.length > 0 ? AMBER : GREEN, icon: <ShieldAlert className="w-4 h-4" /> },
                  { label: "Quality Picks", value: result.quality_picks.length,               color: BLUE,   icon: <Sparkles className="w-4 h-4" /> },
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

              {/* Factor explanations — new version's layout */}
              <FactorExplanations />

              {/* Risk Assessment */}
              {(riskLoading || riskResult !== null || riskError !== null) && (
                <section className="mb-8">
                  <div className="flex items-center gap-2 mb-1">
                    <Activity className="w-4 h-4" style={{ color: GREEN }} />
                    <h2 className="text-base font-semibold" style={{ color: TEXT_PRI }}>Risk Assessment</h2>
                    {riskLoading && (
                      <span className="text-xs px-2 py-0.5 rounded-full ml-1 flex items-center gap-1"
                        style={{ background: BLUE_DIM, color: BLUE }}>
                        <Loader2 className="w-3 h-3 animate-spin" />running
                      </span>
                    )}
                  </div>
                  <p className="text-xs mb-4" style={{ color: TEXT_MUT }}>
                    K-Medoids sector picks decorrelated from your portfolio · Monte Carlo simulation ({riskResult?.risk.horizon_days ?? 63}-day horizon, {(riskResult?.risk.n_simulations ?? 1000).toLocaleString()} paths)
                  </p>

                  {riskLoading && !riskResult && (
                    <div className="rounded-xl px-6 py-10 text-center" style={CARD}>
                      <Loader2 className="w-8 h-8 mx-auto mb-3 animate-spin" style={{ color: BLUE }} />
                      <p className="text-sm font-medium mb-1" style={{ color: TEXT_PRI }}>Running clustering and simulation</p>
                      <p className="text-xs leading-relaxed max-w-xs mx-auto" style={{ color: TEXT_SEC }}>
                        Querying BigQuery, running K-Medoids sweep, and simulating Monte Carlo paths — typically 30–60 seconds
                      </p>
                    </div>
                  )}

                  {riskError && (
                    <div className="rounded-xl px-5 py-4 flex items-center gap-3"
                      style={{ background: "hsla(0,84%,60%,0.1)", border: "1px solid hsla(0,84%,60%,0.3)" }}>
                      <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: RED }} />
                      <p className="text-sm" style={{ color: RED }}>{riskError}</p>
                    </div>
                  )}

                  {riskResult && (
                    <RiskResultPanel
                      result={riskResult}
                      onTickerClick={handleTickerClick}
                    />
                  )}
                </section>
              )}

              {/* Quality Picks */}
              {qualityPicks.length > 0 && (
                <section className="mb-8">
                  <SectionHeader
                    icon={<Sparkles className="w-4 h-4" />}
                    label="Quality Picks" count={qualityPicks.length}
                    color={BLUE} dimColor={BLUE_DIM}
                    subtitle="Top stocks ranked by five portfolio-aware quality factors — click a ticker on the chart or in the legend to explore its full profile"
                  />
                  <div className="mb-4">
                    <SpiderChart
                      recommendations={qualityPicks}
                      activeIdx={activeSpiderIdx}
                      onActiveChange={setActiveSpiderIdx}
                      onTickerClick={handleTickerClick}
                      companyNames={companyNames}
                    />
                  </div>
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
                {result.independent_recommendations.length > 0 && (
                  <p className="text-xs mb-4 px-1 leading-relaxed" style={{ color: TEXT_MUT }}>
                    These stocks have no detected relationship to your holdings, so signal strength,
                    durability, and coverage are not applicable. Ranked purely by new sector exposure
                    and market network centrality.
                  </p>
                )}
                {result.independent_recommendations.length === 0 ? (
                  <div className="rounded-xl px-6 py-8 text-center" style={CARD}>
                    <p className="text-sm" style={{ color: TEXT_SEC }}>No independent stocks found.</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {result.independent_recommendations.map((rec, i) => (
                      <IndependentRecCard
                        key={rec.ticker} rec={rec} rank={i + 1}
                        companyNames={companyNames} onTickerClick={handleTickerClick}
                      />
                    ))}
                  </div>
                )}
              </section>

              {/* K-Medoids Sector Picks */}
              {riskResult && riskResult.recommendations.length > 0 && (
                <section className="mb-8">
                  <div className="flex items-center gap-2 mb-1">
                    <Sparkles className="w-4 h-4" style={{ color: GREEN }} />
                    <h2 className="text-base font-semibold" style={{ color: TEXT_PRI }}>K-Medoids Sector Picks</h2>
                    <span className="text-xs px-2 py-0.5 rounded-full ml-1"
                      style={{ background: GREEN_DIM, color: GREEN }}>
                      {riskResult.recommendations.length} picks
                    </span>
                  </div>
                  <p className="text-xs mb-4" style={{ color: TEXT_MUT }}>
                    Stocks selected via K-Medoids clustering — decorrelated from your portfolio by sector
                  </p>
                  <div className="grid grid-cols-3 gap-3">
                    {riskResult.recommendations.map((rec: ClusteringPick) => (
                      <div key={rec.stock} className="rounded-xl p-4 flex flex-col gap-2" style={CARD}>
                        <div className="flex items-center justify-between">
                          <button className="text-sm font-bold hover:underline" style={{ color: TEXT_PRI }}
                            onClick={() => handleTickerClick(rec.stock)}>
                            {rec.stock}
                            {rec.is_medoid && <span className="ml-1 text-xs" style={{ color: AMBER }}>★</span>}
                          </button>
                          <span className="text-xs px-1.5 py-0.5 rounded font-mono"
                            style={{ background: GREEN_DIM, color: GREEN }}>
                            {rec.avg_dcor_to_portfolio.toFixed(3)}
                          </span>
                        </div>
                        <SectorTag sector={rec.sector} />
                      </div>
                    ))}
                  </div>
                  {/* Per-stock risk table */}
                  <div className="rounded-xl overflow-hidden mt-4" style={CARD}>
                    <div className="px-5 py-3" style={{ borderBottom: `1px solid ${BORDER_D}` }}>
                      <p className="text-xs font-semibold tracking-wide" style={{ color: TEXT_SEC }}>PER-STOCK RISK</p>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr style={{ borderBottom: `1px solid ${BORDER_D}` }}>
                            {["Ticker", "Sector", "VaR 95%", "CVaR 95%", "Prob Loss", "Max Drawdown"].map(h => (
                              <th key={h} className="px-4 py-2 text-left font-medium" style={{ color: TEXT_MUT }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {riskResult.risk.tickers.map((ticker, i) => {
                            const s = riskResult.risk.per_stock[ticker];
                            if (!s) return null;
                            const rec = riskResult.recommendations.find((r: ClusteringPick) => r.stock === ticker);
                            const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
                            return (
                              <tr key={ticker}
                                style={{ borderBottom: i < riskResult.risk.tickers.length - 1 ? `1px solid ${BORDER_D}` : "none" }}
                                onMouseEnter={e => (e.currentTarget.style.background = CARD_H)}
                                onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                              >
                                <td className="px-4 py-2.5">
                                  <button className="font-bold hover:underline" style={{ color: TEXT_PRI }}
                                    onClick={() => handleTickerClick(ticker)}>
                                    {ticker}
                                  </button>
                                </td>
                                <td className="px-4 py-2.5">
                                  {rec && <SectorTag sector={rec.sector} />}
                                </td>
                                <td className="px-4 py-2.5 font-mono" style={{ color: RED }}>{pct(s.var_95)}</td>
                                <td className="px-4 py-2.5 font-mono" style={{ color: RED }}>{pct(s.cvar_95)}</td>
                                <td className="px-4 py-2.5 font-mono"
                                  style={{ color: s.prob_loss > 0.5 ? RED : AMBER }}>{pct(s.prob_loss)}</td>
                                <td className="px-4 py-2.5 font-mono" style={{ color: RED }}>{pct(s.expected_max_drawdown)}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {riskResult.risk.missing.length > 0 ? (
                    <p className="text-xs mt-3 px-1" style={{ color: TEXT_MUT }}>
                      ★ = cluster medoid &nbsp;·&nbsp; dcor badge = decorrelation from your portfolio (lower is better)
                      &nbsp;·&nbsp; {riskResult.risk.missing.length} ticker{riskResult.risk.missing.length > 1 ? "s" : ""} unavailable: {riskResult.risk.missing.join(", ")}
                    </p>
                  ) : (
                    <p className="text-xs mt-3 px-1" style={{ color: TEXT_MUT }}>
                      ★ = cluster medoid &nbsp;·&nbsp; dcor badge = decorrelation from your portfolio (lower is better)
                    </p>
                  )}
                </section>
              )}

            </div>
          )}

          {/* Empty state */}
          {!hasResult && !loading && tickers.length === 0 && (
            <div className="rounded-xl px-6 py-12 text-center" style={CARD}>
              <div className="w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-4"
                style={{ background: BLUE_DIM }}>
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

      {modalStock && (
        <StockModal stock={modalStock} onClose={() => setModalStock(null)} />
      )}
    </div>
  );
}
