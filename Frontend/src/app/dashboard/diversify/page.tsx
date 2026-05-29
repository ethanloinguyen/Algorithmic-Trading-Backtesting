// Frontend/src/app/dashboard/diversify/page.tsx
"use client";
import { useState, useRef, useEffect, useCallback } from "react";
import { useAuth } from "@/src/app/context/AuthContext";
import Sidebar from "@/components/ui/Sidebar";
import SpiderChart from "@/components/ui/SpiderChart";
import SectorDonut from "@/components/ui/SectorDonut";
import StockModal, { type Stock } from "@/components/ui/StockModal";
import {
  analyzePortfolio,
  fetchStockSummaries,
  fetchAllStocks,
  runPortfolioRiskAssessment,
  runClusteringPipeline,
  type OverlapResult,
  type Recommendation,
  type IndependentRecommendation,
  type QualityRecommendation,
  type AnalysisMode,
  type PortfolioRiskResult,
  type ClusteringPipelineResult,
  type ClusteringPick,
  type StockSummary,
} from "@/src/app/lib/api";
import {
  AlertTriangle, TrendingUp, Plus, X,
  ChevronDown, ChevronUp, Loader2, Sparkles, ArrowRight,
  ShieldAlert, BarChart3, Info, Zap, Globe, Link2, Layers, Timer,
  Activity, Star, Briefcase,
} from "lucide-react";
import { PageHelp } from "@/components/ui/PageHelp";

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

const SECTOR_COLORS: Record<string, string> = {
  Technology: BLUE, Financials: GREEN, Healthcare: PURPLE,
  Energy: AMBER, Consumer: "hsl(15, 90%, 58%)",
  Industrials: "hsl(195, 80%, 50%)", Utilities: "hsl(60, 70%, 50%)",
};
const sc = (s: string) => SECTOR_COLORS[s] ?? TEXT_SEC;

const PRESETS = [
  { label: "Big Tech",   tickers: ["NVDA","AMD","AAPL","MSFT","GOOGL"] },
  { label: "Financials", tickers: ["JPM","GS","MS","BAC","V"] },
  { label: "Energy",     tickers: ["XOM","CVX","COP","SLB"] },
  { label: "Healthcare", tickers: ["JNJ","UNH","MRK","PFE"] },
];

const FACTOR_EXPLANATIONS = [
  {
    label: "Momentum", weight: "35%", color: BLUE, Icon: TrendingUp,
    what: "6-month price return, rank-normalised across the filtered universe of ~800 quality stocks.",
    how: "Latest close divided by the closing price closest to 6 months ago (±7-day window), then percentile-ranked 0–100 across all stocks.",
    why: "Momentum is one of the most empirically validated equity factors. A high score means the stock has been outperforming peers recently, showing a signal of positive market sentiment and trend continuation.",
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
    how: "Absolute difference between candidate annualised vol and portfolio average vol, rank-normalised; smallest difference scores 100.",
    why: "Adding a stock whose risk profile is compatible with your existing holdings avoids unintended volatility spikes. A high score here means the stock has a similar volatility to your current portfolio, making it easier to integrate without needing to adjust position sizes drastically.",
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
            <button className="text-sm font-bold hover:underline" style={{ color: TEXT_PRI }}
              onClick={e => { e.stopPropagation(); onTickerClick(overlap.ticker_leader); }}>
              {overlap.ticker_leader}
            </button>
            {companyNames[overlap.ticker_leader] && (
              <span className="text-xs truncate" style={{ color: TEXT_MUT }}>
                {companyNames[overlap.ticker_leader]}
              </span>
            )}
            <ArrowRight className="w-3.5 h-3.5 flex-shrink-0" style={{ color: AMBER }} />
            <button className="text-sm font-bold hover:underline" style={{ color: TEXT_PRI }}
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
        <div className="flex items-center gap-2 mt-2">
          <SectorTag sector={overlap.sector_leader} />
          <ArrowRight className="w-3 h-3" style={{ color: TEXT_MUT }} />
          <SectorTag sector={overlap.sector_follower} />
        </div>
      </div>
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
            style={{
              background: rank <= 3 ? PURPLE_DIM : "hsl(215,25%,16%)",
              color:      rank <= 3 ? PURPLE     : TEXT_SEC,
            }}>
            {rank}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <button className="text-sm font-bold hover:underline" style={{ color: TEXT_PRI }}
                onClick={e => { e.stopPropagation(); onTickerClick(rec.ticker); }}>
                {rec.ticker}
              </button>
              {companyNames[rec.ticker] && (
                <span className="text-xs" style={{ color: TEXT_MUT }}>{companyNames[rec.ticker]}</span>
              )}
              <SectorTag sector={rec.sector} />
              <span className="text-xs px-2 py-0.5 rounded-full"
                style={{ background: PURPLE_DIM, color: PURPLE }}>
                No overlap detected
              </span>
            </div>
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
              { label: "Sector Gap",        val: rec.sector_gap_score,  desc: "How underrepresented this sector is (70% of score)", color: PURPLE },
              { label: "Market Centrality", val: rec.centrality_score,  desc: "How central this stock is in the market network (30%)", color: TEXT_SEC },
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

// ── Metric info tooltip ───────────────────────────────────────────────────────
function MetricHint({ detail, position = "above" }: { detail: string; position?: "above" | "below" }) {
  const [visible, setVisible] = useState(false);
  return (
    <span
      className="relative inline-flex items-center"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      <Info
        className="w-3 h-3 ml-1 flex-shrink-0 cursor-help"
        style={{ color: visible ? TEXT_SEC : TEXT_MUT }}
      />
      {visible && (
        <span
          className="absolute z-50 rounded-lg text-xs leading-relaxed pointer-events-none"
          style={{
            background:  "hsl(215,25%,11%)",
            border:      `1px solid ${BORDER}`,
            boxShadow:   "0 8px 28px rgba(0,0,0,0.55)",
            color:       TEXT_SEC,
            padding:     "10px 12px",
            width:       252,
            left:        "50%",
            transform:   "translateX(-50%)",
            ...(position === "above"
              ? { bottom: "calc(100% + 8px)" }
              : { top:    "calc(100% + 8px)" }),
          }}
        >
          {detail}
        </span>
      )}
    </span>
  );
}

// ── Detailed explanations for every risk metric ───────────────────────────────
const METRIC_DETAILS: Record<string, string> = {
  "VaR 95%":
    "Value at Risk — with 95% confidence your portfolio will not lose more than this percentage over the simulation horizon. Only 1-in-20 paths produce a larger loss. It marks the boundary between normal volatility and tail-risk territory.",
  "CVaR 95%":
    "Conditional Value at Risk (Expected Shortfall) — given that losses do cross the VaR boundary, this is their average severity across the worst 5% of paths. CVaR penalises fat tails more than VaR and is the preferred measure for capturing extreme events.",
  "Expected Drawdown":
    "The mean peak-to-trough decline across all simulation paths. A drawdown measures how far the portfolio falls from its running high before recovering. This is a 'typical bad stretch' estimate, not the worst case.",
  "Prob of Loss":
    "The fraction of Monte Carlo paths where the portfolio closes below its starting value at the horizon date. Above 50% means a loss is statistically more likely than a gain. Pair this with CVaR to understand both the likelihood and severity of losses.",
  "Diversif. Benefit":
    "How much your portfolio's VaR improves relative to a simple equal-weighted average of each stock's individual VaR. Positive means correlations are reducing joint risk — diversification is working. Negative means the holdings are moving too closely together to offset one another.",
  "Worst-Case DD p95":
    "The 95th-percentile peak-to-trough drawdown — 95% of simulated paths had a smaller decline than this value. Think of it as a stress-test drawdown: uncommon under normal conditions but firmly within the model's plausible range.",
  // Per-stock table columns
  "VaR 95% (stock)":
    "The maximum loss this individual stock is expected to exceed only 5% of the time, measured in isolation over the simulation horizon.",
  "CVaR 95% (stock)":
    "The average loss this stock experiences across its worst 5% of simulation paths — a direct measure of tail severity for a single position.",
  "Prob Loss":
    "The percentage of simulation paths where this stock finishes below its starting price at the horizon date.",
  "Max Drawdown":
    "The average peak-to-trough decline for this stock across all simulation paths, representing typical downside exposure.",
};

function RiskResultPanel({ result, onTickerClick }: {
  result: PortfolioRiskResult;
  onTickerClick: (ticker: string) => void;
}) {
  const { risk } = result;
  const p = risk.portfolio;
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const lossColor = RED;
  const benefitColor = (v: number) => v > 0 ? GREEN : RED;

  return (
    <div className="rounded-xl p-5 mb-4" style={CARD}>
      <p className="text-xs font-semibold mb-4 tracking-wide" style={{ color: TEXT_SEC }}>
        PORTFOLIO METRICS — {risk.horizon_days}-DAY HORIZON · {risk.n_simulations.toLocaleString()} PATHS
      </p>
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "VaR 95%",           value: pct(p.var_95),                      color: lossColor,              hint: "Worst loss 95% of the time" },
          { label: "CVaR 95%",          value: pct(p.cvar_95),                     color: lossColor,              hint: "Avg loss in worst 5% of scenarios" },
          { label: "Expected Drawdown", value: pct(p.expected_max_drawdown),       color: lossColor,              hint: "Average peak-to-trough decline" },
          { label: "Prob of Loss",      value: pct(p.prob_loss),                   color: p.prob_loss > 0.5 ? RED : AMBER, hint: "Probability of any loss at horizon" },
          { label: "Diversif. Benefit", value: pct(p.diversification_benefit_95),  color: benefitColor(p.diversification_benefit_95), hint: "VaR improvement vs. equal-weight avg" },
          { label: "Worst-Case DD p95", value: pct(p.worst_case_max_drawdown_p95), color: lossColor,              hint: "Drawdown exceeded only 5% of paths" },
        ].map(({ label, value, color, hint }) => (
          <div key={label} className="rounded-lg p-3"
            style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER_D}` }}>
            <p className="text-xs mb-1 flex items-center" style={{ color: TEXT_MUT }}>
              {label}
              {METRIC_DETAILS[label] && <MetricHint detail={METRIC_DETAILS[label]} />}
            </p>
            <p className="text-xl font-bold" style={{ color }}>{value}</p>
            <p className="text-xs mt-1 leading-tight" style={{ color: TEXT_MUT }}>{hint}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Ticker input with autocomplete ────────────────────────────────────────────

function TickerInput({
  tickers,
  onAdd,
  onRemove,
  allStocks,
}: {
  tickers:   string[];
  onAdd:     (ticker: string) => void;
  onRemove:  (ticker: string) => void;
  allStocks: StockSummary[];
}) {
  const [inputVal,     setInputVal]     = useState("");
  const [suggestions,  setSuggestions]  = useState<StockSummary[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [highlighted,  setHighlighted]  = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  useEffect(() => {
    const q = inputVal.toUpperCase().trim();
    if (!q || allStocks.length === 0) { setSuggestions([]); setShowDropdown(false); return; }
    const matches = allStocks
      .filter(s =>
        !tickers.includes(s.symbol) &&
        (s.symbol.startsWith(q) || s.name.toUpperCase().includes(q))
      )
      .slice(0, 8);
    setSuggestions(matches);
    setShowDropdown(matches.length > 0);
    setHighlighted(-1);
  }, [inputVal, allStocks, tickers]);

  const commitRaw = (raw: string) => {
    const parts = raw.toUpperCase().split(/[\s,]+/).filter(Boolean);
    parts.forEach(t => { if (t.length <= 10 && !tickers.includes(t)) onAdd(t); });
    setInputVal("");
    setSuggestions([]);
    setShowDropdown(false);
  };

  const commitSuggestion = (symbol: string) => {
    if (!tickers.includes(symbol)) onAdd(symbol);
    setInputVal("");
    setSuggestions([]);
    setShowDropdown(false);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (showDropdown && suggestions.length > 0) {
      if (e.key === "ArrowDown") { e.preventDefault(); setHighlighted(h => Math.min(h + 1, suggestions.length - 1)); return; }
      if (e.key === "ArrowUp")   { e.preventDefault(); setHighlighted(h => Math.max(h - 1, -1)); return; }
      if (e.key === "Enter" && highlighted >= 0) { e.preventDefault(); commitSuggestion(suggestions[highlighted].symbol); return; }
      if (e.key === "Escape") { setShowDropdown(false); return; }
    }
    if (["Enter", ",", " ", "Tab"].includes(e.key)) {
      e.preventDefault();
      if (inputVal.trim()) commitRaw(inputVal.trim());
    }
    if (e.key === "Backspace" && !inputVal && tickers.length) onRemove(tickers[tickers.length - 1]);
  };

  return (
    <div className="relative">
      <div
        className="flex flex-wrap gap-2 min-h-12 p-2 rounded-lg cursor-text"
        style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER}` }}
        onClick={() => inputRef.current?.focus()}
      >
        {tickers.map(t => (
          <span key={t} className="flex items-center gap-1 px-2.5 py-1 rounded-md text-sm font-semibold"
            style={{ background: BLUE_DIM, color: BLUE }}>
            {t}
            <button onClick={e => { e.stopPropagation(); onRemove(t); }} className="hover:opacity-60 ml-0.5">
              <X className="w-3 h-3" />
            </button>
          </span>
        ))}
        <input
          ref={inputRef}
          value={inputVal}
          onChange={e => setInputVal(e.target.value.toUpperCase())}
          onKeyDown={handleKeyDown}
          onFocus={() => suggestions.length > 0 && setShowDropdown(true)}
          onBlur={() => {
            setTimeout(() => {
              setShowDropdown(false);
              if (inputVal.trim()) commitRaw(inputVal.trim());
            }, 150);
          }}
          placeholder={tickers.length === 0 ? "AAPL, MSFT, NVDA…" : ""}
          className="flex-1 min-w-24 bg-transparent outline-none text-sm"
          style={{ color: TEXT_PRI }}
        />
      </div>

      {/* Autocomplete dropdown */}
      {showDropdown && suggestions.length > 0 && (
        <div
          className="absolute left-0 right-0 mt-1 rounded-xl overflow-hidden z-30"
          style={{
            background: "hsl(215,25%,11%)",
            border:     `1px solid ${BORDER}`,
            boxShadow:  "0 8px 24px rgba(0,0,0,0.4)",
            top:        "100%",
          }}
        >
          {suggestions.map((s, i) => (
            <button
              key={s.symbol}
              onMouseDown={() => commitSuggestion(s.symbol)}
              className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-left"
              style={{
                background:   i === highlighted ? "hsl(215,25%,16%)" : "transparent",
                borderBottom: i < suggestions.length - 1 ? `1px solid ${BORDER_D}` : "none",
              }}
              onMouseEnter={() => setHighlighted(i)}
              onMouseLeave={() => setHighlighted(-1)}
            >
              <span className="font-semibold" style={{ color: BLUE, minWidth: 56 }}>{s.symbol}</span>
              <span className="truncate text-xs" style={{ color: TEXT_SEC }}>{s.name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Portfolio dropdown ─────────────────────────────────────────────────────────

function PortfolioDropdown({
  portfolios,
  onSelect,
}: {
  portfolios: { id: string; name: string; tickers: string[] }[];
  onSelect:   (tickers: string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md transition-colors"
        style={{ background: "hsla(142,71%,45%,0.1)", border: "1px solid hsla(142,71%,45%,0.3)", color: GREEN }}
        onMouseEnter={e => (e.currentTarget.style.background = "hsla(142,71%,45%,0.2)")}
        onMouseLeave={e => (e.currentTarget.style.background = "hsla(142,71%,45%,0.1)")}
      >
        <Briefcase className="w-3 h-3" />
        Add Portfolio
        <ChevronDown className="w-3 h-3" style={{ transform: open ? "rotate(180deg)" : "none", transition: "transform 0.15s" }} />
      </button>

      {open && (
        <div
          className="absolute top-full left-0 mt-1 rounded-xl overflow-hidden z-30 min-w-48"
          style={{ background: "hsl(215,28%,13%)", border: `1px solid ${BORDER}`, boxShadow: "0 8px 32px rgba(0,0,0,0.5)" }}
        >
          {portfolios.length === 0 ? (
            <div className="px-4 py-3 text-xs" style={{ color: TEXT_MUT }}>
              No portfolios saved yet.
            </div>
          ) : (
            portfolios.map(p => (
              <button
                key={p.id}
                onMouseDown={() => { onSelect(p.tickers); setOpen(false); }}
                className="w-full text-left px-4 py-2.5 flex items-center justify-between gap-3 text-sm"
                style={{ color: TEXT_PRI }}
                onMouseEnter={e => (e.currentTarget.style.background = "hsl(215,25%,18%)")}
                onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
              >
                <span className="font-medium">{p.name}</span>
                <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: BLUE_DIM, color: BLUE }}>
                  {p.tickers.length}
                </span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

// ── Add-to-portfolio dropdown button ──────────────────────────────────────────

function AddToPortfolioButton({ ticker, portfolios, onAdd }: {
  ticker:     string;
  portfolios: { id: string; name: string; tickers: string[] }[];
  onAdd:      (portfolioId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div ref={ref} className="relative" onClick={e => e.stopPropagation()}>
      <button
        onClick={() => setOpen(o => !o)}
        className="transition-all hover:scale-110"
        title="Add to portfolio"
        aria-label="Add to portfolio"
      >
        <Briefcase className="w-3.5 h-3.5" style={{ color: open ? BLUE : TEXT_MUT }} />
      </button>
      {open && (
        <div
          className="absolute z-50 rounded-xl overflow-hidden min-w-40"
          style={{
            background:  "hsl(215,28%,13%)",
            border:      `1px solid ${BORDER}`,
            boxShadow:   "0 8px 32px rgba(0,0,0,0.5)",
            top:         "calc(100% + 4px)",
            right:       0,
          }}
        >
          {portfolios.length === 0 ? (
            <div className="px-4 py-3 text-xs" style={{ color: TEXT_MUT }}>
              No portfolios saved yet.
            </div>
          ) : (
            portfolios.map((p, i) => {
              const alreadyIn = p.tickers.includes(ticker);
              return (
                <button
                  key={p.id}
                  onMouseDown={() => { onAdd(p.id); setOpen(false); }}
                  className="w-full text-left px-4 py-2.5 flex items-center justify-between gap-3 text-xs"
                  style={{
                    color:       alreadyIn ? BLUE : TEXT_PRI,
                    borderTop:   i > 0 ? `1px solid ${BORDER_D}` : "none",
                  }}
                  onMouseEnter={e => (e.currentTarget.style.background = "hsl(215,25%,18%)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                >
                  <span className="font-medium truncate">{p.name}</span>
                  {alreadyIn && <span style={{ color: BLUE }}>✓</span>}
                </button>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function DiversifyPage() {
  const { savedStocks, customPortfolios, isSaved, toggleSave, updatePortfolio } = useAuth();

  const [analysisMode,  setAnalysisMode]  = useState<AnalysisMode>("broad_market");
  const [resultMode,    setResultMode]    = useState<AnalysisMode>("broad_market");
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
  const [portfolioRiskResult,  setPortfolioRiskResult]  = useState<PortfolioRiskResult | null>(null);
  const [portfolioRiskLoading, setPortfolioRiskLoading] = useState(false);
  const [portfolioRiskError,   setPortfolioRiskError]   = useState<string | null>(null);
  const [clusteringResult,     setClusteringResult]     = useState<ClusteringPipelineResult | null>(null);
  const [clusteringLoading,    setClusteringLoading]    = useState(false);
  const [clusteringError,      setClusteringError]      = useState<string | null>(null);
  const [allStocks,       setAllStocks]       = useState<StockSummary[]>([]);

  // Load autocomplete stock list
  useEffect(() => {
    fetchAllStocks().then(setAllStocks).catch(() => {});
  }, []);

  const hoveredTicker = activeSpiderIdx !== null
    ? (result?.quality_picks?.[activeSpiderIdx]?.ticker ?? null)
    : null;

  const currentSectors = (() => {
    if (!result?.holdings_sectors) return {};
    const dist: Record<string, number> = {};
    Object.values(result.holdings_sectors).forEach(sector => {
      dist[sector] = (dist[sector] ?? 0) + 1;
    });
    return dist;
  })();

  const hoveredRec = result?.quality_picks?.find(r => r.ticker === hoveredTicker) ?? null;

  const addTicker = useCallback((ticker: string) => {
    setTickers(prev => {
      if (prev.includes(ticker) || ticker.length > 10) return prev;
      return [...prev, ticker.toUpperCase()];
    });
  }, []);

  const removeTicker = (t: string) => setTickers(prev => prev.filter(x => x !== t));

  // Add entire watchlist
  const addWatchlist = () => {
    if (!savedStocks.length) return;
    setTickers(prev => {
      const next = [...prev];
      savedStocks.forEach(s => {
        if (!next.includes(s.symbol)) next.push(s.symbol);
      });
      return next;
    });
    setResult(null);
  };

  // Add a named portfolio
  const addPortfolio = (portfolioTickers: string[]) => {
    setTickers(prev => {
      const next = [...prev];
      portfolioTickers.forEach(t => {
        if (!next.includes(t)) next.push(t);
      });
      return next;
    });
    setResult(null);
  };

  const handleAnalyze = async () => {
    if (!tickers.length) return;
    setLoading(true); setError(null); setResult(null); setActiveSpiderIdx(0);
    setPortfolioRiskResult(null); setPortfolioRiskLoading(true); setPortfolioRiskError(null);
    setClusteringResult(null); setClusteringLoading(true); setClusteringError(null);
    try {
      const data = await analyzePortfolio(tickers, analysisMode);
      setResult(data);
      setResultMode(analysisMode);
      const allT = [
        ...data.tickers_analyzed,
        ...data.signal_recommendations.map(r => r.ticker),
        ...(data.quality_picks ?? []).map(r => r.ticker),
      ];
      const unique = [...new Set(allT)];
      if (unique.length > 0) {
        fetchStockSummaries(unique)
          .then(summaries => {
            const names: Record<string, string> = {};
            summaries.forEach(s => { names[s.symbol] = s.name; });
            setCompanyNames(names);
          })
          .catch(() => {});
      }
      if (data.tickers_analyzed.length > 0) {
        // Fire both pipeline calls in parallel — portfolio risk resolves first.
        runPortfolioRiskAssessment(data.tickers_analyzed)
          .then(r => setPortfolioRiskResult(r))
          .catch(err => setPortfolioRiskError(err instanceof Error ? err.message : "Risk assessment failed."))
          .finally(() => setPortfolioRiskLoading(false));

        runClusteringPipeline(data.tickers_analyzed)
          .then(r => setClusteringResult(r))
          .catch(err => setClusteringError(err instanceof Error ? err.message : "K-Medoids clustering failed."))
          .finally(() => setClusteringLoading(false));
      } else {
        const msg = "No recognized tickers — risk assessment requires at least one known stock.";
        setPortfolioRiskError(msg); setPortfolioRiskLoading(false);
        setClusteringError(msg);    setClusteringLoading(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setPortfolioRiskLoading(false);
      setClusteringLoading(false);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setTickers([]); setResult(null); setError(null);
    setActiveSpiderIdx(null); setCompanyNames({}); setModalStock(null);
    setPortfolioRiskResult(null); setPortfolioRiskLoading(false); setPortfolioRiskError(null);
    setClusteringResult(null); setClusteringLoading(false); setClusteringError(null);
  };

  const hasResult = result !== null;
  const qualityPicks = result?.quality_picks ?? [];

  const handleTickerClick = (ticker: string) => {
    setModalStock({
      symbol:   ticker,
      name:     companyNames[ticker] ?? ticker,
      price:    "—", change: "—", volume: "—", positive: true,
    });
  };

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
              <div className="ml-auto">
                <PageHelp
                  title="Diversify Page Guide"
                  subtitle="Understand how portfolio analysis works and what each result section means."
                  sections={[
                    {
                      title: "Enter Your Holdings",
                      body: "Type any stock tickers (e.g. AAPL, MSFT) into the input box and press Enter, comma, or Space after each. You can also use preset buttons, load your watchlist, or import a custom portfolio you saved on your Profile page.",
                    },
                    {
                      title: "Overlaps — Hidden Lead-Lag Relationships",
                      body: "Overlaps are pairs of your holdings where one stock historically leads the other. This means they share correlated risk: if you own both, one stock's move may predict the other's within days. Click a row to see detailed statistics like dCor, Sharpe, and half-life.",
                      color: "hsl(38, 92%, 50%)",
                    },
                    {
                      title: "Quality Picks & Spider Chart",
                      body: "The radar chart ranks recommended stocks across 5 factors: Momentum (35%), Fundamental Quality (25%), Sector Diversity (20%), Volatility Fit (10%), and Market Centrality (10%). Click any ticker in the chart or legend to focus on it and see its full score breakdown.",
                      color: "hsl(142, 71%, 45%)",
                    },
                    {
                      title: "Sector Donut Chart",
                      body: "Shows your current portfolio's sector distribution. Hover over a quality pick to preview how adding that stock would shift your sector exposure — the new slice appears highlighted in the donut.",
                      color: "hsl(270, 70%, 65%)",
                    },
                    {
                      title: "Monte Carlo Risk Assessment",
                      body: "Simulates your portfolio over a 63-day horizon using 1,000 paths. VaR 95% = worst loss 95% of the time. CVaR 95% = average loss in the worst 5% of scenarios. Diversification Benefit = how much your portfolio's VaR improves vs. holding each stock in isolation.",
                      color: "hsl(0, 84%, 60%)",
                    },
                    {
                      title: "K-Medoids Sector Picks",
                      body: "Uses K-Medoids clustering to select stocks from sectors that are least correlated with your holdings. The dcor badge shows decorrelation from your portfolio — lower is better. The ★ marker indicates the cluster medoid (the most representative stock in its cluster).",
                      color: "hsl(195, 80%, 50%)",
                    },
                  ]}
                />
              </div>
            </div>
            <p className="text-sm leading-relaxed" style={{ color: TEXT_SEC }}>
              Enter your holdings to uncover hidden lead-lag relationships and receive
              signal-backed diversification recommendations.
            </p>
          </div>

          {/* Input card */}
          <div className="rounded-xl p-5 mb-6" style={CARD}>
            <p className="text-xs font-medium mb-3" style={{ color: TEXT_SEC }}>
              YOUR HOLDINGS — type a ticker and press Enter, Space, or comma
            </p>

            {/* Ticker chip input with autocomplete */}
            <TickerInput
              tickers={tickers}
              onAdd={addTicker}
              onRemove={removeTicker}
              allStocks={allStocks}
            />

            {/* Quick-add row */}
            <div className="flex items-center gap-2 mt-3 flex-wrap">
              <span className="text-xs" style={{ color: TEXT_MUT }}>Presets:</span>
              {PRESETS.map(p => (
                <button key={p.label}
                  onClick={() => { setTickers(p.tickers); setResult(null); }}
                  className="text-xs px-2.5 py-1 rounded-md transition-colors"
                  style={{ background: "hsl(215,25%,14%)", border: `1px solid ${BORDER}`, color: TEXT_SEC }}
                  onMouseEnter={e => (e.currentTarget.style.background = "hsl(215,25%,18%)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "hsl(215,25%,14%)")}>
                  {p.label}
                </button>
              ))}

              {/* Divider */}
              <span className="text-xs" style={{ color: "hsl(215,20%,28%)" }}>|</span>

              {/* Add Watchlist button */}
              <button
                onClick={addWatchlist}
                disabled={savedStocks.length === 0}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md transition-colors disabled:opacity-40"
                style={{ background: "hsla(48,96%,53%,0.1)", border: "1px solid hsla(48,96%,53%,0.3)", color: "hsl(48,96%,53%)" }}
                onMouseEnter={e => { if (savedStocks.length > 0) e.currentTarget.style.background = "hsla(48,96%,53%,0.2)"; }}
                onMouseLeave={e => (e.currentTarget.style.background = "hsla(48,96%,53%,0.1)")}
                title={savedStocks.length === 0 ? "Your watchlist is empty" : `Add ${savedStocks.length} watchlist stocks`}
              >
                <Star className="w-3 h-3" />
                Add Watchlist
                {savedStocks.length > 0 && (
                  <span className="text-xs px-1 rounded" style={{ background: "hsla(48,96%,53%,0.2)" }}>
                    {savedStocks.length}
                  </span>
                )}
              </button>

              {/* Add Portfolio dropdown — only shown if user has portfolios */}
              <PortfolioDropdown
                portfolios={customPortfolios}
                onSelect={addPortfolio}
              />
            </div>

            {/* Action row */}
            <div className="flex items-center gap-3 mt-4">
              <button onClick={handleAnalyze}
                disabled={loading || tickers.length === 0}
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

              <div className="grid grid-cols-3 gap-3 mb-6">
                {[
                  { label: "Analyzed",      value: result.tickers_analyzed.length, color: BLUE,  icon: <BarChart3 className="w-4 h-4" /> },
                  { label: "Overlaps",      value: result.overlaps.length,         color: result.overlaps.length > 0 ? AMBER : GREEN, icon: <ShieldAlert className="w-4 h-4" /> },
                  { label: "Quality Picks", value: qualityPicks.length,            color: BLUE,  icon: <Sparkles className="w-4 h-4" /> },
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

              {(riskLoading || riskResult !== null || riskError !== null) && (
                <section className="mb-8">
                  <div className="flex items-center gap-2 mb-1">
                    <Activity className="w-4 h-4" style={{ color: GREEN }} />
                    <h2 className="text-base font-semibold" style={{ color: TEXT_PRI }}>Risk Assessment</h2>
                    {portfolioRiskLoading && (
                      <span className="text-xs px-2 py-0.5 rounded-full ml-1 flex items-center gap-1"
                        style={{ background: BLUE_DIM, color: BLUE }}>
                        <Loader2 className="w-3 h-3 animate-spin" />running
                      </span>
                    )}
                  </div>
                  <p className="text-xs mb-4" style={{ color: TEXT_MUT }}>
                    Monte Carlo risk profile of your inputted portfolio · {portfolioRiskResult?.risk.horizon_days ?? 63}-day horizon · {(portfolioRiskResult?.risk.n_simulations ?? 1000).toLocaleString()} simulated paths · VaR, drawdown, and diversification metrics
                  </p>
                  {portfolioRiskLoading && !portfolioRiskResult && (
                    <div className="rounded-xl px-6 py-10 text-center" style={CARD}>
                      <Loader2 className="w-8 h-8 mx-auto mb-3 animate-spin" style={{ color: BLUE }} />
                      <p className="text-sm font-medium mb-1" style={{ color: TEXT_PRI }}>Simulating portfolio risk</p>
                      <p className="text-xs leading-relaxed max-w-xs mx-auto" style={{ color: TEXT_SEC }}>
                        Running Monte Carlo simulation on your holdings — typically a few seconds
                      </p>
                    </div>
                  )}
                  {portfolioRiskError && (
                    <div className="rounded-xl px-5 py-4 flex items-center gap-3"
                      style={{ background: "hsla(0,84%,60%,0.1)", border: "1px solid hsla(0,84%,60%,0.3)" }}>
                      <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: RED }} />
                      <p className="text-sm" style={{ color: RED }}>{portfolioRiskError}</p>
                    </div>
                  )}
                  {portfolioRiskResult && <RiskResultPanel result={portfolioRiskResult} onTickerClick={handleTickerClick} />}
                </section>
              )}

              <FactorExplanations />

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
                      isSaved={isSaved}
                      onToggleSave={(ticker, name) => toggleSave({ symbol: ticker, name })}
                      portfolios={customPortfolios}
                      onAddToPortfolio={(ticker, portfolioId) => {
                        const p = customPortfolios.find(p => p.id === portfolioId);
                        if (p && !p.tickers.includes(ticker))
                          updatePortfolio(portfolioId, p.name, [...p.tickers, ticker]);
                      }}
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

              {(clusteringLoading || clusteringResult !== null || clusteringError !== null) && (
                <section className="mb-8">
                  <div className="flex items-center gap-2 mb-1">
                    <Sparkles className="w-4 h-4" style={{ color: GREEN }} />
                    <h2 className="text-base font-semibold" style={{ color: TEXT_PRI }}>K-Medoids Sector Picks</h2>
                    {clusteringLoading && (
                      <span className="text-xs px-2 py-0.5 rounded-full ml-1 flex items-center gap-1"
                        style={{ background: GREEN_DIM, color: GREEN }}>
                        <Loader2 className="w-3 h-3 animate-spin" />running
                      </span>
                    )}
                    {clusteringResult && (
                      <span className="text-xs px-2 py-0.5 rounded-full ml-1" style={{ background: GREEN_DIM, color: GREEN }}>
                        {clusteringResult.recommendations.length} picks
                      </span>
                    )}
                  </div>
                  <p className="text-xs mb-4" style={{ color: TEXT_MUT }}>
                    Stocks selected via K-Medoids clustering — decorrelated from your portfolio by sector
                  </p>
                  {clusteringLoading && !clusteringResult && (
                    <div className="rounded-xl px-6 py-10 text-center" style={CARD}>
                      <Loader2 className="w-8 h-8 mx-auto mb-3 animate-spin" style={{ color: GREEN }} />
                      <p className="text-sm font-medium mb-1" style={{ color: TEXT_PRI }}>Running K-Medoids clustering</p>
                      <p className="text-xs leading-relaxed max-w-xs mx-auto" style={{ color: TEXT_SEC }}>
                        Querying BigQuery and running K-Medoids sweep — typically 30–60 seconds
                      </p>
                    </div>
                  )}
                  {clusteringError && (
                    <div className="rounded-xl px-5 py-4 flex items-center gap-3"
                      style={{ background: "hsla(0,84%,60%,0.1)", border: "1px solid hsla(0,84%,60%,0.3)" }}>
                      <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: RED }} />
                      <p className="text-sm" style={{ color: RED }}>{clusteringError}</p>
                    </div>
                  )}
                  {clusteringResult && clusteringResult.recommendations.length > 0 && (
                    <>
                      <div className="grid grid-cols-3 gap-3">
                        {clusteringResult.recommendations.map((rec: ClusteringPick) => (
                          <div key={rec.stock} className="rounded-xl p-4 flex flex-col gap-2" style={CARD}>
                            <div className="flex items-center justify-between gap-2">
                              <button className="text-sm font-bold hover:underline min-w-0 truncate" style={{ color: TEXT_PRI }}
                                onClick={() => handleTickerClick(rec.stock)}>
                                {rec.stock}
                                {rec.is_medoid && <span className="ml-1 text-xs" style={{ color: AMBER }}>★</span>}
                              </button>
                              <div className="flex items-center gap-2 flex-shrink-0">
                                <button
                                  onClick={(e) => { e.stopPropagation(); toggleSave({ symbol: rec.stock, name: companyNames[rec.stock] ?? rec.stock }); }}
                                  className="transition-all hover:scale-110"
                                  title={isSaved(rec.stock) ? "Remove from watchlist" : "Save to watchlist"}
                                  aria-label={isSaved(rec.stock) ? "Remove from watchlist" : "Save to watchlist"}
                                >
                                  <Star className="w-3.5 h-3.5" style={isSaved(rec.stock)
                                    ? { fill: "hsl(48,96%,53%)", color: "hsl(48,96%,53%)" }
                                    : { fill: "transparent",     color: TEXT_MUT }} />
                                </button>
                                <AddToPortfolioButton
                                  ticker={rec.stock}
                                  portfolios={customPortfolios}
                                  onAdd={(portfolioId) => {
                                    const p = customPortfolios.find(p => p.id === portfolioId);
                                    if (p && !p.tickers.includes(rec.stock))
                                      updatePortfolio(portfolioId, p.name, [...p.tickers, rec.stock]);
                                  }}
                                />
                                <span className="text-xs px-1.5 py-0.5 rounded font-mono"
                                  style={{ background: GREEN_DIM, color: GREEN }}>
                                  {rec.avg_dcor_to_portfolio.toFixed(3)}
                                </span>
                              </div>
                            </div>
                            <SectorTag sector={rec.sector} />
                          </div>
                        ))}
                      </div>
                      <div className="rounded-xl overflow-hidden mt-4" style={CARD}>
                        <div className="px-5 py-3" style={{ borderBottom: `1px solid ${BORDER_D}` }}>
                          <p className="text-xs font-semibold tracking-wide" style={{ color: TEXT_SEC }}>PER-STOCK RISK</p>
                        </div>
                        <div className="overflow-x-auto">
                          <table className="w-full text-xs">
                            <thead>
                              <tr style={{ borderBottom: `1px solid ${BORDER_D}` }}>
                                {(["Ticker","Sector","VaR 95%","CVaR 95%","Prob Loss","Max Drawdown"] as const).map(h => {
                                  const detailKey =
                                    h === "VaR 95%"      ? "VaR 95% (stock)"  :
                                    h === "CVaR 95%"     ? "CVaR 95% (stock)" :
                                    h === "Prob Loss"    ? "Prob Loss"         :
                                    h === "Max Drawdown" ? "Max Drawdown"      : null;
                                  return (
                                    <th key={h} className="px-4 py-2 text-left font-medium" style={{ color: TEXT_MUT }}>
                                      <span className="inline-flex items-center">
                                        {h}
                                        {detailKey && <MetricHint detail={METRIC_DETAILS[detailKey]} position="below" />}
                                      </span>
                                    </th>
                                  );
                                })}
                              </tr>
                            </thead>
                            <tbody>
                              {clusteringResult.risk.tickers.map((ticker, i) => {
                                const s = clusteringResult.risk.per_stock[ticker];
                                if (!s) return null;
                                const rec = clusteringResult.recommendations.find((r: ClusteringPick) => r.stock === ticker);
                                const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
                                return (
                                  <tr key={ticker}
                                    style={{ borderBottom: i < clusteringResult.risk.tickers.length - 1 ? `1px solid ${BORDER_D}` : "none" }}
                                    onMouseEnter={e => (e.currentTarget.style.background = CARD_H)}
                                    onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                                  >
                                    <td className="px-4 py-2.5">
                                      <button className="font-bold hover:underline" style={{ color: TEXT_PRI }}
                                        onClick={() => handleTickerClick(ticker)}>{ticker}</button>
                                    </td>
                                    <td className="px-4 py-2.5">{rec && <SectorTag sector={rec.sector} />}</td>
                                    <td className="px-4 py-2.5 font-mono" style={{ color: RED }}>{pct(s.var_95)}</td>
                                    <td className="px-4 py-2.5 font-mono" style={{ color: RED }}>{pct(s.cvar_95)}</td>
                                    <td className="px-4 py-2.5 font-mono" style={{ color: s.prob_loss > 0.5 ? RED : AMBER }}>{pct(s.prob_loss)}</td>
                                    <td className="px-4 py-2.5 font-mono" style={{ color: RED }}>{pct(s.expected_max_drawdown)}</td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </div>
                      {clusteringResult.risk.missing.length > 0 ? (
                        <p className="text-xs mt-3 px-1" style={{ color: TEXT_MUT }}>
                          ★ = cluster medoid · dcor badge = decorrelation from your portfolio (lower is better)
                          · {clusteringResult.risk.missing.length} ticker{clusteringResult.risk.missing.length > 1 ? "s" : ""} unavailable: {clusteringResult.risk.missing.join(", ")}
                        </p>
                      ) : (
                        <p className="text-xs mt-3 px-1" style={{ color: TEXT_MUT }}>
                          ★ = cluster medoid · dcor badge = decorrelation from your portfolio (lower is better)
                        </p>
                      )}
                    </>
                  )}
                </section>
              )}
            </div>
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

      {modalStock && (
        <StockModal stock={modalStock} onClose={() => setModalStock(null)} />
      )}
    </div>
  );
}