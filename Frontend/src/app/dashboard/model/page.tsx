// frontend/src/app/dashboard/model/page.tsx
"use client";
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  TrendingUp, TrendingDown, Search, BarChart2,
  LogOut, User, LayoutDashboard, LineChart, Grid3X3,
  Loader2, ChevronRight, Layers, Star, Info,
} from "lucide-react";
import { useAuth } from "@/src/app/context/AuthContext";
import StockModal, { type Stock } from "@/components/ui/StockModal";

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

// ─── Navbar ───────────────────────────────────────────────────────────────────
function Navbar({ displayName, onLogout }: { displayName: string; onLogout: () => void }) {
  const router = useRouter();
  const navItems = [
    { label: "Dashboard", icon: LayoutDashboard, href: "/dashboard" },
    { label: "Analysis",  icon: LineChart,       href: "/dashboard/analysis" },
    { label: "Model",     icon: BarChart2,       href: "/dashboard/model" },
    { label: "Heatmap",   icon: Grid3X3,         href: "/dashboard/heatmap" },
  ];
  return (
    <header
      className="fixed top-0 left-0 right-0 z-40 flex items-center justify-between px-6 h-14"
      style={{ background: "hsl(215, 25%, 10%)", borderBottom: "1px solid hsl(215, 20%, 17%)" }}
    >
      <div className="flex items-center gap-2 flex-shrink-0">
        <div className="w-7 h-7 rounded-md flex items-center justify-center" style={{ background: "hsl(217, 91%, 60%)" }}>
          <TrendingUp className="w-3.5 h-3.5 text-white" strokeWidth={2.5} />
        </div>
        <span className="text-base font-bold" style={{ color: "hsl(217, 91%, 60%)" }}>LagLens</span>
      </div>
      <nav className="flex items-center gap-1">
        {navItems.map(({ label, icon: Icon, href }) => (
          <button
            key={label}
            onClick={() => router.push(href)}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all"
            style={{
              background: href === "/dashboard/model" ? "hsl(217, 91%, 60% / 0.15)" : "transparent",
              color:      href === "/dashboard/model" ? "hsl(217, 91%, 70%)"         : "hsl(215, 15%, 55%)",
            }}
          >
            <Icon className="w-3.5 h-3.5" />{label}
          </button>
        ))}
      </nav>
      <div className="flex items-center gap-3">
        <button
          onClick={() => router.push("/dashboard/profile")}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-all"
          style={{ background: "hsl(215, 25%, 15%)", border: "1px solid hsl(215, 20%, 22%)", color: "hsl(210, 40%, 85%)" }}
        >
          <User className="w-3.5 h-3.5" style={{ color: "hsl(215, 15%, 55%)" }} />
          {displayName}
        </button>
        <button onClick={onLogout} className="p-2 rounded-lg hover:opacity-70" style={{ color: "hsl(215, 15%, 50%)" }}>
          <LogOut className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
}

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
  const { user, loading: authLoading, isSaved, toggleSave, logout } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!authLoading && !user) router.replace("/");
  }, [user, authLoading, router]);

  const [tickers,        setTickers]        = useState<TickerOption[]>([]);
  const [query,          setQuery]          = useState("");
  const [showDropdown,   setShowDropdown]   = useState(false);
  const [selectedTicker, setSelectedTicker] = useState<TickerOption | null>(null);
  const [analyzing,      setAnalyzing]      = useState(false);
  const [result,         setResult]         = useState<AnalyzeResult | null>(null);
  const [error,          setError]          = useState<string | null>(null);
  const [selectedStock,  setSelectedStock]  = useState<Stock | null>(null);
  const [lagWindow,      setLagWindow]      = useState<5 | 10 | 30>(10);

  const inputRef   = useRef<HTMLInputElement>(null);
  const displayName = user?.displayName || user?.email?.split("@")[0] || "User";

  // Load ticker list on mount
  useEffect(() => {
    fetch(`${BASE}/api/model/tickers`)
      .then(r => r.json())
      .then(d => setTickers(d.tickers ?? []))
      .catch(() => {});
  }, []);

  // Autocomplete
  const filtered = query.length >= 1
    ? tickers.filter(t =>
        t.symbol.toLowerCase().startsWith(query.toLowerCase()) ||
        t.name.toLowerCase().includes(query.toLowerCase())
      ).slice(0, 8)
    : [];

  const selectTicker = (t: TickerOption) => {
    setSelectedTicker(t);
    setQuery(t.symbol);
    setShowDropdown(false);
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

  const handleLogout = async () => {
    await logout();
    document.cookie = "ll_authed=; Max-Age=0; path=/";
    router.push("/");
  };

  // Filter by lag window, then deduplicate by symbol keeping the
  // highest attn_weight entry for each stock (same stock can appear
  // at multiple lag_days values in the raw model output).
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
      <Navbar displayName={displayName} onLogout={handleLogout} />

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
            </div>
            <p className="text-sm ml-12" style={{ color: "hsl(215, 15%, 50%)" }}>
              Identify which stocks historically lead a target — using a 2-layer GRU
              with cross-attention trained on the Russell 1000 + top-1000 Russell 2000.
              Results are as of <span style={{ color: "hsl(215, 15%, 65%)" }}>2024-12-30</span>.
            </p>
          </div>

          {/* ── Input row ── */}
          <div className="flex items-end gap-4 mb-8">
            <div className="flex flex-col gap-1.5 flex-1 max-w-xs">
              <label className="text-xs font-medium" style={{ color: "hsl(215, 15%, 55%)" }}>Target Stock</label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 pointer-events-none" style={{ color: "hsl(215, 15%, 45%)" }} />
                <input
                  ref={inputRef}
                  type="text"
                  value={query}
                  onChange={(e) => { setQuery(e.target.value); setShowDropdown(true); setSelectedTicker(null); }}
                  onFocus={() => setShowDropdown(true)}
                  onBlur={() => setTimeout(() => setShowDropdown(false), 150)}
                  placeholder="e.g. AAPL, JPM, XOM…"
                  autoComplete="off"
                  className="w-full pl-9 pr-4 py-2.5 rounded-lg text-sm outline-none transition-all"
                  style={{ background: "hsl(215, 25%, 13%)", border: "1px solid hsl(215, 20%, 22%)", color: "hsl(210, 40%, 92%)" }}
                  onFocusCapture={(e) => (e.currentTarget.style.borderColor = "hsl(217, 91%, 60%)")}
                  onBlurCapture={(e)  => (e.currentTarget.style.borderColor = "hsl(215, 20%, 22%)")}
                />
                {showDropdown && filtered.length > 0 && (
                  <div
                    className="absolute top-full left-0 right-0 mt-1 rounded-xl overflow-hidden z-30"
                    style={{ background: "hsl(215, 25%, 13%)", border: "1px solid hsl(215, 20%, 22%)", boxShadow: "0 12px 32px hsl(213, 27%, 3% / 0.8)" }}
                  >
                    {filtered.map((t) => (
                      <button
                        key={t.symbol}
                        onMouseDown={() => selectTicker(t)}
                        className="w-full flex items-center justify-between px-4 py-2.5 text-sm transition-colors"
                        style={{ borderTop: "1px solid hsl(215, 20%, 17%)" }}
                        onMouseEnter={(e) => (e.currentTarget.style.background = "hsl(215, 25%, 17%)")}
                        onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                      >
                        <div className="flex items-center gap-3">
                          <span className="font-bold w-14 text-left" style={{ color: "hsl(210, 40%, 92%)" }}>{t.symbol}</span>
                          <span className="text-xs" style={{ color: "hsl(215, 15%, 50%)" }}>{t.name}</span>
                        </div>
                        <span
                          className="text-xs px-1.5 py-0.5 rounded ml-2 flex-shrink-0"
                          style={{ background: `${SECTOR_COLORS[t.sector] ?? "hsl(215,15%,40%)"}22`, color: SECTOR_COLORS[t.sector] ?? "hsl(215,15%,55%)" }}
                        >
                          {t.sector}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Lag window */}
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium" style={{ color: "hsl(215, 15%, 55%)" }}>Lag Window</label>
              <div className="flex items-center rounded-lg overflow-hidden" style={{ border: "1px solid hsl(215, 20%, 22%)", background: "hsl(215, 25%, 13%)" }}>
                {([5, 10, 30] as const).map((v) => (
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
                    {([5, 10, 30] as const).map((v) => (
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

        </div>
      </main>

      {selectedStock && (
        <StockModal stock={selectedStock} onClose={() => setSelectedStock(null)} />
      )}
    </div>
  );
}