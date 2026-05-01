// Frontend/src/app/dashboard/profile/page.tsx
"use client";
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  TrendingUp, Star, User, Mail, LogOut, Loader2, BarChart2,
  LayoutDashboard, LineChart, Grid3X3, ChevronDown, Layers,
  Briefcase, Plus, X, Trash2, ArrowRight,
} from "lucide-react";
import { useAuth, type SavedPortfolio } from "@/src/app/context/AuthContext";
import { fetchStockSummaries, type StockSummary } from "@/src/app/lib/api";
import { SECTORS, ALL_SECTORS, filterBySector } from "@/src/app/lib/sectorData";

// ─── Shared nav (no sidebar — using top nav like dashboard) ──────────────────
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
      <div className="flex items-center gap-2">
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
            style={{ color: "hsl(215, 15%, 55%)" }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "hsl(210, 40%, 85%)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "hsl(215, 15%, 55%)")}
          >
            <Icon className="w-3.5 h-3.5" />
            {label}
          </button>
        ))}
      </nav>
      <div className="flex items-center gap-3">
        <button
          onClick={() => router.push("/dashboard/profile")}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-all"
          style={{ background: "hsl(217, 91%, 60% / 0.15)", border: "1px solid hsl(217, 91%, 60% / 0.3)", color: "hsl(217, 91%, 70%)" }}
        >
          <User className="w-3.5 h-3.5" />
          {displayName}
        </button>
        <button onClick={onLogout} className="p-2 rounded-lg transition-all hover:opacity-70" style={{ color: "hsl(215, 15%, 50%)" }} aria-label="Sign out">
          <LogOut className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
}

const card = { background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" };

// ─── Sector dropdown ──────────────────────────────────────────────────────────
function SectorDropdown({
  value,
  onChange,
}: {
  value: string;
  onChange: (sector: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-all"
        style={{
          background: value !== ALL_SECTORS ? "hsla(217,91%,60%,0.15)" : "hsl(215,25%,8%)",
          border:     value !== ALL_SECTORS ? "1px solid hsla(217,91%,60%,0.4)" : "1px solid hsl(215,20%,20%)",
          color:      value !== ALL_SECTORS ? "hsl(217,91%,70%)" : "hsl(210,40%,75%)",
          minWidth:   148,
        }}
      >
        <Layers className="w-3.5 h-3.5 flex-shrink-0" />
        <span className="flex-1 text-left truncate">{value}</span>
        <ChevronDown
          className="w-3.5 h-3.5 flex-shrink-0 transition-transform"
          style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
        />
      </button>

      {open && (
        <div
          className="absolute right-0 top-full mt-1 rounded-xl overflow-y-auto z-50"
          style={{
            background: "hsl(215,25%,11%)",
            border:     "1px solid hsl(215,20%,20%)",
            boxShadow:  "0 8px 24px rgba(0,0,0,0.4)",
            minWidth:   180,
            maxHeight:  320,
          }}
        >
          {SECTORS.map((sector) => (
            <button
              key={sector}
              onClick={() => { onChange(sector); setOpen(false); }}
              className="w-full text-left px-4 py-2.5 text-sm transition-colors"
              style={{
                background:   value === sector ? "hsla(217,91%,60%,0.15)" : "transparent",
                color:        value === sector ? "hsl(217,91%,70%)" : "hsl(210,40%,80%)",
                borderBottom: "1px solid hsl(215,20%,16%)",
              }}
              onMouseEnter={(e) =>
                value !== sector && (e.currentTarget.style.background = "hsl(215,25%,15%)")
              }
              onMouseLeave={(e) =>
                value !== sector && (e.currentTarget.style.background = "transparent")
              }
            >
              {sector === ALL_SECTORS ? (
                <span className="flex items-center gap-2">
                  <Layers className="w-3.5 h-3.5" />
                  {sector}
                </span>
              ) : sector}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Portfolio manager ────────────────────────────────────────────────────────
function PortfolioManager({
  portfolios,
  onSave,
  onDelete,
}: {
  portfolios: SavedPortfolio[];
  onSave:     (name: string, tickers: string[]) => Promise<void>;
  onDelete:   (name: string) => Promise<void>;
}) {
  const router = useRouter();
  const [name,     setName]     = useState("");
  const [tickerIn, setTickerIn] = useState("");
  const [tickers,  setTickers]  = useState<string[]>([]);
  const [saving,   setSaving]   = useState(false);
  const [nameErr,  setNameErr]  = useState("");

  const addTicker = (raw: string) => {
    const parts = raw.toUpperCase().split(/[\s,]+/).filter(Boolean);
    setTickers(prev => {
      const next = [...prev];
      for (const t of parts) if (t.length <= 10 && !next.includes(t)) next.push(t);
      return next;
    });
    setTickerIn("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (["Enter", ",", " ", "Tab"].includes(e.key)) {
      e.preventDefault();
      if (tickerIn.trim()) addTicker(tickerIn.trim());
    }
    if (e.key === "Backspace" && !tickerIn && tickers.length)
      setTickers(prev => prev.slice(0, -1));
  };

  const handleSave = async () => {
    const trimmedName = name.trim();
    if (!trimmedName) { setNameErr("Portfolio name is required."); return; }
    if (!tickers.length) { setNameErr("Add at least one ticker."); return; }
    setNameErr("");
    setSaving(true);
    try {
      await onSave(trimmedName, tickers);
      setName(""); setTickers([]); setTickerIn("");
    } finally {
      setSaving(false);
    }
  };

  const BG   = "hsl(213,27%,7%)";
  const BLUE = "hsl(217,91%,60%)";
  const GREEN= "hsl(142,71%,45%)";
  const MUTED= "hsl(215,15%,55%)";
  const DIM  = "hsl(215,15%,40%)";
  const BORDER = "hsl(215,20%,18%)";

  return (
    <div className="rounded-xl mt-6" style={{ ...card, overflow: "visible" }}>

      {/* Header */}
      <div className="px-6 py-4 flex items-center justify-between"
        style={{ borderBottom: `1px solid ${BORDER}` }}>
        <div className="flex items-center gap-2">
          <Briefcase className="w-4 h-4" style={{ color: BLUE }} />
          <h2 className="text-base font-semibold" style={{ color: "hsl(210,40%,92%)" }}>
            My Portfolios
          </h2>
          {portfolios.length > 0 && (
            <span className="text-xs px-2 py-0.5 rounded-full"
              style={{ background: "hsla(217,91%,60%,0.15)", color: BLUE }}>
              {portfolios.length}
            </span>
          )}
        </div>
        <p className="text-xs" style={{ color: DIM }}>
          Portfolios appear as presets on the Diversify page
        </p>
      </div>

      {/* Existing portfolios */}
      {portfolios.length === 0 && (
        <div className="py-10 text-center">
          <Briefcase className="w-9 h-9 mx-auto mb-3" style={{ color: "hsl(215,15%,28%)" }} />
          <p className="text-sm font-medium mb-1" style={{ color: MUTED }}>No portfolios yet</p>
          <p className="text-xs" style={{ color: DIM }}>Create one below to use it as a preset in the Diversifier.</p>
        </div>
      )}

      {portfolios.map(p => (
        <div key={p.name} className="px-6 py-4 flex items-start gap-3 transition-colors"
          style={{ borderTop: `1px solid ${BORDER}` }}
          onMouseEnter={e => (e.currentTarget.style.background = "hsl(215,25%,13%)")}
          onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm font-semibold" style={{ color: "hsl(210,40%,92%)" }}>
                {p.name}
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full"
                style={{ background: "hsla(142,71%,45%,0.12)", color: GREEN }}>
                {p.tickers.length} stock{p.tickers.length !== 1 ? "s" : ""}
              </span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {p.tickers.map(t => (
                <span key={t} className="text-xs px-2 py-0.5 rounded-md font-medium"
                  style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER}`, color: MUTED }}>
                  {t}
                </span>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2 flex-shrink-0 mt-0.5">
            <button
              onClick={() => router.push("/dashboard/diversify")}
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-colors"
              style={{ background: "hsla(217,91%,60%,0.12)", color: BLUE, border: "1px solid hsla(217,91%,60%,0.3)" }}
              onMouseEnter={e => (e.currentTarget.style.background = "hsla(217,91%,60%,0.2)")}
              onMouseLeave={e => (e.currentTarget.style.background = "hsla(217,91%,60%,0.12)")}>
              Diversify <ArrowRight className="w-3 h-3" />
            </button>
            <button
              onClick={() => onDelete(p.name)}
              className="p-1.5 rounded-lg transition-colors hover:opacity-70"
              style={{ color: "hsl(0,84%,60%)" }}
              aria-label={`Delete ${p.name}`}>
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      ))}

      {/* Create portfolio form */}
      <div className="px-6 py-5" style={{ borderTop: `1px solid ${BORDER}`, background: "hsl(215,25%,9%)" }}>
        <p className="text-xs font-semibold mb-3" style={{ color: MUTED }}>
          CREATE NEW PORTFOLIO
        </p>

        {/* Name field */}
        <input
          type="text"
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder="Portfolio name (e.g. My Growth Picks)"
          className="w-full px-3 py-2 rounded-lg text-sm outline-none mb-3 transition-all"
          style={{ background: "hsl(215,25%,7%)", border: "1px solid hsl(215,20%,22%)", color: "hsl(210,40%,92%)" }}
          onFocus={e => (e.currentTarget.style.borderColor = BLUE)}
          onBlur={e  => (e.currentTarget.style.borderColor = "hsl(215,20%,22%)")}
        />

        {/* Ticker chip input */}
        <div
          className="flex flex-wrap gap-2 min-h-10 p-2 rounded-lg cursor-text mb-3"
          style={{ background: "hsl(215,25%,7%)", border: "1px solid hsl(215,20%,22%)" }}
          onClick={e => (e.currentTarget.querySelector("input") as HTMLInputElement)?.focus()}>
          {tickers.map(t => (
            <span key={t} className="flex items-center gap-1 px-2.5 py-0.5 rounded-md text-xs font-semibold"
              style={{ background: "hsla(217,91%,60%,0.15)", color: BLUE }}>
              {t}
              <button onClick={() => setTickers(prev => prev.filter(x => x !== t))} className="hover:opacity-60 ml-0.5">
                <X className="w-2.5 h-2.5" />
              </button>
            </span>
          ))}
          <input
            value={tickerIn}
            onChange={e => setTickerIn(e.target.value.toUpperCase())}
            onKeyDown={handleKeyDown}
            onBlur={() => { if (tickerIn.trim()) addTicker(tickerIn.trim()); }}
            placeholder={tickers.length === 0 ? "Add tickers — AAPL, MSFT…" : ""}
            className="flex-1 min-w-24 bg-transparent outline-none text-sm"
            style={{ color: "hsl(210,40%,92%)" }}
          />
        </div>

        {nameErr && (
          <p className="text-xs mb-3" style={{ color: "hsl(0,84%,60%)" }}>{nameErr}</p>
        )}

        <button
          onClick={handleSave}
          disabled={saving}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold disabled:opacity-50 transition-opacity hover:opacity-80"
          style={{ background: BLUE, color: "white" }}>
          {saving
            ? <><Loader2 className="w-4 h-4 animate-spin" /> Saving…</>
            : <><Plus className="w-4 h-4" /> Save Portfolio</>}
        </button>
      </div>
    </div>
  );
}

export default function ProfilePage() {
  const { user, savedStocks, savedPortfolios, toggleSave, savePortfolio, deletePortfolio, trackClick, loading: authLoading, logout } = useAuth();
  const router = useRouter();

  const [liveData,       setLiveData]       = useState<StockSummary[]>([]);
  const [pricesLoading,  setPricesLoading]  = useState(false);
  const [pricesError,    setPricesError]    = useState(false);
  const [selectedSector, setSelectedSector] = useState(ALL_SECTORS);

  // Redirect unauthenticated
  useEffect(() => {
    if (!authLoading && !user) router.replace("/");
  }, [user, authLoading, router]);

  // Fetch live prices whenever saved stocks change
  useEffect(() => {
    if (!savedStocks.length) { setLiveData([]); return; }
    setPricesLoading(true);
    setPricesError(false);
    fetchStockSummaries(savedStocks.map((s) => s.symbol))
      .then((data) => { setLiveData(data); setPricesLoading(false); })
      .catch(()    => { setPricesError(true); setPricesLoading(false); });
  }, [savedStocks]);

  const handleLogout = async () => {
    await logout();
    document.cookie = "ll_authed=; Max-Age=0; path=/";
    router.push("/");
  };

  const displayName = user?.displayName || user?.email?.split("@")[0] || "User";
  const joinDate = user?.metadata?.creationTime
    ? new Date(user.metadata.creationTime).toLocaleDateString("en-US", { month: "long", year: "numeric" })
    : "—";

  // Merge Firestore saved list with live prices
  const mergedStocks = savedStocks.map((saved) => {
    const live = liveData.find((l) => l.symbol === saved.symbol);
    return {
      symbol:   saved.symbol,
      name:     saved.name,
      price:    live?.price    ?? "—",
      change:   live?.change   ?? "—",
      volume:   live?.volume   ?? "—",
      positive: live?.positive ?? true,
    };
  });

  // Apply sector filter to the merged list
  const filteredStocks = filterBySector(mergedStocks, selectedSector);

  // Avg. change — based only on sector-filtered stocks that have live data
  const avgChange = (() => {
    const sectorLive = filterBySector(liveData, selectedSector);
    const vals = sectorLive
      .map((s) => parseFloat(s.change.replace("%", "").replace("+", "")))
      .filter((n) => !isNaN(n));
    if (!vals.length) return "—";
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    return `${avg >= 0 ? "+" : ""}${avg.toFixed(2)}%`;
  })();
  const avgPositive = avgChange !== "—" && !avgChange.startsWith("-");

  return (
    <div className="min-h-screen" style={{ background: "hsl(213, 27%, 7%)" }}>
      <Navbar displayName={displayName} onLogout={handleLogout} />

      <main className="pt-14">
        <div className="max-w-4xl mx-auto px-6 py-8">

          {/* ── Profile card ── */}
          <div className="rounded-2xl p-6 mb-6 flex items-center justify-between gap-4" style={card}>
            <div className="flex items-center gap-5">
              <div
                className="w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold flex-shrink-0"
                style={{ background: "hsl(217, 91%, 60%)", color: "white" }}
              >
                {displayName[0].toUpperCase()}
              </div>
              <div>
                <h1 className="text-xl font-bold mb-1" style={{ color: "hsl(210, 40%, 92%)" }}>
                  {displayName}
                </h1>
                <div className="flex flex-wrap items-center gap-4">
                  <span className="flex items-center gap-1.5 text-sm" style={{ color: "hsl(215,15%,55%)" }}>
                    <Mail className="w-3.5 h-3.5" /> {user?.email}
                  </span>
                  <span className="flex items-center gap-1.5 text-sm" style={{ color: "hsl(215,15%,55%)" }}>
                    <User className="w-3.5 h-3.5" /> Joined {joinDate}
                  </span>
                </div>
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium flex-shrink-0 hover:opacity-80 transition-opacity"
              style={{ background: "hsl(215,25%,16%)", border: "1px solid hsl(215,20%,22%)", color: "hsl(215,15%,65%)" }}
            >
              <LogOut className="w-4 h-4" /> Sign Out
            </button>
          </div>

          {/* ── Stats ── */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            {[
              { label: "Saved Stocks", value: savedStocks.length.toString(),    color: "hsl(217,91%,60%)" },
              { label: "My Portfolios",value: savedPortfolios.length.toString(), color: "hsl(142,71%,45%)" },
              { label: "Avg. Change",  value: avgChange, color: avgPositive ? "hsl(142,71%,45%)" : "hsl(0,84%,60%)" },
              { label: "Watchlist",    value: savedStocks.length > 0 ? "Active" : "Empty", color: "hsl(280,70%,65%)" },
            ].map(({ label, value, color }) => (
              <div key={label} className="rounded-xl p-5" style={card}>
                <p className="text-xs mb-2" style={{ color: "hsl(215,15%,55%)" }}>{label}</p>
                <p className="text-2xl font-bold" style={{ color }}>{value}</p>
              </div>
            ))}
          </div>

          {/* ── Saved stocks table ── */}
          <div className="rounded-xl" style={{ ...card, overflow: "visible" }}>
            <div
              className="px-6 py-4 flex items-center justify-between gap-3"
              style={{ borderBottom: "1px solid hsl(215,20%,18%)" }}
            >
              <h2 className="text-base font-semibold" style={{ color: "hsl(210,40%,92%)" }}>
                Saved Stocks
              </h2>
              <div className="flex items-center gap-2 ml-auto flex-wrap">
                {pricesError && (
                  <span className="text-xs" style={{ color: "hsl(0,84%,60%)" }}>
                    Live prices unavailable
                  </span>
                )}
                {/* Sector dropdown */}
                <SectorDropdown value={selectedSector} onChange={setSelectedSector} />
                <span
                  className="text-xs px-2 py-1 rounded-full"
                  style={{ background: "hsla(217,91%,60%,0.15)", color: "hsl(217,91%,70%)" }}
                >
                  {filteredStocks.length} stock{filteredStocks.length !== 1 ? "s" : ""}
                </span>
              </div>
            </div>

            {authLoading && (
              <div className="flex items-center justify-center py-16">
                <Loader2 className="w-5 h-5 animate-spin" style={{ color: "hsl(217,91%,60%)" }} />
              </div>
            )}

            {!authLoading && savedStocks.length === 0 && (
              <div className="py-16 text-center">
                <Star className="w-10 h-10 mx-auto mb-3" style={{ color: "hsl(215,15%,28%)" }} />
                <p className="text-sm font-medium mb-1" style={{ color: "hsl(215,15%,55%)" }}>
                  No saved stocks yet
                </p>
                <p className="text-xs" style={{ color: "hsl(215,15%,40%)" }}>
                  Star stocks from the dashboard to track them here.
                </p>
              </div>
            )}

            {!authLoading && savedStocks.length > 0 && filteredStocks.length === 0 && (
              <div className="py-12 text-center">
                <Layers className="w-8 h-8 mx-auto mb-3" style={{ color: "hsl(215,15%,28%)" }} />
                <p className="text-sm font-medium mb-1" style={{ color: "hsl(215,15%,55%)" }}>
                  No saved stocks in this sector
                </p>
                <p className="text-xs" style={{ color: "hsl(215,15%,40%)" }}>
                  Try selecting a different sector or &quot;All Sectors&quot;.
                </p>
              </div>
            )}

            {!authLoading && filteredStocks.length > 0 && (
              <>
                <div
                  className="grid px-6 py-2.5 text-xs font-medium"
                  style={{
                    gridTemplateColumns: "1fr 1fr 1fr 1fr 36px",
                    color: "hsl(215,15%,50%)",
                    borderBottom: "1px solid hsl(215,20%,16%)",
                  }}
                >
                  <span>Ticker</span>
                  <span className="text-right">Price</span>
                  <span className="text-right">Change</span>
                  <span className="text-right">Volume</span>
                  <span />
                </div>

                {filteredStocks.map((stock) => (
                  <div
                    key={stock.symbol}
                    onClick={() => trackClick(stock.symbol)}
                    className="grid px-6 py-3.5 cursor-pointer items-center transition-colors"
                    style={{
                      gridTemplateColumns: "1fr 1fr 1fr 1fr 36px",
                      borderTop: "1px solid hsl(215,20%,16%)",
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = "hsl(215,25%,14%)")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold" style={{ color: "hsl(210,40%,92%)" }}>
                        {stock.symbol}
                      </span>
                      <span className="text-xs" style={{ color: "hsl(215,15%,50%)" }}>
                        {stock.name}
                      </span>
                    </div>

                    {pricesLoading ? (
                      <>
                        <span />
                        <span className="flex justify-end">
                          <Loader2 className="w-3.5 h-3.5 animate-spin" style={{ color: "hsl(217,91%,60%)" }} />
                        </span>
                        <span />
                      </>
                    ) : (
                      <>
                        <span className="text-sm font-medium text-right" style={{ color: "hsl(210,40%,92%)" }}>
                          {stock.price}
                        </span>
                        <span
                          className="text-sm font-semibold text-right"
                          style={{ color: stock.positive ? "hsl(142,71%,45%)" : "hsl(0,84%,60%)" }}
                        >
                          {stock.change}
                        </span>
                        <span className="text-sm text-right" style={{ color: "hsl(215,15%,55%)" }}>
                          {stock.volume}
                        </span>
                      </>
                    )}

                    <div className="flex justify-end">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleSave({ symbol: stock.symbol, name: stock.name });
                        }}
                        aria-label="Remove from saved"
                        className="transition-opacity hover:opacity-60"
                      >
                        <Star
                          className="w-4 h-4"
                          style={{ fill: "hsl(48,96%,53%)", color: "hsl(48,96%,53%)" }}
                        />
                      </button>
                    </div>
                  </div>
                ))}
              </>
            )}
          </div>
          {/* ── My Portfolios ── */}
          <PortfolioManager
            portfolios={savedPortfolios}
            onSave={savePortfolio}
            onDelete={deletePortfolio}
          />

        </div>
      </main>
    </div>
  );
}