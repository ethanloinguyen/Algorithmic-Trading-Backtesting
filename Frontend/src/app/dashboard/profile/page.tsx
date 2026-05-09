// Frontend/src/app/dashboard/profile/page.tsx
"use client";
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  TrendingUp, Star, User, Mail, LogOut, Loader2, BarChart2,
  LayoutDashboard, LineChart, Grid3X3, ChevronDown, Layers,
  FolderPlus, Trash2, X, Plus, Briefcase,
} from "lucide-react";
import { useAuth } from "@/src/app/context/AuthContext";
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

export default function ProfilePage() {
  const { user, savedStocks, toggleSave, trackClick, loading: authLoading, logout, customPortfolios, addPortfolio, removePortfolio } = useAuth();
  const router = useRouter();

  const [liveData,       setLiveData]       = useState<StockSummary[]>([]);
  const [pricesLoading,  setPricesLoading]  = useState(false);
  const [pricesError,    setPricesError]    = useState(false);
  const [selectedSector, setSelectedSector] = useState(ALL_SECTORS);

  // Custom portfolio form state
  const [portfolioName,    setPortfolioName]    = useState("");
  const [portfolioTickers, setPortfolioTickers] = useState<string[]>([]);
  const [portfolioInput,   setPortfolioInput]   = useState("");
  const [savingPortfolio,  setSavingPortfolio]  = useState(false);
  const portfolioInputRef = useRef<HTMLInputElement>(null);

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
          <div className="grid grid-cols-3 gap-4 mb-6">
            {[
              { label: "Saved Stocks", value: savedStocks.length.toString(), color: "hsl(217,91%,60%)" },
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

          {/* ── Custom Portfolios ── */}
          <div className="mt-6">
            <div className="flex items-center gap-2 mb-4">
              <Briefcase className="w-4 h-4" style={{ color: "hsl(217,91%,60%)" }} />
              <h2 className="text-base font-semibold" style={{ color: "hsl(210,40%,92%)" }}>
                Custom Portfolios
              </h2>
              <span
                className="text-xs px-2 py-0.5 rounded-full"
                style={{ background: "hsla(217,91%,60%,0.15)", color: "hsl(217,91%,70%)" }}
              >
                {customPortfolios.length} saved
              </span>
            </div>
            <p className="text-xs mb-4" style={{ color: "hsl(215,15%,45%)" }}>
              Create named portfolios and they&apos;ll appear as presets on the Diversify page.
            </p>

            {/* Existing portfolios */}
            {customPortfolios.length > 0 && (
              <div className="space-y-2 mb-5">
                {customPortfolios.map((portfolio) => (
                  <div
                    key={portfolio.id}
                    className="rounded-xl px-5 py-4 flex items-center justify-between gap-4"
                    style={{ background: "hsl(215,25%,11%)", border: "1px solid hsl(215,20%,18%)" }}
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold mb-1" style={{ color: "hsl(210,40%,92%)" }}>
                        {portfolio.name}
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {portfolio.tickers.map((t) => (
                          <span
                            key={t}
                            className="text-xs px-2 py-0.5 rounded-md font-semibold"
                            style={{ background: "hsla(217,91%,60%,0.15)", color: "hsl(217,91%,70%)" }}
                          >
                            {t}
                          </span>
                        ))}
                      </div>
                    </div>
                    <button
                      onClick={() => removePortfolio(portfolio.id)}
                      className="p-2 rounded-lg transition-all hover:opacity-70 flex-shrink-0"
                      style={{ color: "hsl(0,84%,60%)", background: "hsla(0,84%,60%,0.1)" }}
                      aria-label="Delete portfolio"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Create new portfolio form */}
            <div className="rounded-xl p-5" style={{ background: "hsl(215,25%,11%)", border: "1px solid hsl(215,20%,18%)" }}>
              <p className="text-xs font-medium mb-3" style={{ color: "hsl(215,15%,55%)" }}>
                <FolderPlus className="w-3.5 h-3.5 inline mr-1.5 relative" style={{ top: -1 }} />
                CREATE NEW PORTFOLIO
              </p>

              {/* Name input */}
              <input
                type="text"
                placeholder="Portfolio name (e.g. Growth Picks)"
                value={portfolioName}
                onChange={(e) => setPortfolioName(e.target.value)}
                className="w-full px-4 py-2 rounded-lg text-sm mb-3 outline-none transition-all"
                style={{
                  background: "hsl(215,25%,8%)",
                  border: "1px solid hsl(215,20%,20%)",
                  color: "hsl(210,40%,85%)",
                }}
                onFocus={(e) => (e.currentTarget.style.borderColor = "hsl(217,91%,60%)")}
                onBlur={(e) => (e.currentTarget.style.borderColor = "hsl(215,20%,20%)")}
              />

              {/* Ticker tag input */}
              <div
                className="flex flex-wrap gap-2 min-h-10 p-2 rounded-lg cursor-text mb-3"
                style={{ background: "hsl(215,25%,8%)", border: "1px solid hsl(215,20%,20%)" }}
                onClick={() => portfolioInputRef.current?.focus()}
              >
                {portfolioTickers.map((t) => (
                  <span
                    key={t}
                    className="flex items-center gap-1 px-2.5 py-0.5 rounded-md text-xs font-semibold"
                    style={{ background: "hsla(217,91%,60%,0.15)", color: "hsl(217,91%,70%)" }}
                  >
                    {t}
                    <button
                      onClick={(e) => { e.stopPropagation(); setPortfolioTickers(prev => prev.filter(x => x !== t)); }}
                      className="hover:opacity-60 ml-0.5"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </span>
                ))}
                <input
                  ref={portfolioInputRef}
                  value={portfolioInput}
                  onChange={(e) => setPortfolioInput(e.target.value.toUpperCase())}
                  onKeyDown={(e) => {
                    if (["Enter", ",", " ", "Tab"].includes(e.key)) {
                      e.preventDefault();
                      const val = portfolioInput.trim();
                      if (val && !portfolioTickers.includes(val) && val.length <= 10) {
                        setPortfolioTickers(prev => [...prev, val]);
                      }
                      setPortfolioInput("");
                    }
                    if (e.key === "Backspace" && !portfolioInput && portfolioTickers.length) {
                      setPortfolioTickers(prev => prev.slice(0, -1));
                    }
                  }}
                  onBlur={() => {
                    const val = portfolioInput.trim();
                    if (val && !portfolioTickers.includes(val) && val.length <= 10) {
                      setPortfolioTickers(prev => [...prev, val]);
                      setPortfolioInput("");
                    }
                  }}
                  placeholder={portfolioTickers.length === 0 ? "Type tickers, press Enter or comma" : ""}
                  className="flex-1 min-w-24 bg-transparent outline-none text-sm"
                  style={{ color: "hsl(210,40%,85%)" }}
                />
              </div>

              {/* Add from saved stocks shortcut */}
              {savedStocks.length > 0 && (
                <div className="flex items-center gap-2 mb-3 flex-wrap">
                  <span className="text-xs" style={{ color: "hsl(215,15%,45%)" }}>Add from saved:</span>
                  {savedStocks.slice(0, 8).map((s) => (
                    <button
                      key={s.symbol}
                      onClick={() => {
                        if (!portfolioTickers.includes(s.symbol)) {
                          setPortfolioTickers(prev => [...prev, s.symbol]);
                        }
                      }}
                      disabled={portfolioTickers.includes(s.symbol)}
                      className="text-xs px-2 py-0.5 rounded-md transition-colors"
                      style={{
                        background: portfolioTickers.includes(s.symbol) ? "hsla(217,91%,60%,0.08)" : "hsl(215,25%,14%)",
                        border: "1px solid hsl(215,20%,20%)",
                        color: portfolioTickers.includes(s.symbol) ? "hsl(215,15%,38%)" : "hsl(210,40%,75%)",
                        cursor: portfolioTickers.includes(s.symbol) ? "default" : "pointer",
                      }}
                    >
                      <Plus className="w-2.5 h-2.5 inline mr-1" style={{ position: "relative", top: -0.5 }} />
                      {s.symbol}
                    </button>
                  ))}
                </div>
              )}

              <button
                onClick={async () => {
                  const finalTickers = portfolioInput.trim()
                    ? [...portfolioTickers, portfolioInput.trim().toUpperCase()].filter((v, i, a) => a.indexOf(v) === i)
                    : portfolioTickers;
                  if (!portfolioName.trim() || finalTickers.length === 0) return;
                  setSavingPortfolio(true);
                  await addPortfolio(portfolioName, finalTickers);
                  setPortfolioName("");
                  setPortfolioTickers([]);
                  setPortfolioInput("");
                  setSavingPortfolio(false);
                }}
                disabled={savingPortfolio || !portfolioName.trim() || (portfolioTickers.length === 0 && !portfolioInput.trim())}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold disabled:opacity-40 transition-opacity"
                style={{ background: "hsl(217,91%,60%)", color: "white" }}
              >
                {savingPortfolio
                  ? <><Loader2 className="w-4 h-4 animate-spin" /> Saving…</>
                  : <><FolderPlus className="w-4 h-4" /> Save Portfolio</>}
              </button>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}