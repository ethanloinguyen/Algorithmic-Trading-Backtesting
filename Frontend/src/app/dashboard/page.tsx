// frontend/src/app/dashboard/page.tsx
"use client";
import { useState, useEffect, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  TrendingUp, TrendingDown, Search, BarChart2,
  Star, LogOut, User, LayoutDashboard, LineChart, Grid3X3,
  ChevronDown, Layers,
} from "lucide-react";
import { useAuth } from "@/src/app/context/AuthContext";
import {
  fetchAllStocks, fetchIndices,
  type StockSummary, type IndexSummary,
} from "@/src/app/lib/api";
import { getCachedSummaries, getCachedIndices } from "@/src/app/lib/stockCache";
import StockModal, { type Stock } from "@/components/ui/StockModal";
import { SECTORS, ALL_SECTORS, filterBySector } from "@/src/app/lib/sectorData";

// ─── Top nav ──────────────────────────────────────────────────────────────────
function Navbar({ displayName, onLogout }: { displayName: string; onLogout: () => void }) {
  const router = useRouter();
  const [active, setActive] = useState("Dashboard");

  const navItems = [
    { label: "Dashboard", icon: LayoutDashboard, href: "/dashboard" },
    { label: "Analysis",  icon: LineChart,       href: "/dashboard/analysis" },
    { label: "Model",     icon: BarChart2,       href: "/dashboard/model" },
    { label: "Heatmap",   icon: Grid3X3,         href: "/dashboard/heatmap" },
  ];

  return (
    <header
      className="fixed top-0 left-0 right-0 z-40 flex items-center justify-between px-6 h-14"
      style={{
        background:   "hsl(215, 25%, 10%)",
        borderBottom: "1px solid hsl(215, 20%, 17%)",
      }}
    >
      {/* Logo */}
      <div className="flex items-center gap-2 flex-shrink-0">
        <div
          className="w-7 h-7 rounded-md flex items-center justify-center"
          style={{ background: "hsl(217, 91%, 60%)" }}
        >
          <TrendingUp className="w-3.5 h-3.5 text-white" strokeWidth={2.5} />
        </div>
        <span className="text-base font-bold" style={{ color: "hsl(217, 91%, 60%)" }}>
          LagLens
        </span>
      </div>

      {/* Nav links */}
      <nav className="flex items-center gap-1">
        {navItems.map(({ label, icon: Icon, href }) => (
          <button
            key={label}
            onClick={() => { setActive(label); router.push(href); }}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all"
            style={{
              background: active === label ? "hsl(217, 91%, 60% / 0.15)" : "transparent",
              color:      active === label ? "hsl(217, 91%, 70%)"         : "hsl(215, 15%, 55%)",
            }}
          >
            <Icon className="w-3.5 h-3.5" />
            {label}
          </button>
        ))}
      </nav>

      {/* User + logout */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => router.push("/dashboard/profile")}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-all"
          style={{
            background: "hsl(215, 25%, 15%)",
            border:     "1px solid hsl(215, 20%, 22%)",
            color:      "hsl(210, 40%, 85%)",
          }}
        >
          <User className="w-3.5 h-3.5" style={{ color: "hsl(215, 15%, 55%)" }} />
          {displayName}
        </button>
        <button
          onClick={onLogout}
          className="p-2 rounded-lg transition-all hover:opacity-70"
          style={{ color: "hsl(215, 15%, 50%)" }}
          aria-label="Sign out"
        >
          <LogOut className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
}

// ─── Index card ───────────────────────────────────────────────────────────────
function IndexCard({ idx }: { idx: IndexSummary }) {
  return (
    <div
      className="rounded-xl px-5 py-4 flex-1 min-w-0"
      style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
    >
      <p className="text-xs font-medium mb-1.5" style={{ color: "hsl(215, 15%, 50%)" }}>
        {idx.symbol}
      </p>
      <p className="text-2xl font-bold mb-1" style={{ color: "hsl(210, 40%, 92%)" }}>
        {idx.value}
      </p>
      <div className="flex items-center gap-1.5">
        {idx.positive
          ? <TrendingUp  className="w-3 h-3" style={{ color: "hsl(142, 71%, 45%)" }} />
          : <TrendingDown className="w-3 h-3" style={{ color: "hsl(0, 84%, 60%)" }} />}
        <p
          className="text-sm font-semibold"
          style={{ color: idx.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)" }}
        >
          {idx.change} ({idx.pct})
        </p>
        <div className="ml-auto">
          <svg width="60" height="20" viewBox="0 0 60 20">
            <line
              x1="0"  y1={idx.positive ? "16" : "4"}
              x2="60" y2={idx.positive ? "4"  : "16"}
              stroke={idx.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)"}
              strokeWidth="1.5"
            />
          </svg>
        </div>
      </div>
    </div>
  );
}

// ─── Skeleton row ─────────────────────────────────────────────────────────────
function SkeletonRow() {
  return (
    <div
      className="grid px-5 py-4 items-center animate-pulse"
      style={{
        gridTemplateColumns: "1fr 1fr 1fr 1fr 36px",
        borderTop:           "1px solid hsl(215, 20%, 16%)",
      }}
    >
      {[140, 80, 70, 60].map((w, i) => (
        <div
          key={i}
          className="h-3 rounded"
          style={{ background: "hsl(215, 20%, 18%)", width: w, maxWidth: "80%" }}
        />
      ))}
      <div />
    </div>
  );
}

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

  // Close on outside click
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
          background: value !== ALL_SECTORS ? "hsla(217,91%,60%,0.15)" : "hsl(215, 25%, 8%)",
          border:     value !== ALL_SECTORS ? "1px solid hsla(217,91%,60%,0.4)" : "1px solid hsl(215, 20%, 20%)",
          color:      value !== ALL_SECTORS ? "hsl(217, 91%, 70%)" : "hsl(210, 40%, 75%)",
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
            background:  "hsl(215, 25%, 11%)",
            border:      "1px solid hsl(215, 20%, 20%)",
            boxShadow:   "0 8px 24px rgba(0,0,0,0.4)",
            minWidth:    180,
            maxHeight:   320,
          }}
        >
          {SECTORS.map((sector) => (
            <button
              key={sector}
              onClick={() => { onChange(sector); setOpen(false); }}
              className="w-full text-left px-4 py-2.5 text-sm transition-colors"
              style={{
                background: value === sector ? "hsla(217,91%,60%,0.15)" : "transparent",
                color:      value === sector ? "hsl(217, 91%, 70%)" : "hsl(210, 40%, 80%)",
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

// ─── Main dashboard ───────────────────────────────────────────────────────────
export default function DashboardPage() {
  const { user, loading: authLoading, isSaved, toggleSave, trackClick, logout } = useAuth();
  const router = useRouter();

  const [stocks,        setStocks]        = useState<StockSummary[]>([]);
  const [indices,       setIndices]       = useState<IndexSummary[]>([]);
  const [search,        setSearch]        = useState("");
  const [selectedSector, setSelectedSector] = useState(ALL_SECTORS);
  const [dataLoading,   setDataLoading]   = useState(true);
  const [selectedStock, setSelectedStock] = useState<Stock | null>(null);

  // Auth guard
  useEffect(() => {
    if (!authLoading && !user) router.replace("/");
  }, [user, authLoading, router]);

  // Load data — Firestore cache first, then API
  const loadData = useCallback(async () => {
    setDataLoading(true);
    try {
      const [cachedStocks, cachedIndices] = await Promise.all([
        getCachedSummaries(),
        getCachedIndices(),
      ]);
      if (cachedStocks && !cachedStocks.stale) setStocks(cachedStocks.data);
      if (cachedIndices && !cachedIndices.stale) setIndices(cachedIndices.data);

      const [freshStocks, freshIndices] = await Promise.all([
        fetchAllStocks(),
        fetchIndices(),
      ]);
      setStocks(freshStocks);
      setIndices(freshIndices);
    } catch {
      // keep whatever we loaded from cache
    } finally {
      setDataLoading(false);
    }
  }, []);

  useEffect(() => {
    if (user) loadData();
  }, [user, loadData]);

  const handleLogout = async () => {
    await logout();
    document.cookie = "ll_authed=; Max-Age=0; path=/";
    router.push("/");
  };

  const handleRowClick = (stock: StockSummary) => {
    setSelectedStock(stock);
    trackClick(stock.symbol);
  };

  const displayName = user?.displayName || user?.email?.split("@")[0] || "User";

  // Sector-filtered base set
  const sectorStocks = filterBySector(stocks, selectedSector);

  // Filtered watchlist (sector + search)
  const watchlist = sectorStocks.filter(
    (s) =>
      s.symbol.toLowerCase().includes(search.toLowerCase()) ||
      s.name.toLowerCase().includes(search.toLowerCase()),
  );

  // Gainers / Losers (sector-scoped)
  const sorted = [...sectorStocks].sort((a, b) => {
    const pa = parseFloat(a.change.replace(/[^-\d.]/g, ""));
    const pb = parseFloat(b.change.replace(/[^-\d.]/g, ""));
    return pb - pa;
  });
  const topGainers = sorted.filter((s) => s.positive).slice(0, 5);
  const topLosers  = sorted.filter((s) => !s.positive).slice(-5).reverse();

  return (
    <div className="min-h-screen" style={{ background: "hsl(213, 27%, 7%)" }}>
      <Navbar displayName={displayName} onLogout={handleLogout} />

      <main className="pt-14">
        <div className="max-w-7xl mx-auto px-6 py-6">

          {/* ── Index Cards ── */}
          <div className="flex gap-4 mb-6">
            {indices.length > 0
              ? indices.map((idx) => <IndexCard key={idx.symbol} idx={idx} />)
              : [1, 2, 3].map((i) => (
                  <div
                    key={i}
                    className="rounded-xl px-5 py-4 flex-1 animate-pulse"
                    style={{
                      background: "hsl(215, 25%, 11%)",
                      border:     "1px solid hsl(215, 20%, 18%)",
                      minHeight:  90,
                    }}
                  />
                ))}
          </div>

          {/* ── Watchlist + Side panels ── */}
          <div className="flex gap-4">

            {/* Watchlist table */}
            <div
              className="flex-1 rounded-xl"
              style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)", overflow: "visible" }}
            >
              {/* Header */}
              <div
                className="flex items-center justify-between px-5 py-4 gap-3 flex-wrap"
                style={{ borderBottom: "1px solid hsl(215, 20%, 18%)" }}
              >
                <h2 className="text-base font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>
                  Watchlist
                </h2>
                <div className="flex items-center gap-2 ml-auto">
                  {/* Sector dropdown */}
                  <SectorDropdown value={selectedSector} onChange={setSelectedSector} />

                  {/* Search */}
                  <div className="relative">
                    <Search
                      className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5"
                      style={{ color: "hsl(215, 15%, 45%)" }}
                    />
                    <input
                      type="search"
                      placeholder="Search"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      className="pl-9 pr-4 py-1.5 rounded-lg text-sm outline-none transition-all"
                      style={{
                        background: "hsl(215, 25%, 8%)",
                        border:     "1px solid hsl(215, 20%, 20%)",
                        color:      "hsl(210, 40%, 85%)",
                        width:      160,
                      }}
                      onFocus={(e) => (e.currentTarget.style.borderColor = "hsl(217, 91%, 60%)")}
                      onBlur={(e)  => (e.currentTarget.style.borderColor = "hsl(215, 20%, 20%)")}
                    />
                  </div>
                </div>
              </div>

              {/* Column headers */}
              <div
                className="grid px-5 py-2.5 text-xs font-medium"
                style={{
                  gridTemplateColumns: "1fr 1fr 1fr 1fr 36px",
                  color:               "hsl(215, 15%, 45%)",
                  borderBottom:        "1px solid hsl(215, 20%, 16%)",
                }}
              >
                <span>Ticker</span>
                <span className="text-right">Price</span>
                <span className="text-right">Change</span>
                <span className="text-right">Volume</span>
                <span />
              </div>

              {/* Rows — NO sparklines */}
              {dataLoading
                ? Array.from({ length: 8 }).map((_, i) => <SkeletonRow key={i} />)
                : watchlist.map((stock) => {
                    const saved = isSaved(stock.symbol);
                    return (
                      <div
                        key={stock.symbol}
                        onClick={() => handleRowClick(stock)}
                        className="grid px-5 py-3.5 items-center cursor-pointer transition-colors"
                        style={{
                          gridTemplateColumns: "1fr 1fr 1fr 1fr 36px",
                          borderTop:           "1px solid hsl(215, 20%, 16%)",
                        }}
                        onMouseEnter={(e) => (e.currentTarget.style.background = "hsl(215, 25%, 14%)")}
                        onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                      >
                        {/* Ticker + company name */}
                        <div className="flex flex-col min-w-0">
                          <span
                            className="text-sm font-semibold"
                            style={{ color: "hsl(210, 40%, 92%)" }}
                          >
                            {stock.symbol}
                          </span>
                          <span
                            className="text-xs truncate"
                            style={{ color: "hsl(215, 15%, 50%)" }}
                          >
                            {stock.name}
                          </span>
                        </div>

                        <span
                          className="text-sm font-medium text-right"
                          style={{ color: "hsl(210, 40%, 92%)" }}
                        >
                          {stock.price}
                        </span>

                        <span
                          className="text-sm font-semibold text-right"
                          style={{ color: stock.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)" }}
                        >
                          {stock.change}
                        </span>

                        <span
                          className="text-sm text-right"
                          style={{ color: "hsl(215, 15%, 55%)" }}
                        >
                          {stock.volume}
                        </span>

                        {/* Star */}
                        <div className="flex justify-end">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleSave({ symbol: stock.symbol, name: stock.name });
                            }}
                            className="transition-all hover:scale-110"
                            aria-label={saved ? "Unsave stock" : "Save stock"}
                          >
                            <Star
                              className="w-4 h-4"
                              style={
                                saved
                                  ? { fill: "hsl(48, 96%, 53%)", color: "hsl(48, 96%, 53%)" }
                                  : { fill: "transparent",       color: "hsl(215, 15%, 38%)" }
                              }
                            />
                          </button>
                        </div>
                      </div>
                    );
                  })}
            </div>

            {/* Right column */}
            <div className="w-60 flex flex-col gap-4">

              {/* Top Gainers */}
              <div
                className="rounded-xl overflow-hidden"
                style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
              >
                <div
                  className="px-5 py-3 flex items-center gap-2"
                  style={{ borderBottom: "1px solid hsl(215, 20%, 18%)" }}
                >
                  <TrendingUp className="w-4 h-4" style={{ color: "hsl(142, 71%, 45%)" }} />
                  <h3 className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>
                    Top Gainers
                  </h3>
                </div>
                {dataLoading
                  ? Array.from({ length: 5 }).map((_, i) => (
                      <div key={i} className="flex items-center justify-between px-5 py-3 animate-pulse">
                        <div className="h-3 w-10 rounded" style={{ background: "hsl(215, 20%, 18%)" }} />
                        <div className="h-3 w-12 rounded" style={{ background: "hsl(215, 20%, 18%)" }} />
                      </div>
                    ))
                  : topGainers.map((s) => (
                      <div
                        key={s.symbol}
                        className="flex items-center justify-between px-5 py-3 cursor-pointer transition-colors"
                        style={{ borderTop: "1px solid hsl(215, 20%, 16%)" }}
                        onClick={() => handleRowClick(s)}
                        onMouseEnter={(e) => (e.currentTarget.style.background = "hsl(215, 25%, 14%)")}
                        onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                      >
                        <span className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 85%)" }}>
                          {s.symbol}
                        </span>
                        <span className="text-sm font-semibold" style={{ color: "hsl(142, 71%, 45%)" }}>
                          {s.change}
                        </span>
                      </div>
                    ))}
              </div>

              {/* Top Losers */}
              <div
                className="rounded-xl overflow-hidden"
                style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
              >
                <div
                  className="px-5 py-3 flex items-center gap-2"
                  style={{ borderBottom: "1px solid hsl(215, 20%, 18%)" }}
                >
                  <TrendingDown className="w-4 h-4" style={{ color: "hsl(0, 84%, 60%)" }} />
                  <h3 className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>
                    Top Losers
                  </h3>
                </div>
                {dataLoading
                  ? Array.from({ length: 5 }).map((_, i) => (
                      <div key={i} className="flex items-center justify-between px-5 py-3 animate-pulse">
                        <div className="h-3 w-10 rounded" style={{ background: "hsl(215, 20%, 18%)" }} />
                        <div className="h-3 w-12 rounded" style={{ background: "hsl(215, 20%, 18%)" }} />
                      </div>
                    ))
                  : topLosers.map((s) => (
                      <div
                        key={s.symbol}
                        className="flex items-center justify-between px-5 py-3 cursor-pointer transition-colors"
                        style={{ borderTop: "1px solid hsl(215, 20%, 16%)" }}
                        onClick={() => handleRowClick(s)}
                        onMouseEnter={(e) => (e.currentTarget.style.background = "hsl(215, 25%, 14%)")}
                        onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                      >
                        <span className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 85%)" }}>
                          {s.symbol}
                        </span>
                        <span className="text-sm font-semibold" style={{ color: "hsl(0, 84%, 60%)" }}>
                          {s.change}
                        </span>
                      </div>
                    ))}
              </div>

            </div>
          </div>
        </div>
      </main>

      {/* ── Stock Modal ── */}
      {selectedStock && (
        <StockModal
          stock={selectedStock}
          onClose={() => setSelectedStock(null)}
        />
      )}
    </div>
  );
}