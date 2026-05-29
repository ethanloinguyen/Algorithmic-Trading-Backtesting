// Frontend/src/app/dashboard/profile/page.tsx
"use client";
import { useState, useEffect, useRef, useMemo } from "react";
import { useRouter } from "next/navigation";
import {
  Star, User, Mail, LogOut, Loader2,
  ChevronDown, Layers,
  FolderPlus, Trash2, X, Plus, Briefcase, AlertTriangle, Eye, EyeOff, Pencil, Check,
} from "lucide-react";
import Sidebar from "@/components/ui/Sidebar";
import { useAuth } from "@/src/app/context/AuthContext";
import { fetchStockSummaries, fetchAllStocks, type StockSummary } from "@/src/app/lib/api";
import StockModal, { type Stock } from "@/components/ui/StockModal";
import { auth, db } from "@/src/app/lib/firebase";
import { reauthenticateWithCredential, EmailAuthProvider, deleteUser } from "firebase/auth";
import { doc, deleteDoc } from "firebase/firestore";
import { SECTORS, ALL_SECTORS, filterBySector } from "@/src/app/lib/sectorData";
import { PageHelp } from "@/components/ui/PageHelp";

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
  const { user, savedStocks, toggleSave, trackClick, loading: authLoading, logout, customPortfolios, createPortfolio, updatePortfolio, deletePortfolio } = useAuth();
  const router = useRouter();

  const [liveData,       setLiveData]       = useState<StockSummary[]>([]);
  const [pricesLoading,  setPricesLoading]  = useState(false);
  const [pricesError,    setPricesError]    = useState(false);
  const [selectedSector, setSelectedSector] = useState(ALL_SECTORS);

  // Delete account modal state
  const [showDeleteModal,   setShowDeleteModal]   = useState(false);
  const [deletePassword,    setDeletePassword]    = useState("");
  const [deleteError,       setDeleteError]       = useState("");
  const [deletingAccount,   setDeletingAccount]   = useState(false);
  const [showDeletePw,      setShowDeletePw]      = useState(false);
  // Prevents the unauthenticated-redirect useEffect from racing with our own navigation
  const isDeletingRef = useRef(false);

  const handleDeleteAccount = async () => {
    const currentUser = auth.currentUser;
    if (!currentUser || !currentUser.email) return;

    setDeleteError("");
    setDeletingAccount(true);
    isDeletingRef.current = true;

    // Reject after ms — used to prevent any single Firebase call from hanging forever
    const deadline = (ms: number) =>
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(Object.assign(new Error("timeout"), { code: "app/timeout" })), ms)
      );

    // Step 1: Re-authenticate — only step that shows user-visible errors on failure
    try {
      const credential = EmailAuthProvider.credential(currentUser.email, deletePassword);
      await Promise.race([
        reauthenticateWithCredential(currentUser, credential),
        deadline(10_000),
      ]);
    } catch (err: unknown) {
      isDeletingRef.current = false;
      setDeletingAccount(false);
      const code = (err as { code?: string })?.code ?? "";
      if (code === "auth/wrong-password" || code === "auth/invalid-credential") {
        setDeleteError("Incorrect password. Please try again.");
      } else if (code === "auth/too-many-requests") {
        setDeleteError("Too many attempts. Please try again later.");
      } else if (code === "app/timeout") {
        setDeleteError("Request timed out. Check your connection and try again.");
      } else {
        setDeleteError("Failed to verify password. Please try again.");
      }
      return;
    }

    const uid = currentUser.uid;

    // Step 2: Delete Firestore data and Firebase Auth account concurrently.
    // Each is raced against a deadline so a hung operation never blocks navigation.
    await Promise.allSettled([
      Promise.race([deleteDoc(doc(db, "users", uid)), deadline(5_000)]),
      Promise.race([deleteUser(currentUser), deadline(5_000)]),
    ]);

    // Step 3: Hard-navigate — page fully reloads, auth cookie cleared
    document.cookie = "ll_authed=; Max-Age=0; path=/";
    window.location.href = "/";
  };

  // Custom portfolio form state
  const [portfolioName,    setPortfolioName]    = useState("");
  const [portfolioTickers, setPortfolioTickers] = useState<string[]>([]);
  const [portfolioInput,   setPortfolioInput]   = useState("");
  const [savingPortfolio,  setSavingPortfolio]  = useState(false);
  const [allStocks,        setAllStocks]        = useState<StockSummary[]>([]);
  const [showSuggestions,  setShowSuggestions]  = useState(false);
  const [activeSuggestion, setActiveSuggestion] = useState(-1);
  const portfolioInputRef  = useRef<HTMLInputElement>(null);
  const suggestionsRef     = useRef<HTMLDivElement>(null);

  // Stock modal
  const [modalStock, setModalStock] = useState<Stock | null>(null);

  // Edit portfolio state
  const [editingPortfolioId,   setEditingPortfolioId]   = useState<string | null>(null);
  const [editTickers,          setEditTickers]          = useState<string[]>([]);
  const [editInput,            setEditInput]            = useState("");
  const [editShowSuggestions,  setEditShowSuggestions]  = useState(false);
  const [editActiveSuggestion, setEditActiveSuggestion] = useState(-1);
  const [savingEdit,           setSavingEdit]           = useState(false);
  const editInputRef   = useRef<HTMLInputElement>(null);
  const editSuggestRef = useRef<HTMLDivElement>(null);

  // Redirect unauthenticated (skip during account deletion — handled by window.location)
  useEffect(() => {
    if (!authLoading && !user && !isDeletingRef.current) router.replace("/");
  }, [user, authLoading, router]);

  // Load all stocks for ticker autocomplete
  useEffect(() => {
    fetchAllStocks().then(setAllStocks).catch(() => {});
  }, []);

  const tickerSuggestions = useMemo(() => {
    const q = portfolioInput.trim().toUpperCase();
    if (!q || q.length < 1) return [];
    return allStocks
      .filter(
        (s) =>
          !portfolioTickers.includes(s.symbol) &&
          (s.symbol.startsWith(q) || s.name.toUpperCase().includes(q))
      )
      .slice(0, 8);
  }, [portfolioInput, allStocks, portfolioTickers]);

  const editTickerSuggestions = useMemo(() => {
    const q = editInput.trim().toUpperCase();
    if (!q || q.length < 1) return [];
    return allStocks
      .filter(
        (s) =>
          !editTickers.includes(s.symbol) &&
          (s.symbol.startsWith(q) || s.name.toUpperCase().includes(q))
      )
      .slice(0, 8);
  }, [editInput, allStocks, editTickers]);

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
      <Sidebar />

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
            <div className="flex items-center gap-3 flex-shrink-0">
              <PageHelp
                title="Profile Page Guide"
                subtitle="Manage your watchlist and create portfolios for the Diversify page."
                sections={[
                  {
                    title: "Saved Stocks Watchlist",
                    body: "Stocks you've starred anywhere in the app appear here with live prices, daily % change, and volume. Use the sector dropdown in the Saved Stocks header to filter your watchlist by industry.",
                  },
                  {
                    title: "How to Save a Stock",
                    body: "Go to the Model page, search for any stock, click Analyze, then click the star icon next to any leader in the results table. You can also star stocks from the Home dashboard. Stars are synced to your account across sessions.",
                    color: "hsl(48, 96%, 53%)",
                  },
                  {
                    title: "Account Stats",
                    body: "The three stat cards show: Saved Stocks (total count in your watchlist), Avg. Change (average daily % change across sector-filtered stocks with live data), and Watchlist status (Active or Empty).",
                    color: "hsl(217, 91%, 60%)",
                  },
                  {
                    title: "Creating a Custom Portfolio",
                    body: "In the Custom Portfolios section at the bottom, enter a portfolio name (e.g. 'Growth Picks'), then type stock tickers separated by Enter or comma. Use the 'Add from saved' chips to quickly pull in your watchlist stocks, then click Save Portfolio.",
                    color: "hsl(142, 71%, 45%)",
                  },
                  {
                    title: "Using Portfolios on the Diversify Page",
                    body: "Once saved, your portfolios appear in the 'Add Portfolio' dropdown on the Diversify page. Click one to instantly load all of its tickers into the analysis input — no need to re-type them every time.",
                    color: "hsl(270, 70%, 65%)",
                  },
                ]}
              />
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium hover:opacity-80 transition-opacity"
                style={{ background: "hsl(215,25%,16%)", border: "1px solid hsl(215,20%,22%)", color: "hsl(215,15%,65%)" }}
              >
                <LogOut className="w-4 h-4" /> Sign Out
              </button>
            </div>
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
                    onClick={() => { trackClick(stock.symbol); setModalStock(stock); }}
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
                {customPortfolios.map((portfolio) => {
                  const isEditing = editingPortfolioId === portfolio.id;
                  return (
                    <div
                      key={portfolio.id}
                      className="rounded-xl px-5 py-4"
                      style={{ background: "hsl(215,25%,11%)", border: `1px solid ${isEditing ? "hsl(217,91%,45%)" : "hsl(215,20%,18%)"}` }}
                    >
                      {/* Portfolio header row */}
                      <div className="flex items-center justify-between gap-4 mb-2">
                        <p className="text-sm font-semibold" style={{ color: "hsl(210,40%,92%)" }}>
                          {portfolio.name}
                        </p>
                        <div className="flex items-center gap-2 flex-shrink-0">
                          {!isEditing && (
                            <button
                              onClick={() => {
                                setEditingPortfolioId(portfolio.id);
                                setEditTickers([...portfolio.tickers]);
                                setEditInput("");
                                setEditShowSuggestions(false);
                                setEditActiveSuggestion(-1);
                              }}
                              className="p-2 rounded-lg transition-all hover:opacity-80"
                              style={{ color: "hsl(217,91%,60%)", background: "hsla(217,91%,60%,0.1)" }}
                              aria-label="Edit portfolio"
                            >
                              <Pencil className="w-4 h-4" />
                            </button>
                          )}
                          <button
                            onClick={() => deletePortfolio(portfolio.id)}
                            className="p-2 rounded-lg transition-all hover:opacity-70"
                            style={{ color: "hsl(0,84%,60%)", background: "hsla(0,84%,60%,0.1)" }}
                            aria-label="Delete portfolio"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>

                      {/* View mode: ticker chips */}
                      {!isEditing && (
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
                      )}

                      {/* Edit mode: inline ticker tag input */}
                      {isEditing && (
                        <div>
                          <div className="relative mb-3">
                            <div
                              className="flex flex-wrap gap-2 min-h-10 p-2 rounded-lg cursor-text"
                              style={{ background: "hsl(215,25%,8%)", border: "1px solid hsl(217,91%,45%)" }}
                              onClick={() => editInputRef.current?.focus()}
                            >
                              {editTickers.map((t) => (
                                <span
                                  key={t}
                                  className="flex items-center gap-1 px-2.5 py-0.5 rounded-md text-xs font-semibold"
                                  style={{ background: "hsla(217,91%,60%,0.15)", color: "hsl(217,91%,70%)" }}
                                >
                                  {t}
                                  <button
                                    onClick={(e) => { e.stopPropagation(); setEditTickers(prev => prev.filter(x => x !== t)); }}
                                    className="hover:opacity-60 ml-0.5"
                                  >
                                    <X className="w-3 h-3" />
                                  </button>
                                </span>
                              ))}
                              <input
                                ref={editInputRef}
                                value={editInput}
                                onChange={(e) => {
                                  setEditInput(e.target.value.toUpperCase());
                                  setEditActiveSuggestion(-1);
                                  setEditShowSuggestions(true);
                                }}
                                onFocus={() => setEditShowSuggestions(true)}
                                onBlur={() => {
                                  setTimeout(() => {
                                    setEditShowSuggestions(false);
                                    setEditActiveSuggestion(-1);
                                    const val = editInput.trim();
                                    if (val && !editTickers.includes(val) && val.length <= 10) {
                                      setEditTickers(prev => [...prev, val]);
                                      setEditInput("");
                                    }
                                  }, 150);
                                }}
                                onKeyDown={(e) => {
                                  if (e.key === "ArrowDown") { e.preventDefault(); setEditActiveSuggestion(i => Math.min(i + 1, editTickerSuggestions.length - 1)); return; }
                                  if (e.key === "ArrowUp")   { e.preventDefault(); setEditActiveSuggestion(i => Math.max(i - 1, -1)); return; }
                                  if (e.key === "Escape")    { setEditShowSuggestions(false); setEditActiveSuggestion(-1); return; }
                                  if (["Enter", ",", " ", "Tab"].includes(e.key)) {
                                    e.preventDefault();
                                    if (editActiveSuggestion >= 0 && editTickerSuggestions[editActiveSuggestion]) {
                                      const sym = editTickerSuggestions[editActiveSuggestion].symbol;
                                      if (!editTickers.includes(sym)) setEditTickers(prev => [...prev, sym]);
                                      setEditInput(""); setEditShowSuggestions(false); setEditActiveSuggestion(-1);
                                      return;
                                    }
                                    const val = editInput.trim();
                                    if (val && !editTickers.includes(val) && val.length <= 10) setEditTickers(prev => [...prev, val]);
                                    setEditInput(""); setEditShowSuggestions(false); setEditActiveSuggestion(-1);
                                    return;
                                  }
                                  if (e.key === "Backspace" && !editInput && editTickers.length) setEditTickers(prev => prev.slice(0, -1));
                                }}
                                placeholder={editTickers.length === 0 ? "Type tickers, press Enter or comma" : ""}
                                className="flex-1 min-w-24 bg-transparent outline-none text-sm"
                                style={{ color: "hsl(210,40%,85%)" }}
                              />
                            </div>

                            {/* Autocomplete dropdown */}
                            {editShowSuggestions && editTickerSuggestions.length > 0 && (
                              <div
                                ref={editSuggestRef}
                                className="absolute left-0 right-0 top-full mt-1 rounded-xl overflow-hidden z-50"
                                style={{ background: "hsl(215,25%,11%)", border: "1px solid hsl(215,20%,20%)", boxShadow: "0 8px 24px rgba(0,0,0,0.4)" }}
                              >
                                {editTickerSuggestions.map((s, i) => (
                                  <button
                                    key={s.symbol}
                                    onMouseDown={(e) => {
                                      e.preventDefault();
                                      if (!editTickers.includes(s.symbol)) setEditTickers(prev => [...prev, s.symbol]);
                                      setEditInput(""); setEditShowSuggestions(false); setEditActiveSuggestion(-1);
                                      editInputRef.current?.focus();
                                    }}
                                    className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-left transition-colors"
                                    style={{
                                      background:   i === editActiveSuggestion ? "hsl(215,25%,16%)" : "transparent",
                                      borderBottom: i < editTickerSuggestions.length - 1 ? "1px solid hsl(215,20%,16%)" : "none",
                                    }}
                                    onMouseEnter={() => setEditActiveSuggestion(i)}
                                    onMouseLeave={() => setEditActiveSuggestion(-1)}
                                  >
                                    <span className="font-semibold" style={{ color: "hsl(217,91%,70%)", minWidth: 56 }}>{s.symbol}</span>
                                    <span className="truncate text-xs" style={{ color: "hsl(215,15%,55%)" }}>{s.name}</span>
                                  </button>
                                ))}
                              </div>
                            )}
                          </div>

                          {/* Add from saved shortcut */}
                          {savedStocks.length > 0 && (
                            <div className="flex items-center gap-2 mb-3 flex-wrap">
                              <span className="text-xs" style={{ color: "hsl(215,15%,45%)" }}>Add from saved:</span>
                              {savedStocks.slice(0, 8).map((s) => (
                                <button
                                  key={s.symbol}
                                  onClick={() => { if (!editTickers.includes(s.symbol)) setEditTickers(prev => [...prev, s.symbol]); }}
                                  disabled={editTickers.includes(s.symbol)}
                                  className="text-xs px-2 py-0.5 rounded-md transition-colors"
                                  style={{
                                    background: editTickers.includes(s.symbol) ? "hsla(217,91%,60%,0.08)" : "hsl(215,25%,14%)",
                                    border: "1px solid hsl(215,20%,20%)",
                                    color: editTickers.includes(s.symbol) ? "hsl(215,15%,38%)" : "hsl(210,40%,75%)",
                                    cursor: editTickers.includes(s.symbol) ? "default" : "pointer",
                                  }}
                                >
                                  <Plus className="w-2.5 h-2.5 inline mr-1" style={{ position: "relative", top: -0.5 }} />
                                  {s.symbol}
                                </button>
                              ))}
                            </div>
                          )}

                          {/* Save / Cancel */}
                          <div className="flex items-center gap-2">
                            <button
                              onClick={async () => {
                                const finalTickers = editInput.trim()
                                  ? [...editTickers, editInput.trim().toUpperCase()].filter((v, i, a) => a.indexOf(v) === i)
                                  : editTickers;
                                if (finalTickers.length === 0) return;
                                setSavingEdit(true);
                                await updatePortfolio(portfolio.id, portfolio.name, finalTickers);
                                setSavingEdit(false);
                                setEditingPortfolioId(null);
                              }}
                              disabled={savingEdit || editTickers.length === 0}
                              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold disabled:opacity-40 transition-opacity hover:opacity-80"
                              style={{ background: "hsl(217,91%,60%)", color: "white" }}
                            >
                              {savingEdit
                                ? <><Loader2 className="w-4 h-4 animate-spin" /> Saving…</>
                                : <><Check className="w-4 h-4" /> Save Changes</>}
                            </button>
                            <button
                              onClick={() => { setEditingPortfolioId(null); setEditInput(""); }}
                              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium hover:opacity-80 transition-opacity"
                              style={{ background: "hsl(215,25%,16%)", border: "1px solid hsl(215,20%,22%)", color: "hsl(215,15%,65%)" }}
                            >
                              <X className="w-4 h-4" /> Cancel
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
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

              {/* Ticker tag input with autocomplete dropdown */}
              <div className="relative mb-3">
                <div
                  className="flex flex-wrap gap-2 min-h-10 p-2 rounded-lg cursor-text"
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
                    onChange={(e) => {
                      setPortfolioInput(e.target.value.toUpperCase());
                      setActiveSuggestion(-1);
                      setShowSuggestions(true);
                    }}
                    onFocus={() => setShowSuggestions(true)}
                    onBlur={() => {
                      // Delay so suggestion clicks register first
                      setTimeout(() => {
                        setShowSuggestions(false);
                        setActiveSuggestion(-1);
                        const val = portfolioInput.trim();
                        if (val && !portfolioTickers.includes(val) && val.length <= 10) {
                          setPortfolioTickers(prev => [...prev, val]);
                          setPortfolioInput("");
                        }
                      }, 150);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "ArrowDown") {
                        e.preventDefault();
                        setActiveSuggestion(i => Math.min(i + 1, tickerSuggestions.length - 1));
                        return;
                      }
                      if (e.key === "ArrowUp") {
                        e.preventDefault();
                        setActiveSuggestion(i => Math.max(i - 1, -1));
                        return;
                      }
                      if (e.key === "Escape") {
                        setShowSuggestions(false);
                        setActiveSuggestion(-1);
                        return;
                      }
                      if (["Enter", ",", " ", "Tab"].includes(e.key)) {
                        e.preventDefault();
                        if (activeSuggestion >= 0 && tickerSuggestions[activeSuggestion]) {
                          const sym = tickerSuggestions[activeSuggestion].symbol;
                          if (!portfolioTickers.includes(sym)) {
                            setPortfolioTickers(prev => [...prev, sym]);
                          }
                          setPortfolioInput("");
                          setShowSuggestions(false);
                          setActiveSuggestion(-1);
                          return;
                        }
                        const val = portfolioInput.trim();
                        if (val && !portfolioTickers.includes(val) && val.length <= 10) {
                          setPortfolioTickers(prev => [...prev, val]);
                        }
                        setPortfolioInput("");
                        setShowSuggestions(false);
                        setActiveSuggestion(-1);
                        return;
                      }
                      if (e.key === "Backspace" && !portfolioInput && portfolioTickers.length) {
                        setPortfolioTickers(prev => prev.slice(0, -1));
                      }
                    }}
                    placeholder={portfolioTickers.length === 0 ? "Type tickers, press Enter or comma" : ""}
                    className="flex-1 min-w-24 bg-transparent outline-none text-sm"
                    style={{ color: "hsl(210,40%,85%)" }}
                  />
                </div>

                {/* Autocomplete dropdown */}
                {showSuggestions && tickerSuggestions.length > 0 && (
                  <div
                    ref={suggestionsRef}
                    className="absolute left-0 right-0 top-full mt-1 rounded-xl overflow-hidden z-50"
                    style={{
                      background: "hsl(215,25%,11%)",
                      border:     "1px solid hsl(215,20%,20%)",
                      boxShadow:  "0 8px 24px rgba(0,0,0,0.4)",
                    }}
                  >
                    {tickerSuggestions.map((s, i) => (
                      <button
                        key={s.symbol}
                        onMouseDown={(e) => {
                          e.preventDefault();
                          if (!portfolioTickers.includes(s.symbol)) {
                            setPortfolioTickers(prev => [...prev, s.symbol]);
                          }
                          setPortfolioInput("");
                          setShowSuggestions(false);
                          setActiveSuggestion(-1);
                          portfolioInputRef.current?.focus();
                        }}
                        className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-left transition-colors"
                        style={{
                          background:   i === activeSuggestion ? "hsl(215,25%,16%)" : "transparent",
                          borderBottom: i < tickerSuggestions.length - 1 ? "1px solid hsl(215,20%,16%)" : "none",
                        }}
                        onMouseEnter={() => setActiveSuggestion(i)}
                        onMouseLeave={() => setActiveSuggestion(-1)}
                      >
                        <span className="font-semibold" style={{ color: "hsl(217,91%,70%)", minWidth: 56 }}>
                          {s.symbol}
                        </span>
                        <span className="truncate text-xs" style={{ color: "hsl(215,15%,55%)" }}>
                          {s.name}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
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
                  await createPortfolio(portfolioName, finalTickers);
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

          {/* ── Delete Account ── */}
          <div className="mt-6 rounded-xl p-5" style={{ background: "hsl(215,25%,11%)", border: "1px solid hsl(0,84%,30%)" }}>
            <div className="flex items-center gap-2 mb-1">
              <AlertTriangle className="w-4 h-4" style={{ color: "hsl(0,84%,60%)" }} />
              <h2 className="text-base font-semibold" style={{ color: "hsl(0,84%,60%)" }}>
                Delete Account
              </h2>
            </div>
            <p className="text-xs mb-4" style={{ color: "hsl(215,15%,45%)" }}>
              Permanently delete your account and all associated data. This action cannot be undone.
            </p>
            <button
              onClick={() => { setShowDeleteModal(true); setDeletePassword(""); setDeleteError(""); }}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-opacity hover:opacity-80"
              style={{ background: "hsla(0,84%,60%,0.12)", border: "1px solid hsl(0,84%,35%)", color: "hsl(0,84%,65%)" }}
            >
              <Trash2 className="w-4 h-4" /> Delete My Account
            </button>
          </div>

        </div>
      </main>

      {/* ── Stock modal ── */}
      {modalStock && (
        <StockModal stock={modalStock} onClose={() => setModalStock(null)} />
      )}

      {/* ── Delete account modal ── */}
      {showDeleteModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: "rgba(0,0,0,0.7)" }}
          onClick={(e) => { if (e.target === e.currentTarget) setShowDeleteModal(false); }}
        >
          <div
            className="w-full max-w-md rounded-2xl p-6"
            style={{ background: "hsl(215,25%,11%)", border: "1px solid hsl(0,84%,30%)" }}
          >
            <div className="flex items-center gap-3 mb-4">
              <div
                className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0"
                style={{ background: "hsla(0,84%,60%,0.15)" }}
              >
                <AlertTriangle className="w-5 h-5" style={{ color: "hsl(0,84%,60%)" }} />
              </div>
              <div>
                <h3 className="text-base font-bold" style={{ color: "hsl(210,40%,92%)" }}>
                  Delete Account
                </h3>
                <p className="text-xs" style={{ color: "hsl(215,15%,50%)" }}>
                  This will permanently remove all your data.
                </p>
              </div>
            </div>

            <p className="text-sm mb-4" style={{ color: "hsl(215,15%,55%)" }}>
              Enter your password to confirm. Your saved stocks, portfolios, and account will be permanently deleted and cannot be recovered.
            </p>

            <div className="relative mb-4">
              <input
                type={showDeletePw ? "text" : "password"}
                placeholder="Enter your password"
                value={deletePassword}
                onChange={(e) => { setDeletePassword(e.target.value); setDeleteError(""); }}
                className="w-full px-4 py-2.5 pr-11 rounded-lg text-sm outline-none transition-all"
                style={{
                  background: "hsl(215,25%,8%)",
                  border: deleteError ? "1px solid hsl(0,84%,50%)" : "1px solid hsl(215,20%,22%)",
                  color: "hsl(210,40%,92%)",
                }}
                onFocus={(e) => { if (!deleteError) e.currentTarget.style.borderColor = "hsl(0,84%,50%)"; }}
                onBlur={(e) => { if (!deleteError) e.currentTarget.style.borderColor = "hsl(215,20%,22%)"; }}
                onKeyDown={(e) => { if (e.key === "Enter" && deletePassword) handleDeleteAccount(); }}
                autoFocus
              />
              <button
                type="button"
                onClick={() => setShowDeletePw(s => !s)}
                className="absolute right-3 top-1/2 -translate-y-1/2"
                style={{ color: "hsl(215,15%,55%)" }}
              >
                {showDeletePw ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>

            {deleteError && (
              <p
                className="text-xs rounded-lg px-3 py-2 mb-4"
                style={{ color: "hsl(0,84%,65%)", background: "hsl(0,84%,10%)", border: "1px solid hsl(0,84%,20%)" }}
              >
                {deleteError}
              </p>
            )}

            <div className="flex gap-3">
              <button
                onClick={() => setShowDeleteModal(false)}
                className="flex-1 py-2.5 rounded-lg text-sm font-medium transition-opacity hover:opacity-80"
                style={{ background: "hsl(215,25%,16%)", border: "1px solid hsl(215,20%,22%)", color: "hsl(215,15%,65%)" }}
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteAccount}
                disabled={deletingAccount || !deletePassword}
                className="flex-1 py-2.5 rounded-lg text-sm font-semibold transition-opacity hover:opacity-80 disabled:opacity-40 flex items-center justify-center gap-2"
                style={{ background: "hsl(0,84%,45%)", color: "white" }}
              >
                {deletingAccount
                  ? <><Loader2 className="w-4 h-4 animate-spin" /> Deleting…</>
                  : "Delete Account"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}