// Frontend/src/app/dashboard/profile/page.tsx
"use client";
import { useState, useEffect } from "react";
import Sidebar from "@/components/ui/Sidebar";
import StockModal, { type Stock } from "@/components/ui/StockModal";
import { useAuth } from "@/src/app/context/AuthContext";
import { fetchStockSummaries, type StockSummary } from "@/src/app/lib/api";
import { Star, User, Mail, LogOut, Loader2 } from "lucide-react";
import { useRouter } from "next/navigation";

const card = {
  background: "hsl(215, 25%, 11%)",
  border:     "1px solid hsl(215, 20%, 18%)",
};

export default function ProfilePage() {
  const { user, savedStocks, toggleSave, trackClick, loading: authLoading, logout } = useAuth();
  const router = useRouter();

  const [liveData,      setLiveData]      = useState<StockSummary[]>([]);
  const [pricesLoading, setPricesLoading] = useState(false);
  const [pricesError,   setPricesError]   = useState(false);
  const [selectedStock, setSelectedStock] = useState<Stock | null>(null);

  // Fetch live prices from BigQuery whenever saved stocks change
  useEffect(() => {
    if (!savedStocks.length) { setLiveData([]); return; }
    setPricesLoading(true);
    setPricesError(false);
    fetchStockSummaries(savedStocks.map(s => s.symbol))
      .then(data  => { setLiveData(data); setPricesLoading(false); })
      .catch(()   => { setPricesError(true); setPricesLoading(false); });
  }, [savedStocks]);

  const handleLogout = async () => {
    await logout();
    document.cookie = "ll_authed=; Max-Age=0; path=/";
    router.push("/");
  };

  const handleStockClick = (stock: Stock) => {
    setSelectedStock(stock);
    trackClick(stock.symbol);
  };

  const displayName = user?.displayName || user?.email?.split("@")[0] || "User";
  const joinDate    = user?.metadata?.creationTime
    ? new Date(user.metadata.creationTime).toLocaleDateString("en-US", {
        month: "long", year: "numeric",
      })
    : "—";

  // Merge Firestore saved list with live BigQuery prices
  const mergedStocks = savedStocks.map(saved => {
    const live = liveData.find(l => l.symbol === saved.symbol);
    return {
      symbol:   saved.symbol,
      name:     saved.name,
      price:    live?.price    ?? "—",
      change:   live?.change   ?? "—",
      volume:   live?.volume   ?? "—",
      positive: live?.positive ?? true,
    };
  });

  const avgChange = (() => {
    const vals = liveData
      .map(s => parseFloat(s.change.replace("%", "").replace("+", "")))
      .filter(n => !isNaN(n));
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
          <div className="rounded-xl overflow-hidden" style={card}>
            <div className="px-6 py-4 flex items-center justify-between"
              style={{ borderBottom: "1px solid hsl(215,20%,18%)" }}>
              <h2 className="text-base font-semibold" style={{ color: "hsl(210,40%,92%)" }}>
                Saved Stocks
              </h2>
              <div className="flex items-center gap-3">
                {pricesError && (
                  <span className="text-xs" style={{ color: "hsl(0,84%,60%)" }}>
                    Live prices unavailable
                  </span>
                )}
                <span className="text-xs px-2 py-1 rounded-full"
                  style={{ background: "hsla(217,91%,60%,0.15)", color: "hsl(217,91%,70%)" }}>
                  {savedStocks.length} stock{savedStocks.length !== 1 ? "s" : ""}
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

            {!authLoading && savedStocks.length > 0 && (
              <>
                <div className="grid px-6 py-2.5 text-xs font-medium"
                  style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr 36px", color: "hsl(215,15%,50%)", borderBottom: "1px solid hsl(215,20%,16%)" }}>
                  <span>Ticker</span>
                  <span className="text-right">Price</span>
                  <span className="text-right">Change</span>
                  <span className="text-right">Volume</span>
                  <span />
                </div>

                {mergedStocks.map(stock => (
                  <div
                    key={stock.symbol}
                    onClick={() => handleStockClick(stock)}
                    className="grid px-6 py-3.5 cursor-pointer items-center"
                    style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr 36px", borderTop: "1px solid hsl(215,20%,16%)" }}
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
                        <span /><span className="flex justify-end"><Loader2 className="w-3.5 h-3.5 animate-spin" style={{ color: "hsl(217,91%,60%)" }} /></span><span />
                      </>
                    ) : (
                      <>
                        <span className="text-sm font-medium text-right" style={{ color: "hsl(210,40%,92%)" }}>
                          {stock.price}
                        </span>
                        <span className="text-sm font-semibold text-right"
                          style={{ color: stock.positive ? "hsl(142,71%,45%)" : "hsl(0,84%,60%)" }}>
                          {stock.change}
                        </span>
                        <span className="text-sm text-right" style={{ color: "hsl(215,15%,55%)" }}>
                          {stock.volume}
                        </span>
                      </>
                    )}

                    <div className="flex justify-end">
                      <button
                        onClick={(e) => { e.stopPropagation(); toggleSave({ symbol: stock.symbol, name: stock.name }); }}
                        aria-label="Remove from saved"
                        className="transition-opacity hover:opacity-60"
                      >
                        <Star className="w-4 h-4" style={{ fill: "hsl(48,96%,53%)", color: "hsl(48,96%,53%)" }} />
                      </button>
                    </div>
                  </div>
                ))}
              </>
            )}
          </div>

        </div>
      </main>

      {selectedStock && (
        <StockModal stock={selectedStock} onClose={() => setSelectedStock(null)} />
      )}
    </div>
  );
}