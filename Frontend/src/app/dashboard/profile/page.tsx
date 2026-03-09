// Frontend/src/app/dashboard/profile/page.tsx
"use client";
import { useState } from "react";
import Sidebar from "@/components/ui/Sidebar";
import StockModal from "@/components/ui/StockModal";
import { useAuth } from "@/src/app/context/AuthContext";
import { Star, User, Mail, LogOut, Loader2 } from "lucide-react";
import { useRouter } from "next/navigation";

// Mock price data for saved stocks — in production, fetch from /api/stocks/summaries
const mockPrices: Record<string, { price: string; change: string; volume: string; positive: boolean }> = {
  AAPL:  { price: "$189.84", change: "+1.25%", volume: "52.3M",  positive: true  },
  MSFT:  { price: "$425.22", change: "-0.74%", volume: "18.7M",  positive: false },
  GOOGL: { price: "$155.72", change: "+1.23%", volume: "24.1M",  positive: true  },
  AMZN:  { price: "$186.13", change: "+2.49%", volume: "31.5M",  positive: true  },
  NVDA:  { price: "$878.36", change: "+1.76%", volume: "41.2M",  positive: true  },
  TSLA:  { price: "$177.48", change: "-3.10%", volume: "67.8M",  positive: false },
  META:  { price: "$505.95", change: "+1.63%", volume: "14.3M",  positive: true  },
  JPM:   { price: "$198.47", change: "+0.53%", volume:  "9.8M",  positive: true  },
  JNJ:   { price: "$243.33", change: "-0.07%", volume:  "9.1M",  positive: false },
  AVGO:  { price: "$332.54", change: "+7.37%", volume: "18.3M",  positive: true  },
  SMCI:  { price:  "$87.42", change: "+0.53%", volume: "12.1M",  positive: true  },
  INTC:  { price:  "$30.12", change: "-2.15%", volume: "45.2M",  positive: false },
  BA:    { price: "$172.55", change: "-1.87%", volume: "11.4M",  positive: false },
  NFLX:  { price: "$612.44", change: "-1.42%", volume:  "5.3M",  positive: false },
};

const cardStyle = {
  background: "hsl(215, 25%, 11%)",
  border:     "1px solid hsl(215, 20%, 18%)",
};

export default function ProfilePage() {
  const { user, savedStocks, toggleSave, loading, logout } = useAuth();
  const router = useRouter();
  const [selectedStock, setSelectedStock] = useState<null | {
    symbol: string; name: string; price: string;
    change: string; volume: string; positive: boolean;
  }>(null);

  const handleLogout = async () => {
    await logout();
    document.cookie = "ll_authed=; Max-Age=0; path=/";
    router.push("/");
  };

  const displayName = user?.displayName || user?.email?.split("@")[0] || "User";
  const joinDate    = user?.metadata?.creationTime
    ? new Date(user.metadata.creationTime).toLocaleDateString("en-US", { month: "long", year: "numeric" })
    : "—";

  return (
    <div className="min-h-screen" style={{ background: "hsl(213, 27%, 7%)" }}>
      <Sidebar />

      <main className="pt-14">
        <div className="max-w-4xl mx-auto px-6 py-8">

          {/* Profile Card */}
          <div className="rounded-2xl p-6 mb-6 flex items-center justify-between" style={cardStyle}>
            <div className="flex items-center gap-5">
              {/* Avatar */}
              <div
                className="w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold"
                style={{ background: "hsl(217, 91%, 60%)", color: "white" }}
              >
                {displayName[0].toUpperCase()}
              </div>

              {/* Info */}
              <div>
                <h1 className="text-xl font-bold mb-1" style={{ color: "hsl(210, 40%, 92%)" }}>
                  {displayName}
                </h1>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-1.5 text-sm" style={{ color: "hsl(215, 15%, 55%)" }}>
                    <Mail className="w-3.5 h-3.5" />
                    <span>{user?.email}</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-sm" style={{ color: "hsl(215, 15%, 55%)" }}>
                    <User className="w-3.5 h-3.5" />
                    <span>Joined {joinDate}</span>
                  </div>
                </div>
              </div>
            </div>

            <button
              onClick={handleLogout}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all hover:opacity-80"
              style={{
                background: "hsl(215, 25%, 16%)",
                border:     "1px solid hsl(215, 20%, 22%)",
                color:      "hsl(215, 15%, 65%)",
              }}
            >
              <LogOut className="w-4 h-4" />
              Sign Out
            </button>
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            {[
              { label: "Saved Stocks",  value: savedStocks.length.toString(), color: "hsl(217, 91%, 60%)" },
              {
                label: "Avg. Change",
                value: savedStocks.length === 0 ? "—" : (() => {
                  const vals = savedStocks
                    .map(s => mockPrices[s.symbol]?.change)
                    .filter(Boolean)
                    .map(c => parseFloat(c!.replace("%", "")));
                  if (!vals.length) return "—";
                  const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
                  return `${avg >= 0 ? "+" : ""}${avg.toFixed(2)}%`;
                })(),
                color: "hsl(142, 71%, 45%)",
              },
              { label: "Watchlist",     value: savedStocks.length > 0 ? "Active" : "Empty", color: "hsl(280, 70%, 65%)" },
            ].map((card) => (
              <div key={card.label} className="rounded-xl p-5" style={cardStyle}>
                <p className="text-xs mb-2" style={{ color: "hsl(215, 15%, 55%)" }}>{card.label}</p>
                <p className="text-2xl font-bold" style={{ color: card.color }}>{card.value}</p>
              </div>
            ))}
          </div>

          {/* Saved Stocks */}
          <div className="rounded-xl overflow-hidden" style={cardStyle}>
            <div
              className="px-6 py-4 flex items-center justify-between"
              style={{ borderBottom: "1px solid hsl(215, 20%, 18%)" }}
            >
              <h2 className="text-base font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>
                Saved Stocks
              </h2>
              <span className="text-xs px-2 py-1 rounded-full" style={{ background: "hsl(217, 91%, 60% / 0.15)", color: "hsl(217, 91%, 70%)" }}>
                {savedStocks.length} stock{savedStocks.length !== 1 ? "s" : ""}
              </span>
            </div>

            {loading ? (
              <div className="flex items-center justify-center py-16">
                <Loader2 className="w-5 h-5 animate-spin" style={{ color: "hsl(217, 91%, 60%)" }} />
              </div>

            ) : savedStocks.length === 0 ? (
              <div className="py-16 text-center">
                <Star className="w-10 h-10 mx-auto mb-3" style={{ color: "hsl(215, 15%, 28%)" }} />
                <p className="text-sm font-medium mb-1" style={{ color: "hsl(215, 15%, 55%)" }}>
                  No saved stocks yet
                </p>
                <p className="text-xs" style={{ color: "hsl(215, 15%, 40%)" }}>
                  Star stocks from the dashboard to track them here.
                </p>
              </div>

            ) : (
              <>
                {/* Table header */}
                <div
                  className="grid px-6 py-2.5 text-xs font-medium"
                  style={{
                    gridTemplateColumns: "1fr 1fr 1fr 1fr 36px",
                    color: "hsl(215, 15%, 50%)",
                    borderBottom: "1px solid hsl(215, 20%, 16%)",
                  }}
                >
                  <span>Ticker</span>
                  <span className="text-right">Price</span>
                  <span className="text-right">Change</span>
                  <span className="text-right">Volume</span>
                  <span />
                </div>

                {savedStocks.map((stock) => {
                  const data = mockPrices[stock.symbol] ?? {
                    price: "—", change: "—", volume: "—", positive: true,
                  };
                  return (
                    <div
                      key={stock.symbol}
                      onClick={() => setSelectedStock({ ...stock, ...data })}
                      className="grid px-6 py-3.5 cursor-pointer transition-colors items-center"
                      style={{
                        gridTemplateColumns: "1fr 1fr 1fr 1fr 36px",
                        borderTop: "1px solid hsl(215, 20%, 16%)",
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.background = "hsl(215, 25%, 14%)")}
                      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>
                          {stock.symbol}
                        </span>
                        <span className="text-xs" style={{ color: "hsl(215, 15%, 50%)" }}>
                          {stock.name}
                        </span>
                      </div>
                      <span className="text-sm font-medium text-right" style={{ color: "hsl(210, 40%, 92%)" }}>
                        {data.price}
                      </span>
                      <span
                        className="text-sm font-semibold text-right"
                        style={{ color: data.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)" }}
                      >
                        {data.change}
                      </span>
                      <span className="text-sm text-right" style={{ color: "hsl(215, 15%, 55%)" }}>
                        {data.volume}
                      </span>
                      <div className="flex justify-end">
                        <button
                          onClick={(e) => { e.stopPropagation(); toggleSave(stock); }}
                          className="transition-opacity hover:opacity-60"
                          aria-label="Remove from saved"
                        >
                          <Star
                            className="w-4 h-4"
                            style={{ fill: "hsl(48, 96%, 53%)", color: "hsl(48, 96%, 53%)" }}
                          />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </>
            )}
          </div>

        </div>
      </main>

      {/* Stock Modal */}
      {selectedStock && (
        <StockModal
          stock={selectedStock}
          onClose={() => setSelectedStock(null)}
        />
      )}
    </div>
  );
}