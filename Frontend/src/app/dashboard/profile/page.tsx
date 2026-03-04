// frontend/src/app/dashboard/profile/page.tsx
"use client";
import { useEffect, useState } from "react";
import Sidebar from "@/components/ui/Sidebar";
import StockModal from "@/components/ui/StockModal";
import { Star, User } from "lucide-react";
import { StarredProvider, useStarred } from "@/components/ui/StarredContext";

interface StockInfo {
  symbol: string;
  name: string;
  price: string;
  change: string;
  volume: string;
  positive: boolean;
}

function ProfileContent() {
  const { starred, savedSymbols, toggleStar, refreshStarred } = useStarred();
  const [modalStock, setModalStock] = useState<StockInfo | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  // Refresh prices from API every time the profile page is opened
  useEffect(() => {
    setRefreshing(true);
    refreshStarred().finally(() => setRefreshing(false));
  }, []);

  return (
    <div className="min-h-screen" style={{ background: "hsl(213, 27%, 7%)" }}>
      <Sidebar />
      <main className="pt-14">
        <div className="max-w-7xl mx-auto px-6 py-6">

          {/* Profile header */}
          <div className="flex items-center gap-4 mb-8">
            <div
              className="w-12 h-12 rounded-full flex items-center justify-center"
              style={{ background: "hsl(217, 91%, 60%)" }}
            >
              <User className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold" style={{ color: "hsl(210, 40%, 92%)" }}>User Name</h1>
              <p className="text-sm" style={{ color: "hsl(215, 15%, 50%)" }}>user@example.com</p>
            </div>
          </div>

          {/* Starred Stocks table */}
          <div
            className="rounded-xl overflow-hidden"
            style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
          >
            {/* Table header bar */}
            <div
              className="flex items-center justify-between px-6 py-4"
              style={{ borderBottom: "1px solid hsl(215, 20%, 16%)" }}
            >
              <div className="flex items-center gap-2">
                <Star className="w-4 h-4" style={{ fill: "#facc15", color: "#facc15" }} />
                <h2 className="text-base font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>Saved Stocks</h2>
                {refreshing && (
                  <div
                    className="w-3 h-3 rounded-full border border-t-transparent animate-spin ml-1"
                    style={{ borderColor: "hsl(217, 91%, 60%)", borderTopColor: "transparent" }}
                  />
                )}
              </div>
              <span
                className="text-xs px-2 py-0.5 rounded-full"
                style={{ background: "hsl(215, 25%, 16%)", color: "hsl(215, 15%, 55%)" }}
              >
                {starred.length} {starred.length === 1 ? "stock" : "stocks"}
              </span>
            </div>

            {starred.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 gap-3">
                <Star className="w-10 h-10" style={{ color: "hsl(215, 15%, 30%)" }} />
                <p className="text-sm font-medium" style={{ color: "hsl(215, 15%, 50%)" }}>No starred stocks yet</p>
                <p className="text-xs" style={{ color: "hsl(215, 15%, 38%)" }}>
                  Click the ★ next to any stock on the dashboard to save it here.
                </p>
              </div>
            ) : (
              <>
                {/* Column headers */}
                <div
                  className="grid px-6 py-2 text-xs font-medium"
                  style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr 32px", color: "hsl(215, 15%, 50%)" }}
                >
                  <span>Ticker</span>
                  <span className="text-right">Price</span>
                  <span className="text-right">Change</span>
                  <span className="text-right">Volume</span>
                  <span />
                </div>

                {/* Rows */}
                {starred.map(stock => (
                  <div
                    key={stock.symbol}
                    className="grid px-6 py-3 cursor-pointer transition-colors items-center"
                    style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr 32px", borderTop: "1px solid hsl(215, 20%, 16%)" }}
                    onClick={() => setModalStock(stock)}
                    onMouseEnter={e => (e.currentTarget.style.background = "hsl(215, 25%, 14%)")}
                    onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>{stock.symbol}</span>
                      <span className="text-xs" style={{ color: "hsl(215, 15%, 50%)" }}>{stock.name}</span>
                    </div>
                    <span className="text-sm font-medium text-right" style={{ color: "hsl(210, 40%, 92%)" }}>{stock.price}</span>
                    <span
                      className="text-sm font-medium text-right"
                      style={{ color: stock.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)" }}
                    >
                      {stock.change}
                    </span>
                    <span className="text-sm text-right" style={{ color: "hsl(215, 15%, 55%)" }}>{stock.volume}</span>
                    <button
                      onClick={e => { e.stopPropagation(); toggleStar(stock); }}
                      className="flex items-center justify-center transition-all hover:scale-110"
                      aria-label="Unstar"
                    >
                      <Star className="w-3.5 h-3.5" style={{ fill: "#facc15", color: "#facc15" }} />
                    </button>
                  </div>
                ))}
              </>
            )}
          </div>
        </div>
      </main>

      <StockModal stock={modalStock} onClose={() => setModalStock(null)} />
    </div>
  );
}

// Wrap in its own StarredProvider so the profile page works
// independently when navigated to directly
export default function ProfilePage() {
  return (
    <StarredProvider>
      <ProfileContent />
    </StarredProvider>
  );
}