// frontend/src/components/ui/StockGrid.tsx
"use client";
import { useState, useEffect } from "react";
import { Star } from "lucide-react";
import StockModal from "./StockModal";
import { useStarred } from "@/components/ui/StarredContext";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

interface StockInfo {
  symbol: string;
  name: string;
  price: string;
  change: string;
  volume: string;
  positive: boolean;
}

function SkeletonRow() {
  return (
    <div
      className="grid px-6 py-3 items-center animate-pulse"
      style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr 32px", borderTop: "1px solid hsl(215, 20%, 16%)" }}
    >
      <div className="h-3 w-24 rounded" style={{ background: "hsl(215, 20%, 20%)" }} />
      <div className="h-3 w-16 rounded ml-auto" style={{ background: "hsl(215, 20%, 20%)" }} />
      <div className="h-3 w-14 rounded ml-auto" style={{ background: "hsl(215, 20%, 20%)" }} />
      <div className="h-3 w-12 rounded ml-auto" style={{ background: "hsl(215, 20%, 20%)" }} />
      <div />
    </div>
  );
}

function SkeletonGainerRow() {
  return (
    <div className="flex items-center justify-between px-2 py-1 animate-pulse">
      <div className="h-3 w-14 rounded" style={{ background: "hsl(215, 20%, 20%)" }} />
      <div className="h-3 w-12 rounded" style={{ background: "hsl(215, 20%, 20%)" }} />
    </div>
  );
}

export default function StockGrid() {
  const [modalStock, setModalStock] = useState<StockInfo | null>(null);
  const [stocks, setStocks] = useState<StockInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { savedSymbols, toggleStar } = useStarred();

  useEffect(() => {
    async function fetchStocks() {
      try {
        setLoading(true);
        setError(null);
        const res = await fetch(`${API}/api/stocks`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        setStocks(json.data);
      } catch (err) {
        console.error("Failed to fetch stocks:", err);
        setError("Unable to load stock data.");
      } finally {
        setLoading(false);
      }
    }
    fetchStocks();
  }, []);

  const handleStarClick = (e: React.MouseEvent, stock: StockInfo) => {
    e.stopPropagation();
    toggleStar(stock);
  };

  const sorted = [...stocks].sort((a, b) => {
    const aVal = parseFloat(a.change.replace("%", "").replace("+", ""));
    const bVal = parseFloat(b.change.replace("%", "").replace("+", ""));
    return bVal - aVal;
  });
  const topGainers = sorted.filter(s => s.positive).slice(0, 5);
  const topLosers  = sorted.filter(s => !s.positive).slice(-5).reverse();

  return (
    <>
      <div className="flex gap-4">
        {/* Featured Stocks Table */}
        <div
          className="flex-1 rounded-xl overflow-hidden"
          style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
        >
          <div className="px-6 py-4">
            <h2 className="text-base font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>Featured Stocks</h2>
          </div>
          <div
            className="grid px-6 pb-2 text-xs font-medium"
            style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr 32px", color: "hsl(215, 15%, 50%)" }}
          >
            <span>Ticker</span>
            <span className="text-right">Price</span>
            <span className="text-right">Change</span>
            <span className="text-right">Volume</span>
            <span />
          </div>
          <div>
            {loading
              ? Array.from({ length: 8 }).map((_, i) => <SkeletonRow key={i} />)
              : error
              ? <div className="px-6 py-8 text-center text-sm" style={{ color: "hsl(0, 84%, 60%)" }}>{error}</div>
              : stocks.map((stock) => (
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
                  <span className="text-sm font-medium text-right" style={{ color: stock.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)" }}>{stock.change}</span>
                  <span className="text-sm text-right" style={{ color: "hsl(215, 15%, 55%)" }}>{stock.volume}</span>
                  <button
                    onClick={e => handleStarClick(e, stock)}
                    className="flex items-center justify-center transition-all hover:scale-110"
                    aria-label={savedSymbols.has(stock.symbol) ? "Unsave" : "Save"}
                  >
                    <Star className="w-3.5 h-3.5" style={{ fill: savedSymbols.has(stock.symbol) ? "#facc15" : "none", color: savedSymbols.has(stock.symbol) ? "#facc15" : "hsl(215, 15%, 40%)" }} />
                  </button>
                </div>
              ))
            }
          </div>
        </div>

        {/* Right column */}
        <div className="w-72 flex flex-col gap-4">
          {/* Top Gainers */}
          <div className="rounded-xl p-5 flex-1" style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}>
            <div className="flex items-center gap-2 mb-4">
              <span className="text-sm" style={{ color: "hsl(142, 71%, 45%)" }}>📈</span>
              <h3 className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>Top Gainers</h3>
            </div>
            <div className="space-y-3">
              {loading
                ? Array.from({ length: 5 }).map((_, i) => <SkeletonGainerRow key={i} />)
                : topGainers.map((s) => (
                  <div
                    key={s.symbol}
                    className="flex items-center justify-between cursor-pointer rounded-lg px-2 py-1 -mx-2 transition-colors"
                    onClick={() => setModalStock(s)}
                    onMouseEnter={e => (e.currentTarget.style.background = "hsl(215, 25%, 15%)")}
                    onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium" style={{ color: "hsl(210, 40%, 85%)" }}>{s.symbol}</span>
                      <button onClick={e => handleStarClick(e, s)} className="transition-all hover:scale-110">
                        <Star className="w-3 h-3" style={{ fill: savedSymbols.has(s.symbol) ? "#facc15" : "none", color: savedSymbols.has(s.symbol) ? "#facc15" : "hsl(215, 15%, 40%)" }} />
                      </button>
                    </div>
                    <span className="text-sm font-semibold" style={{ color: "hsl(142, 71%, 45%)" }}>{s.change}</span>
                  </div>
                ))
              }
            </div>
          </div>

          {/* Top Losers */}
          <div className="rounded-xl p-5 flex-1" style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}>
            <div className="flex items-center gap-2 mb-4">
              <span className="text-sm" style={{ color: "hsl(0, 84%, 60%)" }}>📉</span>
              <h3 className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>Top Losers</h3>
            </div>
            <div className="space-y-3">
              {loading
                ? Array.from({ length: 5 }).map((_, i) => <SkeletonGainerRow key={i} />)
                : topLosers.map((s) => (
                  <div
                    key={s.symbol}
                    className="flex items-center justify-between cursor-pointer rounded-lg px-2 py-1 -mx-2 transition-colors"
                    onClick={() => setModalStock(s)}
                    onMouseEnter={e => (e.currentTarget.style.background = "hsl(215, 25%, 15%)")}
                    onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium" style={{ color: "hsl(210, 40%, 85%)" }}>{s.symbol}</span>
                      <button onClick={e => handleStarClick(e, s)} className="transition-all hover:scale-110">
                        <Star className="w-3 h-3" style={{ fill: savedSymbols.has(s.symbol) ? "#facc15" : "none", color: savedSymbols.has(s.symbol) ? "#facc15" : "hsl(215, 15%, 40%)" }} />
                      </button>
                    </div>
                    <span className="text-sm font-semibold" style={{ color: "hsl(0, 84%, 60%)" }}>{s.change}</span>
                  </div>
                ))
              }
            </div>
          </div>
        </div>
      </div>

      <StockModal stock={modalStock} onClose={() => setModalStock(null)} />
    </>
  );
}