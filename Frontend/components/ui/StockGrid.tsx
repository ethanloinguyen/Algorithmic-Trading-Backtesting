// Frontend/components/ui/StockGrid.tsx
"use client";
import { useState } from "react";
import { Star } from "lucide-react";
import { useAuth } from "@/src/app/context/AuthContext";
import StockModal, { Stock } from "@/components/ui/StockModal";

const stocks: Stock[] = [
  { symbol: "AAPL",  name: "Apple Inc.",        price: "$189.84", change: "+1.25%", volume: "52.3M",  positive: true  },
  { symbol: "MSFT",  name: "Microsoft Corp.",   price: "$425.22", change: "-0.74%", volume: "18.7M",  positive: false },
  { symbol: "GOOGL", name: "Alphabet Inc.",     price: "$155.72", change: "+1.23%", volume: "24.1M",  positive: true  },
  { symbol: "AMZN",  name: "Amazon.com Inc.",   price: "$186.13", change: "+2.49%", volume: "31.5M",  positive: true  },
  { symbol: "NVDA",  name: "NVIDIA Corp.",      price: "$878.36", change: "+1.76%", volume: "41.2M",  positive: true  },
  { symbol: "TSLA",  name: "Tesla Inc.",        price: "$177.48", change: "-3.10%", volume: "67.8M",  positive: false },
  { symbol: "META",  name: "Meta Platforms",    price: "$505.95", change: "+1.63%", volume: "14.3M",  positive: true  },
  { symbol: "JPM",   name: "JPMorgan Chase",    price: "$198.47", change: "+0.53%", volume:  "9.8M",  positive: true  },
  { symbol: "JNJ",   name: "Johnson & Johnson", price: "$243.33", change: "-0.07%", volume:  "9.1M",  positive: false },
  { symbol: "AVGO",  name: "Broadcom",          price: "$332.54", change: "+7.37%", volume: "18.3M",  positive: true  },
];

const topGainers: Stock[] = [
  { symbol: "SMCI", name: "Super Micro",     price:  "$87.42", change: "+0.53%", volume: "12.1M", positive: true },
  { symbol: "AMZN", name: "Amazon.com Inc.", price: "$186.13", change: "+2.49%", volume: "31.5M", positive: true },
  { symbol: "NVDA", name: "NVIDIA Corp.",    price: "$878.36", change: "+1.76%", volume: "41.2M", positive: true },
  { symbol: "META", name: "Meta Platforms",  price: "$505.95", change: "+1.63%", volume: "14.3M", positive: true },
  { symbol: "AAPL", name: "Apple Inc.",      price: "$189.84", change: "+1.25%", volume: "52.3M", positive: true },
];

const topLosers: Stock[] = [
  { symbol: "TSLA", name: "Tesla Inc.",      price: "$177.48", change: "-3.10%", volume: "67.8M", positive: false },
  { symbol: "INTC", name: "Intel Corp.",     price:  "$30.12", change: "-2.15%", volume: "45.2M", positive: false },
  { symbol: "BA",   name: "Boeing Co.",      price: "$172.55", change: "-1.87%", volume: "11.4M", positive: false },
  { symbol: "NFLX", name: "Netflix Inc.",    price: "$612.44", change: "-1.42%", volume:  "5.3M", positive: false },
  { symbol: "MSFT", name: "Microsoft Corp.", price: "$425.22", change: "-0.74%", volume: "18.7M", positive: false },
];

export default function StockGrid() {
  const { isSaved, toggleSave } = useAuth();
  const [selectedStock, setSelectedStock] = useState<Stock | null>(null);

  return (
    <>
      <div className="flex gap-4">

        {/* ── Watchlist Table ── */}
        <div
          className="flex-1 rounded-xl overflow-hidden"
          style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
        >
          <div className="px-6 py-4">
            <h2 className="text-base font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>
              Watchlist
            </h2>
          </div>

          {/* Header */}
          <div
            className="grid px-6 pb-2 text-xs font-medium"
            style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr 36px", color: "hsl(215, 15%, 50%)" }}
          >
            <span>Ticker</span>
            <span className="text-right">Price</span>
            <span className="text-right">Change</span>
            <span className="text-right">Volume</span>
            <span />
          </div>

          {/* Rows */}
          {stocks.map((stock) => {
            const saved = isSaved(stock.symbol);
            return (
              <div
                key={stock.symbol}
                onClick={() => setSelectedStock(stock)}
                className="grid px-6 py-3 cursor-pointer items-center"
                style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr 36px", borderTop: "1px solid hsl(215, 20%, 16%)" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "hsl(215, 25%, 14%)")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>{stock.symbol}</span>
                  <span className="text-xs" style={{ color: "hsl(215, 15%, 50%)" }}>{stock.name}</span>
                </div>
                <span className="text-sm font-medium text-right" style={{ color: "hsl(210, 40%, 92%)" }}>{stock.price}</span>
                <span className="text-sm font-medium text-right" style={{ color: stock.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)" }}>
                  {stock.change}
                </span>
                <span className="text-sm text-right" style={{ color: "hsl(215, 15%, 55%)" }}>{stock.volume}</span>
                <div className="flex justify-end">
                  <button
                    onClick={(e) => { e.stopPropagation(); toggleSave({ symbol: stock.symbol, name: stock.name }); }}
                    aria-label={saved ? "Unsave" : "Save"}
                  >
                    <Star className="w-4 h-4" style={{ fill: saved ? "hsl(48,96%,53%)" : "transparent", color: saved ? "hsl(48,96%,53%)" : "hsl(215,15%,45%)" }} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {/* ── Right column ── */}
        <div className="w-72 flex flex-col gap-4">

          {/* Top Gainers */}
          <div className="rounded-xl p-5 flex-1" style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}>
            <div className="flex items-center gap-2 mb-3">
              <span style={{ color: "hsl(142, 71%, 45%)" }}>📈</span>
              <h3 className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>Top Gainers</h3>
            </div>
            <div className="space-y-1">
              {topGainers.map((s) => {
                const saved = isSaved(s.symbol);
                return (
                  <div
                    key={s.symbol}
                    onClick={() => setSelectedStock(s)}
                    className="flex items-center gap-2 cursor-pointer rounded-lg px-2 py-1.5"
                    onMouseEnter={(e) => (e.currentTarget.style.background = "hsl(215, 25%, 14%)")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                  >
                    <span className="text-sm font-medium flex-1" style={{ color: "hsl(210, 40%, 85%)" }}>{s.symbol}</span>
                    <span className="text-sm font-semibold" style={{ color: "hsl(142, 71%, 45%)" }}>{s.change}</span>
                    <button
                      onClick={(e) => { e.stopPropagation(); toggleSave({ symbol: s.symbol, name: s.name }); }}
                      aria-label={saved ? "Unsave" : "Save"}
                      className="ml-1 flex-shrink-0"
                    >
                      <Star className="w-3.5 h-3.5" style={{ fill: saved ? "hsl(48,96%,53%)" : "transparent", color: saved ? "hsl(48,96%,53%)" : "hsl(215,15%,40%)" }} />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Top Losers */}
          <div className="rounded-xl p-5 flex-1" style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}>
            <div className="flex items-center gap-2 mb-3">
              <span style={{ color: "hsl(0, 84%, 60%)" }}>📉</span>
              <h3 className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>Top Losers</h3>
            </div>
            <div className="space-y-1">
              {topLosers.map((s) => {
                const saved = isSaved(s.symbol);
                return (
                  <div
                    key={s.symbol}
                    onClick={() => setSelectedStock(s)}
                    className="flex items-center gap-2 cursor-pointer rounded-lg px-2 py-1.5"
                    onMouseEnter={(e) => (e.currentTarget.style.background = "hsl(215, 25%, 14%)")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                  >
                    <span className="text-sm font-medium flex-1" style={{ color: "hsl(210, 40%, 85%)" }}>{s.symbol}</span>
                    <span className="text-sm font-semibold" style={{ color: "hsl(0, 84%, 60%)" }}>{s.change}</span>
                    <button
                      onClick={(e) => { e.stopPropagation(); toggleSave({ symbol: s.symbol, name: s.name }); }}
                      aria-label={saved ? "Unsave" : "Save"}
                      className="ml-1 flex-shrink-0"
                    >
                      <Star className="w-3.5 h-3.5" style={{ fill: saved ? "hsl(48,96%,53%)" : "transparent", color: saved ? "hsl(48,96%,53%)" : "hsl(215,15%,40%)" }} />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

        </div>
      </div>

      {/* Modal */}
      {selectedStock && (
        <StockModal stock={selectedStock} onClose={() => setSelectedStock(null)} />
      )}
    </>
  );
}