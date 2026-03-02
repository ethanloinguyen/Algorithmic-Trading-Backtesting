// frontend/src/components/ui/StockGrid.tsx
"use client";
import { useState } from "react";
import { Star } from "lucide-react";
import StockModal from "./StockModal";

const stocks = [
  { symbol: "AAPL", name: "Apple Inc.", price: "$189.84", change: "+1.25%", volume: "52.3M", positive: true },
  { symbol: "MSFT", name: "Microsoft Corp.", price: "$425.22", change: "-0.74%", volume: "18.7M", positive: false },
  { symbol: "GOOGL", name: "Alphabet Inc.", price: "$155.72", change: "+1.23%", volume: "24.1M", positive: true },
  { symbol: "AMZN", name: "Amazon.com Inc.", price: "$186.13", change: "+2.49%", volume: "31.5M", positive: true },
  { symbol: "NVDA", name: "NVIDIA Corp.", price: "$878.36", change: "+1.76%", volume: "41.2M", positive: true },
  { symbol: "TSLA", name: "Tesla Inc.", price: "$177.48", change: "-3.10%", volume: "67.8M", positive: false },
  { symbol: "META", name: "Meta Platforms", price: "$505.95", change: "+1.63%", volume: "14.3M", positive: true },
  { symbol: "JPM", name: "JPMorgan Chase", price: "$198.47", change: "+0.53%", volume: "9.8M", positive: true },
  { symbol: "JNJ", name: "Johnson & Johnson", price: "$243.33", change: "-0.07%", volume: "9.12M", positive: false },
  { symbol: "AVGO", name: "Broadcom", price: "$332.54", change: "+7.37%", volume: "18.3M", positive: true },
];

const topGainers = [
  { symbol: "SMCI", name: "Super Micro Computer", change: "+0.53%", price: "$812.30", volume: "8.2M", positive: true },
  { symbol: "AMZN", name: "Amazon.com Inc.", change: "+2.49%", price: "$186.13", volume: "31.5M", positive: true },
  { symbol: "NVDA", name: "NVIDIA Corp.", change: "+1.76%", price: "$878.36", volume: "41.2M", positive: true },
  { symbol: "META", name: "Meta Platforms", change: "+1.63%", price: "$505.95", volume: "14.3M", positive: true },
  { symbol: "AAPL", name: "Apple Inc.", change: "+1.25%", price: "$189.84", volume: "52.3M", positive: true },
];

const topLosers = [
  { symbol: "TSLA", name: "Tesla Inc.", change: "-3.10%", price: "$177.48", volume: "67.8M", positive: false },
  { symbol: "INTC", name: "Intel Corp.", change: "-2.15%", price: "$31.22", volume: "47.1M", positive: false },
  { symbol: "BA", name: "Boeing Co.", change: "-1.87%", price: "$172.94", volume: "12.3M", positive: false },
  { symbol: "NFLX", name: "Netflix Inc.", change: "-1.42%", price: "$628.11", volume: "5.4M", positive: false },
  { symbol: "MSFT", name: "Microsoft Corp.", change: "-0.74%", price: "$425.22", volume: "18.7M", positive: false },
];

interface StockInfo {
  symbol: string;
  name: string;
  price: string;
  change: string;
  positive: boolean;
}

interface StockGridProps {
  savedSymbols?: Set<string>;
  onSaveToggle?: (stock: StockInfo) => void;
}

export default function StockGrid({ savedSymbols, onSaveToggle }: StockGridProps) {
  const [modalStock, setModalStock] = useState<StockInfo | null>(null);
  const [localSaved, setLocalSaved] = useState<Set<string>>(new Set());

  const saved = savedSymbols ?? localSaved;

  const toggleSave = (e: React.MouseEvent, stock: StockInfo) => {
    e.stopPropagation();
    if (onSaveToggle) {
      onSaveToggle(stock);
    } else {
      setLocalSaved(prev => {
        const next = new Set(prev);
        next.has(stock.symbol) ? next.delete(stock.symbol) : next.add(stock.symbol);
        return next;
      });
    }
  };

  return (
    <>
      <div className="flex gap-4">
        {/* Watchlist Table */}
        <div
          className="flex-1 rounded-xl overflow-hidden"
          style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
        >
          <div className="px-6 py-4">
            <h2 className="text-base font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>Featured Stocks</h2>
          </div>

          {/* Table Header */}
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

          {/* Rows */}
          <div>
            {stocks.map((stock) => (
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
                  onClick={e => toggleSave(e, stock)}
                  className="flex items-center justify-center transition-all hover:scale-110"
                  aria-label={saved.has(stock.symbol) ? "Unsave" : "Save"}
                >
                  <Star className="w-3.5 h-3.5" style={{ fill: saved.has(stock.symbol) ? "#facc15" : "none", color: saved.has(stock.symbol) ? "#facc15" : "hsl(215, 15%, 40%)" }} />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Right column: Top Gainers + Top Losers */}
        <div className="w-72 flex flex-col gap-4">
          {/* Top Gainers */}
          <div className="rounded-xl p-5 flex-1" style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}>
            <div className="flex items-center gap-2 mb-4">
              <span className="text-sm" style={{ color: "hsl(142, 71%, 45%)" }}>📈</span>
              <h3 className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>Top Gainers</h3>
            </div>
            <div className="space-y-3">
              {topGainers.map((s) => (
                <div
                  key={s.symbol}
                  className="flex items-center justify-between cursor-pointer rounded-lg px-2 py-1 -mx-2 transition-colors"
                  onClick={() => setModalStock(s)}
                  onMouseEnter={e => (e.currentTarget.style.background = "hsl(215, 25%, 15%)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium" style={{ color: "hsl(210, 40%, 85%)" }}>{s.symbol}</span>
                    <button
                      onClick={e => toggleSave(e, s)}
                      className="transition-all hover:scale-110"
                      aria-label={saved.has(s.symbol) ? "Unsave" : "Save"}
                    >
                      <Star className="w-3 h-3" style={{ fill: saved.has(s.symbol) ? "#facc15" : "none", color: saved.has(s.symbol) ? "#facc15" : "hsl(215, 15%, 40%)" }} />
                    </button>
                  </div>
                  <span className="text-sm font-semibold" style={{ color: "hsl(142, 71%, 45%)" }}>{s.change}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Top Losers */}
          <div className="rounded-xl p-5 flex-1" style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}>
            <div className="flex items-center gap-2 mb-4">
              <span className="text-sm" style={{ color: "hsl(0, 84%, 60%)" }}>📉</span>
              <h3 className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>Top Losers</h3>
            </div>
            <div className="space-y-3">
              {topLosers.map((s) => (
                <div
                  key={s.symbol}
                  className="flex items-center justify-between cursor-pointer rounded-lg px-2 py-1 -mx-2 transition-colors"
                  onClick={() => setModalStock(s)}
                  onMouseEnter={e => (e.currentTarget.style.background = "hsl(215, 25%, 15%)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium" style={{ color: "hsl(210, 40%, 85%)" }}>{s.symbol}</span>
                    <button
                      onClick={e => toggleSave(e, s)}
                      className="transition-all hover:scale-110"
                      aria-label={saved.has(s.symbol) ? "Unsave" : "Save"}
                    >
                      <Star className="w-3 h-3" style={{ fill: saved.has(s.symbol) ? "#facc15" : "none", color: saved.has(s.symbol) ? "#facc15" : "hsl(215, 15%, 40%)" }} />
                    </button>
                  </div>
                  <span className="text-sm font-semibold" style={{ color: "hsl(0, 84%, 60%)" }}>{s.change}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <StockModal stock={modalStock} onClose={() => setModalStock(null)} />
    </>
  );
}