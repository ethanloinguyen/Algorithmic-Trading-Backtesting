"use client";
import { Star } from "lucide-react";

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
  { symbol: "SMCI", change: "+0.53%", positive: true },
  { symbol: "AMZN", change: "+2.49%", positive: true },
  { symbol: "NVDA", change: "+1.76%", positive: true },
  { symbol: "META", change: "+1.63%", positive: true },
  { symbol: "AAPL", change: "+1.25%", positive: true },
];

const topLosers = [
  { symbol: "TSLA", change: "-3.10%", positive: false },
  { symbol: "INTC", change: "-2.15%", positive: false },
  { symbol: "BA", change: "-1.87%", positive: false },
  { symbol: "NFLX", change: "-1.42%", positive: false },
  { symbol: "MSFT", change: "-0.74%", positive: false },
];

export default function StockGrid() {
  return (
    <div className="flex gap-4">
      {/* Watchlist Table */}
      <div
        className="flex-1 rounded-xl overflow-hidden"
        style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
      >
        <div className="px-6 py-4">
          <h2 className="text-base font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>
            Watchlist
          </h2>
        </div>

        {/* Table Header */}
        <div
          className="grid px-6 pb-2 text-xs font-medium"
          style={{
            gridTemplateColumns: "1fr 1fr 1fr 1fr",
            color: "hsl(215, 15%, 50%)",
          }}
        >
          <span>Ticker</span>
          <span className="text-right">Price</span>
          <span className="text-right">Change</span>
          <span className="text-right">Volume</span>
        </div>

        {/* Rows */}
        <div>
          {stocks.map((stock) => (
            <div
              key={stock.symbol}
              className="grid px-6 py-3 cursor-pointer transition-colors"
              style={{
                gridTemplateColumns: "1fr 1fr 1fr 1fr",
                borderTop: "1px solid hsl(215, 20%, 16%)",
              }}
              onMouseEnter={e => (e.currentTarget.style.background = "hsl(215, 25%, 14%)")}
              onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
            >
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>
                  {stock.symbol}
                </span>
                <span className="text-xs" style={{ color: "hsl(215, 15%, 50%)" }}>
                  {stock.name}
                </span>
              </div>
              <span className="text-sm font-medium text-right self-center" style={{ color: "hsl(210, 40%, 92%)" }}>
                {stock.price}
              </span>
              <span
                className="text-sm font-medium text-right self-center"
                style={{ color: stock.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)" }}
              >
                {stock.change}
              </span>
              <span className="text-sm text-right self-center" style={{ color: "hsl(215, 15%, 55%)" }}>
                {stock.volume}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Right column: Top Gainers + Top Losers */}
      <div className="w-72 flex flex-col gap-4">
        {/* Top Gainers */}
        <div
          className="rounded-xl p-5 flex-1"
          style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
        >
          <div className="flex items-center gap-2 mb-4">
            <span className="text-sm" style={{ color: "hsl(142, 71%, 45%)" }}>📈</span>
            <h3 className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>
              Top Gainers
            </h3>
          </div>
          <div className="space-y-3">
            {topGainers.map((s) => (
              <div key={s.symbol} className="flex items-center justify-between">
                <span className="text-sm font-medium" style={{ color: "hsl(210, 40%, 85%)" }}>
                  {s.symbol}
                </span>
                <span className="text-sm font-semibold" style={{ color: "hsl(142, 71%, 45%)" }}>
                  {s.change}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Top Losers */}
        <div
          className="rounded-xl p-5 flex-1"
          style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
        >
          <div className="flex items-center gap-2 mb-4">
            <span className="text-sm" style={{ color: "hsl(0, 84%, 60%)" }}>📉</span>
            <h3 className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>
              Top Losers
            </h3>
          </div>
          <div className="space-y-3">
            {topLosers.map((s) => (
              <div key={s.symbol} className="flex items-center justify-between">
                <span className="text-sm font-medium" style={{ color: "hsl(210, 40%, 85%)" }}>
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
  );
}