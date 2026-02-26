"use client";
import { useState } from "react";
import Sidebar from "@/components/ui/Sidebar";
import { Star } from "lucide-react";

const mockSavedStocks = [
  { name: "Apple Inc.", symbol: "AAPL", price: "$185.92", change: "+2.34%", volume: "52.3M", positive: true },
  { name: "Microsoft Corp.", symbol: "MSFT", price: "$412.78", change: "+1.56%", volume: "18.7M", positive: true },
  { name: "NVIDIA Corp.", symbol: "NVDA", price: "$875.28", change: "+3.21%", volume: "41.2M", positive: true },
];

export default function SavedStocksPage() {
  const [savedStocks, setSavedStocks] = useState(mockSavedStocks);

  const handleRemoveStock = (symbol: string) => {
    setSavedStocks((prev) => prev.filter((s) => s.symbol !== symbol));
  };

  return (
    <div className="min-h-screen" style={{ background: "hsl(213, 27%, 7%)" }}>
      <Sidebar />

      <main className="pt-14">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <h1 className="text-xl font-semibold mb-6" style={{ color: "hsl(210, 40%, 92%)" }}>
            My Saved Stocks
          </h1>

          {savedStocks.length === 0 ? (
            <div
              className="rounded-xl p-16 text-center"
              style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
            >
              <Star className="w-12 h-12 mx-auto mb-3" style={{ color: "hsl(215, 15%, 30%)" }} />
              <p className="text-base font-medium mb-1" style={{ color: "hsl(215, 15%, 60%)" }}>
                No saved stocks
              </p>
              <p className="text-sm" style={{ color: "hsl(215, 15%, 45%)" }}>
                Star stocks from the dashboard to track them here.
              </p>
            </div>
          ) : (
            <div
              className="rounded-xl overflow-hidden"
              style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
            >
              {/* Table Header */}
              <div
                className="grid px-6 py-3 text-xs font-medium"
                style={{
                  gridTemplateColumns: "1fr 1fr 1fr 1fr 40px",
                  color: "hsl(215, 15%, 50%)",
                  borderBottom: "1px solid hsl(215, 20%, 18%)",
                }}
              >
                <span>Ticker</span>
                <span className="text-right">Price</span>
                <span className="text-right">Change</span>
                <span className="text-right">Volume</span>
                <span />
              </div>

              {savedStocks.map((stock) => (
                <div
                  key={stock.symbol}
                  className="grid px-6 py-3.5 transition-colors cursor-pointer"
                  style={{
                    gridTemplateColumns: "1fr 1fr 1fr 1fr 40px",
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
                  <span className="text-sm font-medium text-right self-center" style={{ color: "hsl(210, 40%, 92%)" }}>
                    {stock.price}
                  </span>
                  <span
                    className="text-sm font-semibold text-right self-center"
                    style={{ color: stock.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)" }}
                  >
                    {stock.change}
                  </span>
                  <span className="text-sm text-right self-center" style={{ color: "hsl(215, 15%, 55%)" }}>
                    {stock.volume}
                  </span>
                  <div className="flex items-center justify-end">
                    <button
                      onClick={() => handleRemoveStock(stock.symbol)}
                      className="transition-opacity hover:opacity-60"
                      aria-label="Remove"
                    >
                      <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Summary Cards */}
          {savedStocks.length > 0 && (
            <div className="grid grid-cols-3 gap-4 mt-6">
              {[
                { label: "Total Stocks", value: savedStocks.length.toString(), color: "hsl(217, 91%, 60%)" },
                { label: "Avg. Change", value: "+2.37%", color: "hsl(142, 71%, 45%)" },
                { label: "Portfolio Value", value: "$1,473.98", color: "hsl(280, 70%, 65%)" },
              ].map((card) => (
                <div
                  key={card.label}
                  className="rounded-xl p-5"
                  style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
                >
                  <p className="text-xs mb-2" style={{ color: "hsl(215, 15%, 55%)" }}>
                    {card.label}
                  </p>
                  <p className="text-2xl font-bold" style={{ color: card.color }}>
                    {card.value}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}