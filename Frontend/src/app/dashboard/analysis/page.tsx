"use client";
import { useState } from "react";
import Sidebar from "@/components/ui/Sidebar";
import { ChevronDown } from "lucide-react";

const timeRanges = ["1W", "1M", "3M", "1Y"];

const stockOptions = [
  { symbol: "AAPL", name: "Apple Inc." },
  { symbol: "MSFT", name: "Microsoft Corp." },
  { symbol: "GOOGL", name: "Alphabet Inc." },
  { symbol: "AMZN", name: "Amazon.com Inc." },
  { symbol: "NVDA", name: "NVIDIA Corp." },
  { symbol: "TSLA", name: "Tesla Inc." },
  { symbol: "META", name: "Meta Platforms" },
];

const fundamentals: Record<string, Record<string, string>> = {
  AAPL:  { sector: "Technology",        marketCap: "$2.94T", pe: "29.8",  high52: "$199.62", low52: "$164.08", avgVol: "54.2M",  dividend: "0.51%" },
  MSFT:  { sector: "Technology",        marketCap: "$3.16T", pe: "36.2",  high52: "$468.35", low52: "$362.90", avgVol: "20.1M",  dividend: "0.72%" },
  GOOGL: { sector: "Communication",     marketCap: "$1.94T", pe: "25.4",  high52: "$193.31", low52: "$130.67", avgVol: "25.3M",  dividend: "—"     },
  AMZN:  { sector: "Consumer Discret.", marketCap: "$1.97T", pe: "41.8",  high52: "$201.20", low52: "$151.61", avgVol: "35.0M",  dividend: "—"     },
  NVDA:  { sector: "Technology",        marketCap: "$2.16T", pe: "72.3",  high52: "$974.00", low52: "$505.25", avgVol: "42.8M",  dividend: "0.03%" },
  TSLA:  { sector: "Consumer Discret.", marketCap: "$566B",  pe: "43.1",  high52: "$299.29", low52: "$138.80", avgVol: "98.3M",  dividend: "—"     },
  META:  { sector: "Communication",     marketCap: "$1.29T", pe: "24.7",  high52: "$531.49", low52: "$279.40", avgVol: "15.6M",  dividend: "—"     },
};

// Generate mock multi-line chart path data
function generatePath(seed: number, points = 12, width = 740, height = 370): string {
  const step = width / (points - 1);
  const coords = Array.from({ length: points }, (_, i) => {
    const y = height * 0.1 + (height * 0.8) * (0.5 + 0.45 * Math.sin((i + seed) * 0.9) * Math.cos((i + seed * 1.3) * 0.5));
    return `${i * step},${height - y}`;
  });
  return coords.join(" ");
}

const COLORS = {
  "2020": "hsl(240, 80%, 65%)",
  "2021": "hsl(10, 80%, 60%)",
  "2022": "hsl(170, 70%, 50%)",
};

export default function AnalysisPage() {
  const [selectedStock, setSelectedStock] = useState("AAPL");
  const [selectedRange, setSelectedRange] = useState("3M");
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const stock = stockOptions.find((s) => s.symbol === selectedStock)!;
  const info = fundamentals[selectedStock];

  return (
    <div className="min-h-screen" style={{ background: "hsl(213, 27%, 7%)" }}>
      <Sidebar />

      <main className="pt-14">
        <div className="max-w-7xl mx-auto px-6 py-6">

          {/* Stock Selector + Time Range */}
          <div className="flex items-center gap-4 mb-6">
            {/* Dropdown */}
            <div className="relative">
              <button
                onClick={() => setDropdownOpen(!dropdownOpen)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all"
                style={{
                  background: "hsl(215, 25%, 14%)",
                  border: "1px solid hsl(215, 20%, 22%)",
                  color: "hsl(210, 40%, 92%)",
                }}
              >
                <span>{stock.symbol} — {stock.name.substring(0, 8)}...</span>
                <ChevronDown className="w-3.5 h-3.5" style={{ color: "hsl(215, 15%, 55%)" }} />
              </button>

              {dropdownOpen && (
                <div
                  className="absolute top-full left-0 mt-1 w-52 rounded-lg overflow-hidden z-20"
                  style={{
                    background: "hsl(215, 25%, 13%)",
                    border: "1px solid hsl(215, 20%, 22%)",
                    boxShadow: "0 8px 24px hsl(213, 27%, 4% / 0.8)",
                  }}
                >
                  {stockOptions.map((s) => (
                    <button
                      key={s.symbol}
                      onClick={() => { setSelectedStock(s.symbol); setDropdownOpen(false); }}
                      className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm transition-colors"
                      style={{
                        background: s.symbol === selectedStock ? "hsl(217, 91%, 60% / 0.15)" : "transparent",
                        color: s.symbol === selectedStock ? "hsl(217, 91%, 70%)" : "hsl(210, 40%, 80%)",
                      }}
                      onMouseEnter={e => { if (s.symbol !== selectedStock) e.currentTarget.style.background = "hsl(215, 25%, 17%)"; }}
                      onMouseLeave={e => { if (s.symbol !== selectedStock) e.currentTarget.style.background = "transparent"; }}
                    >
                      <span className="font-semibold w-12">{s.symbol}</span>
                      <span style={{ color: "hsl(215, 15%, 55%)" }}>{s.name}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Time Range Pills */}
            <div className="flex items-center gap-1">
              {timeRanges.map((r) => (
                <button
                  key={r}
                  onClick={() => setSelectedRange(r)}
                  className="px-3 py-1.5 rounded-md text-sm font-medium transition-all"
                  style={{
                    background: r === selectedRange ? "hsl(217, 91%, 60%)" : "transparent",
                    color: r === selectedRange ? "white" : "hsl(215, 15%, 55%)",
                  }}
                >
                  {r}
                </button>
              ))}
            </div>
          </div>

          {/* Chart + Fundamentals */}
          <div className="flex gap-4">
            {/* Chart */}
            <div
              className="flex-1 rounded-xl p-5"
              style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
            >
              <svg
                width="100%"
                viewBox="0 0 760 400"
                preserveAspectRatio="xMidYMid meet"
                style={{ overflow: "visible" }}
              >
                {/* Grid lines */}
                {[0, 1, 2, 3, 4, 5].map((i) => (
                  <line
                    key={i}
                    x1="40" y1={20 + i * 60} x2="740" y2={20 + i * 60}
                    stroke="hsl(215, 20%, 18%)" strokeWidth="1"
                  />
                ))}
                {/* Y-axis labels */}
                {[100, 80, 60, 40, 20, 0].map((val, i) => (
                  <text
                    key={val}
                    x="32" y={24 + i * 60}
                    textAnchor="end"
                    fontSize="11"
                    fill="hsl(215, 15%, 45%)"
                  >
                    {val}
                  </text>
                ))}
                {/* Chart lines */}
                {Object.entries(COLORS).map(([year, color], idx) => (
                  <polyline
                    key={year}
                    fill="none"
                    stroke={color}
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={generatePath(idx * 2.5 + stockOptions.findIndex(s => s.symbol === selectedStock), 13, 700, 300)
                      .split(" ")
                      .map(pt => {
                        const [x, y] = pt.split(",");
                        return `${parseFloat(x) + 40},${parseFloat(y) + 20}`;
                      })
                      .join(" ")}
                  />
                ))}
              </svg>

              {/* Legend */}
              <div className="flex items-center gap-6 mt-3 px-2">
                {Object.entries(COLORS).map(([year, color]) => (
                  <div key={year} className="flex items-center gap-2">
                    <div className="flex items-center gap-1">
                      <svg width="20" height="10">
                        <line x1="0" y1="5" x2="12" y2="5" stroke={color} strokeWidth="2" />
                        <circle cx="16" cy="5" r="3" fill="none" stroke={color} strokeWidth="1.5" />
                      </svg>
                    </div>
                    <span className="text-xs" style={{ color: "hsl(215, 15%, 55%)" }}>{year}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Fundamentals Panel */}
            <div
              className="w-64 rounded-xl p-5"
              style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
            >
              <p className="text-lg font-bold mb-0.5" style={{ color: "hsl(210, 40%, 92%)" }}>
                {selectedStock}
              </p>
              <p className="text-sm mb-5" style={{ color: "hsl(215, 15%, 50%)" }}>
                {stock.name}
              </p>

              {[
                { label: "Sector",     value: info.sector },
                { label: "Market Cap", value: info.marketCap },
                { label: "P/E Ratio",  value: info.pe },
                { label: "52W High",   value: info.high52 },
                { label: "52W Low",    value: info.low52 },
                { label: "Avg Volume", value: info.avgVol },
                { label: "Dividend",   value: info.dividend },
              ].map(({ label, value }) => (
                <div
                  key={label}
                  className="flex items-center justify-between py-3"
                  style={{ borderBottom: "1px solid hsl(215, 20%, 16%)" }}
                >
                  <span className="text-sm" style={{ color: "hsl(215, 15%, 50%)" }}>{label}</span>
                  <span className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>{value}</span>
                </div>
              ))}
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}