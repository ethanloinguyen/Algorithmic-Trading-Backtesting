// frontend/src/app/dashboard/home.tsx
"use client";
import { useState, useEffect } from "react";
import Sidebar from "@/components/ui/Sidebar";
import SectorFilter from "@/components/ui/SectorFilter";
import StockGrid from "@/components/ui/StockGrid";
import StockModal, { type Stock } from "@/components/ui/StockModal";
import { Search, TrendingUp, TrendingDown, Menu, Loader2 } from "lucide-react";
import { fetchIndices, type IndexSummary } from "@/src/app/lib/api";

// ── Index display names (shorter than backend's full names) ──────────────────
const INDEX_DISPLAY: Record<string, { short: string; full: string }> = {
  SPX:  { short: "S&P 500",  full: "S&P 500 Index" },
  IXIC: { short: "NASDAQ",   full: "NASDAQ Composite" },
  DJI:  { short: "Dow Jones", full: "Dow Jones Industrial Average" },
};

// ── Index Card ────────────────────────────────────────────────────────────────

function IndexCard({
  index,
  onClick,
}: {
  index: IndexSummary;
  onClick: () => void;
}) {
  const display = INDEX_DISPLAY[index.symbol] ?? { short: index.symbol, full: index.name };
  const color   = index.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)";

  return (
    <button
      onClick={onClick}
      className="rounded-xl p-5 text-left w-full transition-all"
      style={{
        background: "hsl(215, 25%, 11%)",
        border:     "1px solid hsl(215, 20%, 18%)",
        cursor:     "pointer",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = "hsl(215, 25%, 14%)";
        e.currentTarget.style.border     = "1px solid hsl(215, 20%, 26%)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = "hsl(215, 25%, 11%)";
        e.currentTarget.style.border     = "1px solid hsl(215, 20%, 18%)";
      }}
    >
      {/* Top row: symbol badge + INDEX label */}
      <div className="flex items-center justify-between mb-3">
        <span
          className="text-xs font-bold px-2 py-0.5 rounded-md"
          style={{ background: "hsl(215, 25%, 18%)", color: "hsl(215, 15%, 60%)" }}
        >
          {index.symbol}
        </span>
        <span
          className="text-[10px] font-semibold tracking-widest uppercase"
          style={{ color: "hsl(215, 15%, 40%)" }}
        >
          Index
        </span>
      </div>

      {/* Index short name */}
      <p className="text-sm font-semibold mb-1" style={{ color: "hsl(210, 40%, 85%)" }}>
        {display.short}
      </p>

      {/* Value */}
      <p
        className="text-2xl font-bold mb-2"
        style={{ color: "hsl(210, 40%, 96%)", fontVariantNumeric: "tabular-nums" }}
      >
        {index.value}
      </p>

      {/* Change */}
      <div className="flex items-center gap-1.5">
        {index.positive
          ? <TrendingUp  className="w-3.5 h-3.5" style={{ color }} />
          : <TrendingDown className="w-3.5 h-3.5" style={{ color }} />}
        <span className="text-sm font-semibold" style={{ color }}>
          {index.change}
        </span>
      </div>
    </button>
  );
}

// ── Skeleton loader for index cards ──────────────────────────────────────────

function IndexCardSkeleton() {
  return (
    <div
      className="rounded-xl p-5 animate-pulse"
      style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="h-5 w-12 rounded-md" style={{ background: "hsl(215, 25%, 18%)" }} />
        <div className="h-3 w-10 rounded"    style={{ background: "hsl(215, 25%, 16%)" }} />
      </div>
      <div className="h-4 w-20 rounded mb-2" style={{ background: "hsl(215, 25%, 16%)" }} />
      <div className="h-8 w-28 rounded mb-3" style={{ background: "hsl(215, 25%, 16%)" }} />
      <div className="h-4 w-16 rounded"      style={{ background: "hsl(215, 25%, 16%)" }} />
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [indices,       setIndices]       = useState<IndexSummary[]>([]);
  const [indicesLoading, setIndicesLoading] = useState(true);
  const [selectedIndex, setSelectedIndex] = useState<Stock | null>(null);

  useEffect(() => {
    fetchIndices()
      .then(setIndices)
      .catch(() => setIndices([]))
      .finally(() => setIndicesLoading(false));
  }, []);

  // Convert IndexSummary → Stock for StockModal
  function indexAsStock(idx: IndexSummary): Stock {
    return {
      symbol:   idx.symbol,
      name:     INDEX_DISPLAY[idx.symbol]?.full ?? idx.name,
      price:    idx.price,
      change:   idx.change,
      volume:   "—",
      positive: idx.positive,
    };
  }

  return (
    <div className="flex min-h-screen bg-white">
      <Sidebar isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} />

      <main className="flex-1 bg-white">
        {/* Hamburger Menu Button */}
        {!isSidebarOpen && (
          <button
            onClick={() => setIsSidebarOpen(true)}
            className="fixed top-4 left-4 z-30 p-2 hover:bg-gray-100 rounded-md transition-colors"
            aria-label="Open menu"
          >
            <Menu className="w-8 h-8 text-gray-800" strokeWidth={1.5} />
          </button>
        )}

        <div className={`p-8 ${!isSidebarOpen ? "pt-20 sm:pt-8" : ""}`}>
          {/* Header Section */}
          <div className="mb-8">
            <div className="flex items-start justify-between mb-6 gap-4">
              <div className={`flex-1 ${!isSidebarOpen ? "pl-0 sm:pl-0" : ""}`}>
                <p className="text-gray-700 text-base leading-relaxed max-w-3xl">
                  Up-to-date information on the top 2000 stocks in the Russell 3000.<br />
                  Explore stock prices, stock correlations, and offset delays.<br />
                  Choose a sector below, or search for a stock to begin your analysis.
                </p>
              </div>
              <TrendingUp className="w-8 h-8 flex-shrink-0 text-gray-800" />
            </div>

            {/* Sector Filters */}
            <SectorFilter />
          </div>

          {/* ── Market Indices ─────────────────────────────────────────────── */}
          <div className="mb-6">
            <p
              className="text-[10px] font-bold uppercase tracking-widest mb-3"
              style={{ color: "hsl(215, 15%, 45%)" }}
            >
              Market Indices
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {indicesLoading
                ? Array.from({ length: 3 }).map((_, i) => <IndexCardSkeleton key={i} />)
                : indices.length > 0
                  ? indices.map((idx) => (
                      <IndexCard
                        key={idx.symbol}
                        index={idx}
                        onClick={() => setSelectedIndex(indexAsStock(idx))}
                      />
                    ))
                  : (
                    // Graceful fallback if indices endpoint fails
                    <p className="col-span-3 text-sm" style={{ color: "hsl(215, 15%, 45%)" }}>
                      Market indices unavailable — make sure the backend is running.
                    </p>
                  )
              }
            </div>
          </div>

          {/* ── Featured Stocks ────────────────────────────────────────────── */}
          <div className="bg-black rounded-xl p-6">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
              <h2 className="text-xl font-semibold text-white">Featured Stocks</h2>
              <div className="relative w-full sm:w-72">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="search"
                  placeholder="Search"
                  className="w-full pl-11 pr-4 py-2 bg-white rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>

            <StockGrid />
          </div>
        </div>
      </main>

      {/* Index modal */}
      {selectedIndex && (
        <StockModal stock={selectedIndex} onClose={() => setSelectedIndex(null)} />
      )}
    </div>
  );
}
