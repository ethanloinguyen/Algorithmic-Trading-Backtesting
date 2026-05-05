// Frontend/components/ui/StockGrid.tsx
"use client";
import { useState, useEffect } from "react";
import { Star, Loader2 } from "lucide-react";
import { useAuth } from "@/src/app/context/AuthContext";
import StockModal, { type Stock } from "@/components/ui/StockModal";
import { fetchAllStocks, type StockSummary } from "@/src/app/lib/api";
import { getCachedSummaries } from "@/src/app/lib/stockCache";

// ── Star button ───────────────────────────────────────────────────────────────

function StarBtn({ stock, size = "md" }: { stock: Stock; size?: "sm" | "md" }) {
  const { isSaved, toggleSave } = useAuth();
  const saved = isSaved(stock.symbol);
  const cls   = size === "sm" ? "w-3.5 h-3.5" : "w-4 h-4";
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        toggleSave({ symbol: stock.symbol, name: stock.name });
      }}
      aria-label={saved ? "Unsave" : "Save"}
      className="flex-shrink-0 transition-opacity hover:opacity-70"
    >
      <Star
        className={cls}
        style={{
          fill:  saved ? "hsl(48,96%,53%)" : "transparent",
          color: saved ? "hsl(48,96%,53%)" : size === "sm" ? "hsl(215,15%,38%)" : "hsl(215,15%,45%)",
        }}
      />
    </button>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function StockGrid() {
  const { trackClick } = useAuth();
  const [selectedStock, setSelectedStock] = useState<Stock | null>(null);
  const [stocks,        setStocks]        = useState<StockSummary[]>([]);
  const [loading,       setLoading]       = useState(true);
  const [error,         setError]         = useState<string | null>(null);
  const [fromCache,     setFromCache]     = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      // ── Step 1: Try Firestore cache first (instant) ──────────────────────
      const cached = await getCachedSummaries();
      if (cached && !cancelled) {
        setStocks(cached.data);
        setLoading(false);
        setFromCache(true);

        // If cache is fresh, stop here — no need to hit BigQuery
        if (!cached.stale) return;
      }

      // ── Step 2: Fetch from BigQuery via backend ──────────────────────────
      try {
        const fresh = await fetchAllStocks();
        if (!cancelled) {
          setStocks(fresh);
          setLoading(false);
          setFromCache(false);
          setError(null);
        }
      } catch (err: unknown) {
        if (!cancelled) {
          const msg = err instanceof Error ? err.message : "Unknown error";
          // Only show error if we have no cached data to show
          if (!cached) {
            setError(msg);
            setLoading(false);
          }
        }
      }
    }

    load();
    return () => { cancelled = true; };
  }, []);

  // Derive top 5 gainers and losers from live data
  const byChange = (s: StockSummary) =>
    parseFloat(s.change.replace("%", "").replace("+", ""));
  const topGainers = [...stocks].sort((a, b) => byChange(b) - byChange(a)).slice(0, 5);
  const topLosers  = [...stocks].sort((a, b) => byChange(a) - byChange(b)).slice(0, 5);

  const handleStockClick = (stock: Stock) => {
    setSelectedStock(stock);
    trackClick(stock.symbol); // records click + pre-warms OHLCV cache
  };

  return (
    <>
      <div className="flex gap-4">

        {/* ── Watchlist ── */}
        <div
          className="flex-1 rounded-xl overflow-hidden"
          style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
        >
          {/* Header */}
          <div className="px-6 py-4 flex items-center justify-between">
            <h2 className="text-base font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>
              Watchlist
            </h2>
            {fromCache && !loading && (
              <span className="text-xs" style={{ color: "hsl(215,15%,40%)" }}>
                cached
              </span>
            )}
          </div>

          {/* Column headers */}
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

          {/* Loading */}
          {loading && (
            <div className="flex items-center justify-center py-14">
              <Loader2 className="w-5 h-5 animate-spin" style={{ color: "hsl(217,91%,60%)" }} />
            </div>
          )}

          {/* Error */}
          {!loading && error && (
            <div className="px-6 py-10 text-center">
              <p className="text-sm mb-1" style={{ color: "hsl(0,84%,60%)" }}>
                Could not load stocks
              </p>
              <p className="text-xs" style={{ color: "hsl(215,15%,45%)" }}>
                Make sure the backend is running: <code>python run.py</code> in <code>backend/</code>
              </p>
            </div>
          )}

          {/* Rows */}
          {!loading && !error && stocks.map((stock) => (
            <div
              key={stock.symbol}
              onClick={() => handleStockClick(stock)}
              className="grid px-6 py-3 cursor-pointer items-center"
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
                {stock.price}
              </span>
              <span
                className="text-sm font-medium text-right"
                style={{ color: stock.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)" }}
              >
                {stock.change}
              </span>
              <span className="text-sm text-right" style={{ color: "hsl(215, 15%, 55%)" }}>
                {stock.volume}
              </span>
              <div className="flex justify-end">
                <StarBtn stock={stock} size="md" />
              </div>
            </div>
          ))}
        </div>

        {/* ── Right column ── */}
        <div className="w-72 flex flex-col gap-4">

          {/* Top Gainers */}
          <div
            className="rounded-xl p-5 flex-1"
            style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
          >
            <div className="flex items-center gap-2 mb-3">
              <span style={{ color: "hsl(142, 71%, 45%)" }}>📈</span>
              <h3 className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>
                Top Gainers
              </h3>
            </div>
            {loading ? (
              <div className="flex justify-center py-4">
                <Loader2 className="w-4 h-4 animate-spin" style={{ color: "hsl(217,91%,60%)" }} />
              </div>
            ) : (
              <div className="space-y-1">
                {topGainers.map((s) => (
                  <div
                    key={s.symbol}
                    onClick={() => handleStockClick(s)}
                    className="flex items-center gap-2 cursor-pointer rounded-lg px-2 py-1.5"
                    onMouseEnter={(e) => (e.currentTarget.style.background = "hsl(215, 25%, 14%)")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                  >
                    <span className="text-sm font-medium flex-1" style={{ color: "hsl(210, 40%, 85%)" }}>
                      {s.symbol}
                    </span>
                    <span className="text-sm font-semibold" style={{ color: "hsl(142, 71%, 45%)" }}>
                      {s.change}
                    </span>
                    <StarBtn stock={s} size="sm" />
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Top Losers */}
          <div
            className="rounded-xl p-5 flex-1"
            style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
          >
            <div className="flex items-center gap-2 mb-3">
              <span style={{ color: "hsl(0, 84%, 60%)" }}>📉</span>
              <h3 className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>
                Top Losers
              </h3>
            </div>
            {loading ? (
              <div className="flex justify-center py-4">
                <Loader2 className="w-4 h-4 animate-spin" style={{ color: "hsl(217,91%,60%)" }} />
              </div>
            ) : (
              <div className="space-y-1">
                {topLosers.map((s) => (
                  <div
                    key={s.symbol}
                    onClick={() => handleStockClick(s)}
                    className="flex items-center gap-2 cursor-pointer rounded-lg px-2 py-1.5"
                    onMouseEnter={(e) => (e.currentTarget.style.background = "hsl(215, 25%, 14%)")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                  >
                    <span className="text-sm font-medium flex-1" style={{ color: "hsl(210, 40%, 85%)" }}>
                      {s.symbol}
                    </span>
                    <span className="text-sm font-semibold" style={{ color: "hsl(0, 84%, 60%)" }}>
                      {s.change}
                    </span>
                    <StarBtn stock={s} size="sm" />
                  </div>
                ))}
              </div>
            )}
          </div>

        </div>
      </div>

      {selectedStock && (
        <StockModal stock={selectedStock} onClose={() => setSelectedStock(null)} />
      )}
    </>
  );
}