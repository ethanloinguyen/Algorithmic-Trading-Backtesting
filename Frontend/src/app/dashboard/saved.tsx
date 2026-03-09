// frontend/src/app/dashboard/saved/page.tsx
"use client";
import Sidebar from "@/components/ui/Sidebar";
import { useAuth } from "@/context/AuthContext";
import { Star, Loader2 } from "lucide-react";

export default function SavedStocksPage() {
  const { savedStocks, toggleSave, loading } = useAuth();

  return (
    <div className="min-h-screen" style={{ background: "hsl(213, 27%, 7%)" }}>
      <Sidebar />

      <main className="pt-14">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <h1 className="text-xl font-semibold mb-6" style={{ color: "hsl(210, 40%, 92%)" }}>
            My Saved Stocks
          </h1>

          {/* Loading state */}
          {loading ? (
            <div className="flex items-center justify-center py-24">
              <Loader2 className="w-6 h-6 animate-spin" style={{ color: "hsl(217, 91%, 60%)" }} />
            </div>

          ) : savedStocks.length === 0 ? (
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
            <>
              <div
                className="rounded-xl overflow-hidden"
                style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
              >
                {/* Table Header */}
                <div
                  className="grid px-6 py-3 text-xs font-medium"
                  style={{
                    gridTemplateColumns: "1fr 1fr 40px",
                    color:               "hsl(215, 15%, 50%)",
                    borderBottom:        "1px solid hsl(215, 20%, 18%)",
                  }}
                >
                  <span>Ticker</span>
                  <span>Company</span>
                  <span />
                </div>

                {savedStocks.map((stock) => (
                  <div
                    key={stock.symbol}
                    className="grid px-6 py-3.5 transition-colors items-center"
                    style={{
                      gridTemplateColumns: "1fr 1fr 40px",
                      borderTop:          "1px solid hsl(215, 20%, 16%)",
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = "hsl(215, 25%, 14%)")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                  >
                    <span className="text-sm font-semibold" style={{ color: "hsl(210, 40%, 92%)" }}>
                      {stock.symbol}
                    </span>
                    <span className="text-sm" style={{ color: "hsl(215, 15%, 55%)" }}>
                      {stock.name}
                    </span>
                    <div className="flex justify-end">
                      <button
                        onClick={() => toggleSave(stock)}
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
                ))}
              </div>

              {/* Summary card */}
              <div className="mt-6">
                <div
                  className="inline-flex items-center gap-3 rounded-xl px-6 py-4"
                  style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
                >
                  <p className="text-sm" style={{ color: "hsl(215, 15%, 55%)" }}>Total saved</p>
                  <p className="text-2xl font-bold" style={{ color: "hsl(217, 91%, 60%)" }}>
                    {savedStocks.length}
                  </p>
                </div>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}