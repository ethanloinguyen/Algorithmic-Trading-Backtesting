// frontend/src/components/ui/SectorFilter.tsx
"use client";
import { useState, useEffect } from "react";
import StockModal from "./StockModal";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

interface IndexData {
  symbol: string;
  name: string;
  value: string;
  change: string;
  pct: string;
  price: string;
  positive: boolean;
}

// Fallback static data shown while loading or on error
const FALLBACK: IndexData[] = [
  { symbol: "SPX", name: "S&P 500 Index", value: "5,248.49", change: "+32.64", pct: "+0.63%", price: "$5,248.49", positive: true },
  { symbol: "IXIC", name: "NASDAQ Composite", value: "16,742.39", change: "-45.32", pct: "-0.27%", price: "$16,742.39", positive: false },
  { symbol: "DJI", name: "Dow Jones Industrial Average", value: "39,512.84", change: "+125.69", pct: "+0.32%", price: "$39,512.84", positive: true },
];

export default function SectorFilter() {
  const [indices, setIndices]       = useState<IndexData[]>(FALLBACK);
  const [loading, setLoading]       = useState(true);
  const [modalStock, setModalStock] = useState<IndexData | null>(null);

  useEffect(() => {
    async function fetchIndices() {
      try {
        const res = await fetch(`${API}/api/indices`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (json.data && json.data.length > 0) {
          setIndices(json.data);
        } else {
          setIndices(FALLBACK);
        }
      } catch (err) {
        console.error("Failed to fetch indices:", err);
        // Keep fallback — dashboard is still usable
      } finally {
        setLoading(false);
      }
    }
    fetchIndices();
  }, []);

  return (
    <>
      <div className="grid grid-cols-3 gap-4 mb-6">
        {indices.map((idx) => (
          <div
            key={idx.symbol}
            className="rounded-xl p-5 cursor-pointer transition-all"
            style={{ background: "hsl(215, 25%, 11%)", border: "1px solid hsl(215, 20%, 18%)" }}
            onClick={() => !loading && setModalStock(idx)}
            onMouseEnter={e => {
              if (!loading) {
                (e.currentTarget as HTMLDivElement).style.border = "1px solid hsl(215, 20%, 28%)";
                (e.currentTarget as HTMLDivElement).style.background = "hsl(215, 25%, 13%)";
              }
            }}
            onMouseLeave={e => {
              (e.currentTarget as HTMLDivElement).style.border = "1px solid hsl(215, 20%, 18%)";
              (e.currentTarget as HTMLDivElement).style.background = "hsl(215, 25%, 11%)";
            }}
          >
            <p className="text-xs font-medium mb-2" style={{ color: "hsl(215, 15%, 55%)" }}>{idx.symbol}</p>

            <div className="flex items-center justify-between mb-1">
              {loading ? (
                <div className="h-7 w-28 rounded animate-pulse" style={{ background: "hsl(215, 20%, 20%)" }} />
              ) : (
                <p className="text-2xl font-bold" style={{ color: "hsl(210, 40%, 92%)" }}>{idx.value}</p>
              )}
              <svg width="60" height="20" viewBox="0 0 60 20">
                <polyline
                  fill="none"
                  stroke={idx.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)"}
                  strokeWidth="2"
                  strokeLinecap="round"
                  points={idx.positive
                    ? "0,14 12,12 24,10 36,8 48,9 60,5"
                    : "0,5 12,8 24,9 36,12 48,11 60,14"}
                />
              </svg>
            </div>

            {loading ? (
              <div className="h-4 w-24 rounded animate-pulse mt-1" style={{ background: "hsl(215, 20%, 20%)" }} />
            ) : (
              <p className="text-sm font-medium" style={{ color: idx.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)" }}>
                {idx.change} ({idx.pct})
              </p>
            )}
          </div>
        ))}
      </div>

      {modalStock && (
        <StockModal
          stock={{
            symbol:   modalStock.symbol,
            name:     modalStock.name,
            price:    modalStock.price,
            change:   `${modalStock.change} (${modalStock.pct})`,
            positive: modalStock.positive,
          }}
          onClose={() => setModalStock(null)}
        />
      )}
    </>
  );
}