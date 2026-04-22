// Frontend/components/ui/SectorFilter.tsx
"use client";
import { useEffect, useState } from "react";
import { fetchIndices, type IndexSummary } from "@/src/app/lib/api";
import { getCachedIndices } from "@/src/app/lib/stockCache";

const DISPLAY_ORDER = ["SPX", "IXIC", "DJI"];
const SPARKLINES = {
  up:   "0,22 12,18 24,20 36,14 48,16 60,10 72,13 84,9  96,11 100,8",
  down: "0,8  12,11 24,9  36,14 48,12 60,18 72,15 84,20 96,17 100,22",
};

export default function SectorFilter() {
  const [indices, setIndices] = useState<IndexSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      // ── 1. Read Firestore cache written by backend ────────────────────────
      const cached = await getCachedIndices();
      if (cached && !cancelled) {
        setIndices(cached.data); // ← no cast needed — getCachedIndices returns IndexSummary[]
        setLoading(false);
        if (!cached.stale) return;
      }

      // ── 2. Call backend (checks its own Firestore cache, then BigQuery) ───
      try {
        const fresh = await fetchIndices();
        if (!cancelled) {
          setIndices(fresh);
          setLoading(false);
          setError(false);
        }
      } catch {
        if (!cancelled && !cached) {
          setError(true);
          setLoading(false);
        }
      }
    }

    load();
    return () => { cancelled = true; };
  }, []);

  if (loading) {
    return (
      <div className="flex gap-4 mb-6">
        {[0, 1, 2].map(i => (
          <div key={i} className="flex-1 rounded-xl animate-pulse"
            style={{ background: "hsl(215,25%,11%)", border: "1px solid hsl(215,20%,18%)", height: 96 }} />
        ))}
      </div>
    );
  }

  if (error || indices.length === 0) {
    return (
      <div className="rounded-xl px-5 py-3 mb-6 text-xs"
        style={{ background: "hsl(215,25%,11%)", border: "1px solid hsl(215,20%,18%)", color: "hsl(215,15%,45%)" }}>
        Index data unavailable — make sure the backend is running.
      </div>
    );
  }

  const display = DISPLAY_ORDER
    .map(sym => indices.find(i => i.symbol === sym))
    .filter((i): i is IndexSummary => Boolean(i));
  const cards = display.length ? display : indices;

  return (
    <div className="flex gap-4 mb-6">
      {cards.map((idx) => {
        const lineColor = idx.positive ? "hsl(142,71%,45%)" : "hsl(0,84%,60%)";
        const sparkline = idx.positive ? SPARKLINES.up : SPARKLINES.down;
        return (
          <div key={idx.symbol} className="flex-1 rounded-xl p-5"
            style={{ background: "hsl(215,25%,11%)", border: "1px solid hsl(215,20%,18%)" }}>
            <div className="flex items-start justify-between mb-3">
              <div>
                <p className="text-xs font-medium mb-0.5" style={{ color: "hsl(215,15%,50%)" }}>
                  {idx.name}
                </p>
                <p className="text-lg font-bold" style={{ color: "hsl(210,40%,92%)" }}>
                  {idx.value}
                </p>
              </div>
              <div className="text-right">
                <span className="text-xs font-semibold px-2 py-0.5 rounded-full"
                  style={{
                    background: idx.positive ? "hsla(142,71%,45%,0.12)" : "hsla(0,84%,60%,0.12)",
                    color: lineColor,
                  }}>
                  {idx.pct}
                </span>
                <p className="text-xs mt-1" style={{ color: "hsl(215,15%,50%)" }}>{idx.change}</p>
              </div>
            </div>
            <svg width="100%" viewBox="0 0 100 30" preserveAspectRatio="none" style={{ display: "block" }}>
              <defs>
                <linearGradient id={`spark-${idx.symbol}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor={lineColor} stopOpacity="0.2" />
                  <stop offset="100%" stopColor={lineColor} stopOpacity="0"   />
                </linearGradient>
              </defs>
              <polygon fill={`url(#spark-${idx.symbol})`} points={`0,30 ${sparkline} 100,30`} />
              <polyline fill="none" stroke={lineColor} strokeWidth="1.5"
                strokeLinecap="round" strokeLinejoin="round" points={sparkline} />
            </svg>
          </div>
        );
      })}
    </div>
  );
}