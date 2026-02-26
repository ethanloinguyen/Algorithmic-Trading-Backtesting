"use client";

const indices = [
  { symbol: "SPX", value: "5,248.49", change: "+32.64", pct: "+0.63%", positive: true },
  { symbol: "IXIC", value: "16,742.39", change: "-45.32", pct: "-0.27%", positive: false },
  { symbol: "DJI", value: "39,512.84", change: "+125.69", pct: "+0.32%", positive: true },
];

export default function SectorFilter() {
  return (
    <div className="grid grid-cols-3 gap-4 mb-6">
      {indices.map((idx) => (
        <div
          key={idx.symbol}
          className="rounded-xl p-5"
          style={{
            background: "hsl(215, 25%, 11%)",
            border: "1px solid hsl(215, 20%, 18%)",
          }}
        >
          <p className="text-xs font-medium mb-2" style={{ color: "hsl(215, 15%, 55%)" }}>
            {idx.symbol}
          </p>
          <div className="flex items-center justify-between mb-1">
            <p className="text-2xl font-bold" style={{ color: "hsl(210, 40%, 92%)" }}>
              {idx.value}
            </p>
            {/* Mini sparkline */}
            <svg width="60" height="20" viewBox="0 0 60 20">
              <polyline
                fill="none"
                stroke={idx.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)"}
                strokeWidth="2"
                strokeLinecap="round"
                points={
                  idx.positive
                    ? "0,14 12,12 24,10 36,8 48,9 60,5"
                    : "0,5 12,8 24,9 36,12 48,11 60,14"
                }
              />
            </svg>
          </div>
          <p
            className="text-sm font-medium"
            style={{ color: idx.positive ? "hsl(142, 71%, 45%)" : "hsl(0, 84%, 60%)" }}
          >
            {idx.change} ({idx.pct})
          </p>
        </div>
      ))}
    </div>
  );
}