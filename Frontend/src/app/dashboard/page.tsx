// frontend/src/app/dashboard/page.tsx
"use client";
import Sidebar from "@/components/ui/Sidebar";
import SectorFilter from "@/components/ui/SectorFilter";
import StockGrid from "@/components/ui/StockGrid";
import { StarredProvider } from "@/components/ui/StarredContext";

export default function DashboardPage() {
  return (
    <StarredProvider>
      <div className="min-h-screen" style={{ background: "hsl(213, 27%, 7%)" }}>
        <Sidebar />
        <main className="pt-14">
          <div className="max-w-7xl mx-auto px-6 py-6">
            {/* Index Cards */}
            <SectorFilter />
            {/* Watchlist + Gainers/Losers */}
            <StockGrid />
          </div>
        </main>
      </div>
    </StarredProvider>
  );
}