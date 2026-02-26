"use client";
import Sidebar from "@/components/ui/Sidebar";
import SectorFilter from "@/components/ui/SectorFilter";
import StockGrid from "@/components/ui/StockGrid";

export default function DashboardPage() {
  return (
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
  );
}