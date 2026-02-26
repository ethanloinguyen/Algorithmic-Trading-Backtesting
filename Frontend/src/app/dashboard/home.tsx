"use client";
import { useState } from "react";
import Sidebar from "@/components/ui/Sidebar";
import SectorFilter from "@/components/ui/SectorFilter";
import StockGrid from "@/components/ui/StockGrid";
import { Search, TrendingUp, Menu } from "lucide-react";

export default function DashboardPage() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-screen bg-white">
      <Sidebar isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} />
      
      <main className="flex-1 bg-white">
        {/* Hamburger Menu Button - Shows when sidebar is closed */}
        {!isSidebarOpen && (
          <button
            onClick={() => setIsSidebarOpen(true)}
            className="fixed top-4 left-4 z-30 p-2 hover:bg-gray-100 rounded-md transition-colors"
            aria-label="Open menu"
          >
            <Menu className="w-8 h-8 text-gray-800" strokeWidth={1.5} />
          </button>
        )}

        <div className={`p-8 ${!isSidebarOpen ? 'pt-20 sm:pt-8' : ''}`}>
          {/* Header Section */}
          <div className="mb-8">
            <div className="flex items-start justify-between mb-6 gap-4">
              <div className={`flex-1 ${!isSidebarOpen ? 'pl-0 sm:pl-0' : ''}`}>
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

          {/* Featured Stocks Section */}
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

            {/* Stock Grid */}
            <StockGrid />
          </div>
        </div>
      </main>
    </div>
  );
}