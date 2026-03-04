// frontend/src/context/StarredContext.tsx
"use client";
import { createContext, useContext, useState, useEffect, useCallback } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";
const STORAGE_KEY = "laglens_starred_symbols";

export interface StockInfo {
  symbol: string;
  name: string;
  price: string;
  change: string;
  volume: string;
  positive: boolean;
}

interface StarredContextValue {
  starred: StockInfo[];
  savedSymbols: Set<string>;
  toggleStar: (stock: StockInfo) => void;
  refreshStarred: () => Promise<void>;
}

const StarredContext = createContext<StarredContextValue | null>(null);

export function StarredProvider({ children }: { children: React.ReactNode }) {
  const [starred, setStarred] = useState<StockInfo[]>([]);

  // Load saved symbols from localStorage on mount and fetch fresh prices
  useEffect(() => {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    try {
      const symbols: string[] = JSON.parse(raw);
      if (symbols.length > 0) fetchFreshData(symbols);
    } catch {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  // Fetch latest prices from API for a list of symbols
  const fetchFreshData = async (symbols: string[]) => {
    try {
      const res = await fetch(`${API}/api/stocks/summaries?symbols=${symbols.join(",")}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setStarred(json.data);
    } catch (err) {
      console.error("Failed to refresh starred stocks:", err);
    }
  };

  // Public refresh function for the profile page to call on mount
  const refreshStarred = useCallback(async () => {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    try {
      const symbols: string[] = JSON.parse(raw);
      if (symbols.length > 0) await fetchFreshData(symbols);
    } catch {
      // ignore
    }
  }, []);

  const savedSymbols = new Set(starred.map(s => s.symbol));

  const toggleStar = useCallback((stock: StockInfo) => {
    setStarred(prev => {
      const exists = prev.some(s => s.symbol === stock.symbol);
      const next = exists
        ? prev.filter(s => s.symbol !== stock.symbol)
        : [...prev, stock];

      // Persist just the symbols list to localStorage
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next.map(s => s.symbol)));
      return next;
    });
  }, []);

  return (
    <StarredContext.Provider value={{ starred, savedSymbols, toggleStar, refreshStarred }}>
      {children}
    </StarredContext.Provider>
  );
}

export function useStarred() {
  const ctx = useContext(StarredContext);
  if (!ctx) throw new Error("useStarred must be used inside StarredProvider");
  return ctx;
}