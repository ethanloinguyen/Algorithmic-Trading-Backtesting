// Frontend/src/app/lib/sectorData.ts
// Maps each featured ticker to its GICS sector, and exports helpers.

export const TICKER_SECTOR: Record<string, string> = {
  // Technology
  AAPL:  "Technology",
  MSFT:  "Technology",
  NVDA:  "Technology",
  AVGO:  "Technology",
  AMD:   "Technology",

  // Communication Services
  GOOGL: "Communication Services",
  META:  "Communication Services",

  // Consumer Discretionary
  AMZN:  "Consumer Discretionary",
  TSLA:  "Consumer Discretionary",

  // Financials
  JPM:   "Financials",
  V:     "Financials",

  // Health Care
  UNH:   "Health Care",
  JNJ:   "Health Care",

  // Energy
  XOM:   "Energy",

  // Consumer Staples
  WMT:   "Consumer Staples",
};

// Unique ordered list of sectors (All first, then alphabetical)
export const ALL_SECTORS = "All Sectors";

export const SECTORS: string[] = [
  ALL_SECTORS,
  ...Array.from(new Set(Object.values(TICKER_SECTOR))).sort(),
];

/** Return the sector for a given ticker, or "Other" if unknown. */
export function sectorFor(symbol: string): string {
  return TICKER_SECTOR[symbol.toUpperCase()] ?? "Other";
}

/** Filter a list of items that have a `symbol` field by sector. */
export function filterBySector<T extends { symbol: string }>(
  items: T[],
  sector: string,
): T[] {
  if (sector === ALL_SECTORS) return items;
  return items.filter((item) => sectorFor(item.symbol) === sector);
}
