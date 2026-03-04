import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Search, Star, BarChart2, Trash2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

// Mock API (Replace with FastAPI endpoints later)
const fetchStocks = async (query = "") => {
  const sample = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "NVDA",
    "JPM",
    "V",
    "UNH",
  ];
  return sample.filter((s) => s.includes(query.toUpperCase()));
};

export default function StockDashboard() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [saved, setSaved] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      const res = await fetchStocks(query);
      setResults(res);
      setLoading(false);
    };
    load();
  }, [query]);

  const saveStock = (ticker) => {
    if (!saved.includes(ticker)) {
      setSaved([...saved, ticker]);
    }
  };

  const removeStock = (ticker) => {
    setSaved(saved.filter((s) => s !== ticker));
  };

  const analyzeStocks = () => {
    alert(`Analyzing: ${saved.join(", ")}`);
    // Replace with API call to /analyze endpoint
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-6xl mx-auto mb-8"
      >
        <h1 className="text-3xl font-bold text-gray-900">Stock Correlation Lab</h1>
        <p className="text-gray-600 mt-1">
          Discover relationships and trends between market assets
        </p>
      </motion.header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
        {/* Search Panel */}
        <motion.div
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <Card className="rounded-2xl shadow-sm">
            <CardContent className="p-5">
              <h2 className="font-semibold text-lg mb-3 flex items-center gap-2">
                <Search size={18} /> Search Tickers
              </h2>

              <Input
                placeholder="Search (e.g. AAPL, MSFT...)"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="mb-4"
              />

              {loading && (
                <p className="text-sm text-gray-500">Loading...</p>
              )}

              <div className="space-y-2 max-h-64 overflow-y-auto">
                {results.map((ticker) => (
                  <motion.div
                    key={ticker}
                    layout
                    whileHover={{ scale: 1.02 }}
                    className="flex items-center justify-between p-2 border rounded-xl"
                  >
                    <span className="font-medium">{ticker}</span>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => saveStock(ticker)}
                    >
                      <Star size={16} />
                    </Button>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Saved Stocks */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="rounded-2xl shadow-sm">
            <CardContent className="p-5">
              <h2 className="font-semibold text-lg mb-3 flex items-center gap-2">
                <Star size={18} /> Watchlist
              </h2>

              {saved.length === 0 && (
                <p className="text-gray-500 text-sm">
                  No stocks saved yet
                </p>
              )}

              <div className="space-y-2">
                {saved.map((ticker) => (
                  <motion.div
                    key={ticker}
                    layout
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex items-center justify-between p-2 border rounded-xl bg-white"
                  >
                    <span className="font-medium">{ticker}</span>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => removeStock(ticker)}
                    >
                      <Trash2 size={16} />
                    </Button>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Analysis Panel */}
        <motion.div
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <Card className="rounded-2xl shadow-sm h-full">
            <CardContent className="p-5 flex flex-col h-full">
              <h2 className="font-semibold text-lg mb-3 flex items-center gap-2">
                <BarChart2 size={18} /> Analysis
              </h2>

              <div className="flex-1">
                <p className="text-sm text-gray-600 mb-4">
                  Analyze correlations, lagged covariance, and historical trends
                  between your selected stocks.
                </p>

                <div className="bg-gray-100 rounded-xl p-4 text-sm text-gray-700">
                  <p className="font-medium mb-1">Selected:</p>
                  {saved.length > 0
                    ? saved.join(", ")
                    : "None"}
                </div>
              </div>

              <Button
                className="mt-4 w-full rounded-xl"
                disabled={saved.length < 2}
                onClick={analyzeStocks}
              >
                Run Analysis
              </Button>

              {saved.length < 2 && (
                <p className="text-xs text-gray-500 mt-2 text-center">
                  Select at least 2 stocks to analyze
                </p>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
