"use client";
import { useState, useRef, useMemo, useEffect } from "react";
import { Search } from "lucide-react";

const BG_INPUT  = "hsl(215,25%,9%)";
const BG_DROP   = "hsl(215,25%,13%)";
const BG_HOVER  = "hsl(215,25%,17%)";
const BORDER    = "hsl(215,20%,18%)";
const BORDER_D  = "hsl(215,20%,16%)";
const TEXT_PRI  = "hsl(210,40%,92%)";
const TEXT_SEC  = "hsl(215,15%,55%)";
const TEXT_MUT  = "hsl(215,15%,40%)";

interface Props {
  value: string;
  onChange: (symbol: string) => void;
  stocks: { symbol: string; name: string }[];
  label?: string;
  labelColor?: string;
  placeholder?: string;
}

export default function TickerSearchInput({
  value,
  onChange,
  stocks,
  label,
  labelColor = "hsl(217,91%,60%)",
  placeholder = "e.g. MSFT",
}: Props) {
  const [query,     setQuery]     = useState(value);
  const [open,      setOpen]      = useState(false);
  const [activeIdx, setActiveIdx] = useState(-1);
  const inputRef    = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Keep query in sync when value is changed externally (e.g. default state)
  useEffect(() => { setQuery(value); }, [value]);

  const suggestions = useMemo(() => {
    const q = query.trim().toUpperCase();
    if (!q) return [];
    return stocks
      .filter(s => s.symbol.startsWith(q) || s.name.toUpperCase().includes(q))
      .slice(0, 8);
  }, [query, stocks]);

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const select = (symbol: string) => {
    onChange(symbol);
    setQuery(symbol);
    setOpen(false);
    setActiveIdx(-1);
  };

  return (
    <div className="flex flex-col gap-1" ref={containerRef}>
      {label && (
        <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: labelColor }}>
          {label}
        </span>
      )}
      <div className="relative">
        <div
          className="flex items-center gap-2 px-3 py-2 rounded-lg"
          style={{ background: BG_INPUT, border: `1px solid ${labelColor}40` }}
        >
          <Search className="w-3.5 h-3.5 flex-shrink-0" style={{ color: TEXT_MUT }} />
          <input
            ref={inputRef}
            value={query}
            onChange={e => {
              setQuery(e.target.value.toUpperCase());
              setOpen(true);
              setActiveIdx(-1);
            }}
            onFocus={() => setOpen(true)}
            onKeyDown={e => {
              if (e.key === "ArrowDown") {
                e.preventDefault();
                setActiveIdx(i => Math.min(i + 1, suggestions.length - 1));
              }
              if (e.key === "ArrowUp") {
                e.preventDefault();
                setActiveIdx(i => Math.max(i - 1, -1));
              }
              if (e.key === "Escape") {
                setOpen(false);
                setActiveIdx(-1);
              }
              if (e.key === "Enter") {
                e.preventDefault();
                if (activeIdx >= 0 && suggestions[activeIdx]) {
                  select(suggestions[activeIdx].symbol);
                } else if (query.trim()) {
                  onChange(query.trim().toUpperCase());
                  setOpen(false);
                }
              }
            }}
            maxLength={10}
            placeholder={placeholder}
            className="flex-1 bg-transparent outline-none text-sm font-bold w-20"
            style={{ color: TEXT_PRI }}
          />
        </div>

        {open && suggestions.length > 0 && (
          <div
            className="absolute top-full left-0 mt-1 w-56 rounded-lg overflow-hidden z-20"
            style={{ background: BG_DROP, border: `1px solid ${BORDER}`, boxShadow: "0 8px 24px rgba(0,0,0,0.5)" }}
          >
            {suggestions.map((s, i) => (
              <button
                key={s.symbol}
                onMouseDown={e => { e.preventDefault(); select(s.symbol); }}
                className="w-full flex items-center gap-3 px-4 py-2.5 text-left text-sm"
                style={{
                  background:   i === activeIdx ? BG_HOVER : "transparent",
                  color:        TEXT_PRI,
                  borderBottom: i < suggestions.length - 1 ? `1px solid ${BORDER_D}` : "none",
                }}
                onMouseEnter={() => setActiveIdx(i)}
                onMouseLeave={() => setActiveIdx(-1)}
              >
                <span className="font-bold w-14 flex-shrink-0" style={{ color: labelColor }}>{s.symbol}</span>
                <span className="text-xs truncate" style={{ color: TEXT_SEC }}>{s.name}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
