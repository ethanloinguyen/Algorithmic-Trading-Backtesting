"use client";
import { useState, useEffect, useRef } from "react";
import { X } from "lucide-react";

const BG_MODAL = "hsl(215, 25%, 11%)";
const BORDER   = "hsl(215, 20%, 18%)";
const BORDER_D = "hsl(215, 20%, 16%)";
const TEXT_PRI = "hsl(210, 40%, 92%)";
const TEXT_SEC = "hsl(215, 15%, 55%)";
const BLUE     = "hsl(217, 91%, 60%)";
const BLUE_DIM = "hsla(217, 91%, 60%, 0.15)";

export interface HelpSection {
  title: string;
  body: string;
  color?: string;
}

interface PageHelpProps {
  title: string;
  subtitle?: string;
  sections: HelpSection[];
}

export function PageHelp({ title, subtitle, sections }: PageHelpProps) {
  const [open, setOpen] = useState(false);
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="flex items-center justify-center rounded-full transition-all hover:opacity-80"
        style={{
          width: 28,
          height: 28,
          background: BLUE_DIM,
          border: `1px solid hsla(217, 91%, 60%, 0.35)`,
          color: BLUE,
          flexShrink: 0,
        }}
        aria-label="Help"
        title="Page guide"
      >
        <span style={{ fontSize: 13, fontWeight: 700, lineHeight: 1 }}>i</span>
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-6"
          style={{ background: "rgba(5, 10, 18, 0.78)" }}
          onClick={e => { if (!modalRef.current?.contains(e.target as Node)) setOpen(false); }}
        >
          <div
            ref={modalRef}
            className="w-full flex flex-col rounded-2xl overflow-hidden"
            style={{
              maxWidth: 520,
              maxHeight: "85vh",
              background: BG_MODAL,
              border: `1px solid ${BORDER}`,
              boxShadow: "0 24px 64px rgba(0,0,0,0.75)",
            }}
          >
            {/* Header */}
            <div
              className="px-6 py-5 flex items-start justify-between gap-4 flex-shrink-0"
              style={{ borderBottom: `1px solid ${BORDER}` }}
            >
              <div>
                <h2 className="text-base font-bold mb-0.5" style={{ color: TEXT_PRI }}>{title}</h2>
                {subtitle && (
                  <p className="text-xs leading-relaxed mt-1" style={{ color: TEXT_SEC }}>{subtitle}</p>
                )}
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                <span
                  className="flex items-center justify-center rounded-full text-xs font-bold"
                  style={{ width: 22, height: 22, background: BLUE_DIM, color: BLUE }}
                >
                  i
                </span>
                <button
                  onClick={() => setOpen(false)}
                  className="flex items-center justify-center rounded-lg transition-opacity hover:opacity-70"
                  style={{ width: 28, height: 28, background: "hsl(215,25%,16%)", color: TEXT_SEC }}
                  aria-label="Close"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Sections */}
            <div className="overflow-y-auto px-6 py-5 space-y-4">
              {sections.map((s, i) => (
                <div
                  key={i}
                  className="flex gap-4 rounded-xl p-4"
                  style={{ background: "hsl(215,25%,9%)", border: `1px solid ${BORDER_D}` }}
                >
                  <div
                    className="flex items-center justify-center rounded-full text-xs font-bold flex-shrink-0"
                    style={{
                      width: 26,
                      height: 26,
                      marginTop: 1,
                      background: s.color ? `${s.color}20` : BLUE_DIM,
                      color: s.color ?? BLUE,
                    }}
                  >
                    {i + 1}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold mb-1" style={{ color: TEXT_PRI }}>{s.title}</p>
                    <p className="text-xs leading-relaxed" style={{ color: TEXT_SEC }}>{s.body}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
