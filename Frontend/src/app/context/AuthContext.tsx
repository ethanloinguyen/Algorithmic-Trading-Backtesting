// Frontend/src/app/context/AuthContext.tsx
"use client";
import {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  ReactNode,
} from "react";
import { onAuthStateChanged, signOut, User } from "firebase/auth";
import {
  doc,
  getDoc,
  setDoc,
  updateDoc,
  arrayUnion,
} from "firebase/firestore";
import { auth, db } from "@/src/app/lib/firebase";
import { recordStockClick, getTopClickedSymbols } from "@/src/app/lib/stockCache";
import { fetchOHLCV } from "@/src/app/lib/api";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SavedStock {
  symbol: string;
  name:   string;
}

export interface CustomPortfolio {
  id:      string;    // uuid generated client-side
  name:    string;    // user-given name e.g. "Tech Picks"
  tickers: string[];  // list of ticker symbols e.g. ["AAPL", "MSFT"]
}

interface AuthContextValue {
  user:             User | null;
  loading:          boolean;
  savedStocks:      SavedStock[];
  isSaved:          (symbol: string) => boolean;
  toggleSave:       (stock: SavedStock) => Promise<void>;
  trackClick:       (symbol: string) => Promise<void>;
  logout:           () => Promise<void>;
  // Portfolio management
  customPortfolios:   CustomPortfolio[];
  createPortfolio:    (name: string, tickers: string[]) => Promise<void>;
  updatePortfolio:    (id: string, name: string, tickers: string[]) => Promise<void>;
  deletePortfolio:    (id: string) => Promise<void>;
}

// ── Context ───────────────────────────────────────────────────────────────────

const AuthContext = createContext<AuthContextValue>({
  user:             null,
  loading:          true,
  savedStocks:      [],
  isSaved:          () => false,
  toggleSave:       async () => {},
  trackClick:       async () => {},
  logout:           async () => {},
  customPortfolios: [],
  createPortfolio:  async () => {},
  updatePortfolio:  async () => {},
  deletePortfolio:  async () => {},
});

export function useAuth() {
  return useContext(AuthContext);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function generateId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

// ── Provider ──────────────────────────────────────────────────────────────────

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user,             setUser]             = useState<User | null>(null);
  const [loading,          setLoading]          = useState(true);
  const [savedStocks,      setSavedStocks]      = useState<SavedStock[]>([]);
  const [customPortfolios, setCustomPortfolios] = useState<CustomPortfolio[]>([]);

  // Load user data from Firestore on login
  const loadUserData = useCallback(async (uid: string) => {
    const ref  = doc(db, "users", uid);
    const snap = await getDoc(ref);

    if (snap.exists()) {
      const data = snap.data();
      setSavedStocks((data.savedStocks      as SavedStock[])      ?? []);
      setCustomPortfolios((data.portfolios  as CustomPortfolio[]) ?? []);
    } else {
      // First login — create user document with all fields
      await setDoc(ref, {
        savedStocks:   [],
        clickedStocks: {},
        portfolios:    [],
      });
      setSavedStocks([]);
      setCustomPortfolios([]);
    }

    prewarmTopStocks(uid);
  }, []);

  // Auth state listener
  useEffect(() => {
    const unsub = onAuthStateChanged(auth, async (firebaseUser) => {
      setUser(firebaseUser);
      if (firebaseUser) {
        try {
          await loadUserData(firebaseUser.uid);
        } catch (err) {
          console.error("Failed to load user data from Firestore:", err);
          setSavedStocks([]);
          setCustomPortfolios([]);
        }
      } else {
        setSavedStocks([]);
        setCustomPortfolios([]);
      }
      setLoading(false);
    });
    return unsub;
  }, [loadUserData]);

  // ── Watchlist ──────────────────────────────────────────────────────────────

  const toggleSave = useCallback(async (stock: SavedStock) => {
    if (!user) return;
    const ref          = doc(db, "users", user.uid);
    const alreadySaved = savedStocks.some(s => s.symbol === stock.symbol);

    if (alreadySaved) {
      const updated = savedStocks.filter(s => s.symbol !== stock.symbol);
      setSavedStocks(updated);
      await setDoc(ref, { savedStocks: updated }, { merge: true });
    } else {
      setSavedStocks(prev => [...prev, stock]);
      await updateDoc(ref, { savedStocks: arrayUnion(stock) });
    }
  }, [user, savedStocks]);

  // ── Portfolio CRUD ─────────────────────────────────────────────────────────

  const createPortfolio = useCallback(async (name: string, tickers: string[]) => {
    if (!user) return;
    const newPortfolio: CustomPortfolio = {
      id:      generateId(),
      name:    name.trim(),
      tickers: tickers.map(t => t.toUpperCase()).filter(Boolean),
    };
    const updated = [...customPortfolios, newPortfolio];
    setCustomPortfolios(updated);
    await setDoc(
      doc(db, "users", user.uid),
      { portfolios: updated },
      { merge: true }
    );
  }, [user, customPortfolios]);

  const updatePortfolio = useCallback(async (id: string, name: string, tickers: string[]) => {
    if (!user) return;
    const updated = customPortfolios.map(p =>
      p.id === id
        ? { ...p, name: name.trim(), tickers: tickers.map(t => t.toUpperCase()).filter(Boolean) }
        : p
    );
    setCustomPortfolios(updated);
    await setDoc(
      doc(db, "users", user.uid),
      { portfolios: updated },
      { merge: true }
    );
  }, [user, customPortfolios]);

  const deletePortfolio = useCallback(async (id: string) => {
    if (!user) return;
    const updated = customPortfolios.filter(p => p.id !== id);
    setCustomPortfolios(updated);
    await setDoc(
      doc(db, "users", user.uid),
      { portfolios: updated },
      { merge: true }
    );
  }, [user, customPortfolios]);

  // ── Click tracking ─────────────────────────────────────────────────────────

  const trackClick = useCallback(async (symbol: string) => {
    if (!user) return;
    recordStockClick(user.uid, symbol);
    fetchOHLCV(symbol, "1M").catch(() => {});
  }, [user]);

  const logout = useCallback(async () => {
    await signOut(auth);
    setSavedStocks([]);
    setCustomPortfolios([]);
  }, []);

  const isSaved = useCallback(
    (symbol: string) => savedStocks.some(s => s.symbol === symbol),
    [savedStocks]
  );

  return (
    <AuthContext.Provider value={{
      user, loading,
      savedStocks, isSaved, toggleSave,
      trackClick, logout,
      customPortfolios, createPortfolio, updatePortfolio, deletePortfolio,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

// ── Background pre-warmer ─────────────────────────────────────────────────────

async function prewarmTopStocks(uid: string): Promise<void> {
  try {
    const topSymbols = await getTopClickedSymbols(uid, 5);
    await Promise.allSettled(topSymbols.map(s => fetchOHLCV(s, "1M")));
  } catch {
    // Non-critical
  }
}