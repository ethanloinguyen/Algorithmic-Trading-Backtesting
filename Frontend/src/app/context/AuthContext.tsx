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

export interface SavedPortfolio {
  name:    string;
  tickers: string[];
}

interface AuthContextValue {
  user:             User | null;
  loading:          boolean;
  savedStocks:      SavedStock[];
  savedPortfolios:  SavedPortfolio[];
  isSaved:          (symbol: string) => boolean;
  toggleSave:       (stock: SavedStock) => Promise<void>;
  savePortfolio:    (name: string, tickers: string[]) => Promise<void>;
  deletePortfolio:  (name: string) => Promise<void>;
  trackClick:       (symbol: string) => Promise<void>;
  logout:           () => Promise<void>;
}

// ── Context ───────────────────────────────────────────────────────────────────

const AuthContext = createContext<AuthContextValue>({
  user:             null,
  loading:          true,
  savedStocks:      [],
  savedPortfolios:  [],
  isSaved:          () => false,
  toggleSave:       async () => {},
  savePortfolio:    async () => {},
  deletePortfolio:  async () => {},
  trackClick:       async () => {},
  logout:           async () => {},
});

export function useAuth() {
  return useContext(AuthContext);
}

// ── Provider ──────────────────────────────────────────────────────────────────

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user,             setUser]             = useState<User | null>(null);
  const [loading,          setLoading]          = useState(true);
  const [savedStocks,      setSavedStocks]      = useState<SavedStock[]>([]);
  const [savedPortfolios,  setSavedPortfolios]  = useState<SavedPortfolio[]>([]);

  // Load saved stocks + portfolios from Firestore and pre-warm cache
  const loadUserData = useCallback(async (uid: string) => {
    const ref  = doc(db, "users", uid);
    const snap = await getDoc(ref);

    if (snap.exists()) {
      setSavedStocks(    (snap.data().savedStocks     as SavedStock[])     ?? []);
      setSavedPortfolios((snap.data().savedPortfolios as SavedPortfolio[]) ?? []);
    } else {
      // First login — create user document
      await setDoc(ref, { savedStocks: [], clickedStocks: {}, savedPortfolios: [] });
      setSavedStocks([]);
      setSavedPortfolios([]);
    }

    // Pre-warm: call the backend API for top-clicked stocks.
    // The backend will query BigQuery if needed and write results to Firestore,
    // so subsequent modal opens are instant reads from Firestore.
    prewarmTopStocks(uid);
  }, []);

  // Auth state listener
  useEffect(() => {
    const unsub = onAuthStateChanged(auth, async (firebaseUser) => {
      setUser(firebaseUser);
      if (firebaseUser) {
        await loadUserData(firebaseUser.uid);
      } else {
        setSavedStocks([]);
        setSavedPortfolios([]);
      }
      setLoading(false);
    });
    return unsub;
  }, [loadUserData]);

  // Toggle save/unsave — writes to Firestore users/{uid}/savedStocks
  const toggleSave = useCallback(async (stock: SavedStock) => {
    if (!user) return;
    const ref          = doc(db, "users", user.uid);
    const alreadySaved = savedStocks.some(s => s.symbol === stock.symbol);

    if (alreadySaved) {
      const updated = savedStocks.filter(s => s.symbol !== stock.symbol);
      setSavedStocks(updated);
      // Use setDoc merge to guarantee removal regardless of object shape
      await setDoc(ref, { savedStocks: updated }, { merge: true });
    } else {
      setSavedStocks(prev => [...prev, stock]);
      await updateDoc(ref, { savedStocks: arrayUnion(stock) });
    }
  }, [user, savedStocks]);

  /**
   * Upsert a named portfolio — creates it if the name is new, replaces it if
   * the name already exists. Writes the full updated array to Firestore.
   */
  const savePortfolio = useCallback(async (name: string, tickers: string[]) => {
    if (!user) return;
    const ref     = doc(db, "users", user.uid);
    const updated = [
      ...savedPortfolios.filter(p => p.name !== name),
      { name, tickers },
    ];
    setSavedPortfolios(updated);
    await setDoc(ref, { savedPortfolios: updated }, { merge: true });
  }, [user, savedPortfolios]);

  /**
   * Delete a named portfolio by name.
   */
  const deletePortfolio = useCallback(async (name: string) => {
    if (!user) return;
    const ref     = doc(db, "users", user.uid);
    const updated = savedPortfolios.filter(p => p.name !== name);
    setSavedPortfolios(updated);
    await setDoc(ref, { savedPortfolios: updated }, { merge: true });
  }, [user, savedPortfolios]);

  /**
   * Call when a user opens a stock modal.
   * Records the click in Firestore and triggers a background backend fetch
   * (which also warms the Firestore OHLCV cache) for next time.
   */
  const trackClick = useCallback(async (symbol: string) => {
    if (!user) return;
    // Record click count — fire and forget
    recordStockClick(user.uid, symbol);
    // Warm the backend cache by calling the API (backend writes to Firestore)
    fetchOHLCV(symbol, "1M").catch(() => {});
  }, [user]);

  const logout = useCallback(async () => {
    await signOut(auth);
    setSavedStocks([]);
    setSavedPortfolios([]);
  }, []);

  const isSaved = useCallback(
    (symbol: string) => savedStocks.some(s => s.symbol === symbol),
    [savedStocks]
  );

  return (
    <AuthContext.Provider value={{
      user, loading,
      savedStocks, savedPortfolios,
      isSaved, toggleSave,
      savePortfolio, deletePortfolio,
      trackClick, logout,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

// ── Background pre-warmer ─────────────────────────────────────────────────────

/**
 * After login, call the backend for the user's top 5 most-clicked stocks.
 * The backend checks its own Firestore cache first, only hitting BigQuery
 * if stale. Either way the Firestore cache is refreshed so the next modal
 * open is an instant Firestore read.
 */
async function prewarmTopStocks(uid: string): Promise<void> {
  try {
    const topSymbols = await getTopClickedSymbols(uid, 5);
    await Promise.allSettled(
      topSymbols.map(symbol => fetchOHLCV(symbol, "1M"))
    );
  } catch {
    // Non-critical background task
  }
}