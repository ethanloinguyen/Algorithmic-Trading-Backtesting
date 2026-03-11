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
  arrayRemove,
} from "firebase/firestore";

// ✅ Correct import path — firebase.ts lives at Frontend/src/app/lib/firebase.ts
import { auth, db } from "@/src/app/lib/firebase";
import { getTopClickedSymbols, recordStockClick } from "@/src/app/lib/stockCache";
import { fetchOHLCV } from "@/src/app/lib/api";
import { setCachedOHLCV, getCachedOHLCV } from "@/src/app/lib/stockCache";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SavedStock {
  symbol: string;
  name:   string;
}

interface AuthContextValue {
  user:           User | null;
  loading:        boolean;
  savedStocks:    SavedStock[];
  isSaved:        (symbol: string) => boolean;
  toggleSave:     (stock: SavedStock) => Promise<void>;
  trackClick:     (symbol: string) => Promise<void>;
  logout:         () => Promise<void>;
}

// ── Context ───────────────────────────────────────────────────────────────────

const AuthContext = createContext<AuthContextValue>({
  user:        null,
  loading:     true,
  savedStocks: [],
  isSaved:     () => false,
  toggleSave:  async () => {},
  trackClick:  async () => {},
  logout:      async () => {},
});

export function useAuth() {
  return useContext(AuthContext);
}

// ── Provider ──────────────────────────────────────────────────────────────────

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user,        setUser]        = useState<User | null>(null);
  const [loading,     setLoading]     = useState(true);
  const [savedStocks, setSavedStocks] = useState<SavedStock[]>([]);

  // Load saved stocks + pre-warm OHLCV cache for top-clicked stocks
  const loadUserData = useCallback(async (uid: string) => {
    const ref  = doc(db, "users", uid);
    const snap = await getDoc(ref);

    if (snap.exists()) {
      setSavedStocks((snap.data().savedStocks as SavedStock[]) ?? []);
    } else {
      await setDoc(ref, { savedStocks: [], clickedStocks: {} });
      setSavedStocks([]);
    }

    // Pre-warm Firestore cache for this user's top-clicked stocks (background)
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
      }
      setLoading(false);
    });
    return unsub;
  }, [loadUserData]);

  // Toggle save/unsave a stock in Firestore
  const toggleSave = useCallback(async (stock: SavedStock) => {
    if (!user) return;
    const ref          = doc(db, "users", user.uid);
    const alreadySaved = savedStocks.some(s => s.symbol === stock.symbol);

    if (alreadySaved) {
      const updated = savedStocks.filter(s => s.symbol !== stock.symbol);
      setSavedStocks(updated);
      // Use setDoc merge to guarantee removal even if arrayRemove object-match fails
      await setDoc(ref, { savedStocks: updated }, { merge: true });
    } else {
      setSavedStocks(prev => [...prev, stock]);
      await updateDoc(ref, { savedStocks: arrayUnion(stock) });
    }
  }, [user, savedStocks]);

  /**
   * Call this whenever a user clicks on a stock to open the modal.
   * It increments the click counter in Firestore and pre-warms the
   * 1M OHLCV cache so the next open is instant.
   */
  const trackClick = useCallback(async (symbol: string) => {
    if (!user) return;
    // Fire-and-forget — don't block the UI
    recordStockClick(user.uid, symbol).then(() => {
      // After recording, check if cache needs warming for this symbol
      getCachedOHLCV(symbol, "1M").then(cached => {
        if (!cached || cached.stale) {
          fetchOHLCV(symbol, "1M")
            .then(candles => setCachedOHLCV(symbol, "1M", candles))
            .catch(() => {}); // Non-critical
        }
      });
    });
  }, [user]);

  const logout = useCallback(async () => {
    await signOut(auth);
    setSavedStocks([]);
  }, []);

  const isSaved = useCallback(
    (symbol: string) => savedStocks.some(s => s.symbol === symbol),
    [savedStocks]
  );

  return (
    <AuthContext.Provider value={{ user, loading, savedStocks, isSaved, toggleSave, trackClick, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

// ── Background cache pre-warmer ───────────────────────────────────────────────

/**
 * Runs in the background after login.
 * Fetches the top-5 most-clicked stocks for this user and warms their
 * 1M OHLCV cache in Firestore so modals open instantly.
 */
async function prewarmTopStocks(uid: string): Promise<void> {
  try {
    const topSymbols = await getTopClickedSymbols(uid, 5);
    for (const symbol of topSymbols) {
      const cached = await getCachedOHLCV(symbol, "1M");
      if (!cached || cached.stale) {
        const candles = await fetchOHLCV(symbol, "1M");
        await setCachedOHLCV(symbol, "1M", candles);
      }
    }
  } catch {
    // Non-critical background task
  }
}