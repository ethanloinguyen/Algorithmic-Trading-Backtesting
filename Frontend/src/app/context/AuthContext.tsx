// frontend/src/app/context/AuthContext.tsx
"use client";
import {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  ReactNode,
} from "react";
import {
  onAuthStateChanged,
  signOut,
  User,
} from "firebase/auth";
import {
  doc,
  getDoc,
  setDoc,
  updateDoc,
  arrayUnion,
  arrayRemove,
} from "firebase/firestore";
import { auth, db } from "@/src/app/lib/firebase";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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
  logout:         () => Promise<void>;
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const AuthContext = createContext<AuthContextValue>({
  user:        null,
  loading:     true,
  savedStocks: [],
  isSaved:     () => false,
  toggleSave:  async () => {},
  logout:      async () => {},
});

export function useAuth() {
  return useContext(AuthContext);
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user,        setUser]        = useState<User | null>(null);
  const [loading,     setLoading]     = useState(true);
  const [savedStocks, setSavedStocks] = useState<SavedStock[]>([]);

  // ── Load saved stocks from Firestore whenever the user changes ──────────
  const loadSavedStocks = useCallback(async (uid: string) => {
    const ref  = doc(db, "users", uid);
    const snap = await getDoc(ref);
    if (snap.exists()) {
      setSavedStocks((snap.data().savedStocks as SavedStock[]) ?? []);
    } else {
      // First-time user — create their document with an empty watchlist
      await setDoc(ref, { savedStocks: [] });
      setSavedStocks([]);
    }
  }, []);

  // ── Listen to Firebase Auth state ───────────────────────────────────────
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      setUser(firebaseUser);
      if (firebaseUser) {
        await loadSavedStocks(firebaseUser.uid);
      } else {
        setSavedStocks([]);
      }
      setLoading(false);
    });
    return unsubscribe;
  }, [loadSavedStocks]);

  // ── Toggle a stock in/out of the saved list ─────────────────────────────
  const toggleSave = useCallback(async (stock: SavedStock) => {
    if (!user) return;

    const ref      = doc(db, "users", user.uid);
    const alreadySaved = savedStocks.some((s) => s.symbol === stock.symbol);

    if (alreadySaved) {
      // Remove — filter out any object with this symbol
      const updated = savedStocks.filter((s) => s.symbol !== stock.symbol);
      setSavedStocks(updated);
      await updateDoc(ref, { savedStocks: arrayRemove(stock) });
      // arrayRemove requires the exact object; re-write the full array to be safe
      await setDoc(ref, { savedStocks: updated }, { merge: true });
    } else {
      // Add
      setSavedStocks((prev) => [...prev, stock]);
      await updateDoc(ref, { savedStocks: arrayUnion(stock) });
    }
  }, [user, savedStocks]);

  // ── Logout ───────────────────────────────────────────────────────────────
  const logout = useCallback(async () => {
    await signOut(auth);
    setSavedStocks([]);
  }, []);

  const isSaved = useCallback(
    (symbol: string) => savedStocks.some((s) => s.symbol === symbol),
    [savedStocks]
  );

  return (
    <AuthContext.Provider value={{ user, loading, savedStocks, isSaved, toggleSave, logout }}>
      {children}
    </AuthContext.Provider>
  );
}