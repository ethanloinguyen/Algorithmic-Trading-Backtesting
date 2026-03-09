// Frontend/components/ui/Authform.tsx
"use client";
import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  updateProfile,
  AuthError,
} from "firebase/auth";
import { doc, setDoc } from "firebase/firestore";
import { auth, db } from "@/src/app/lib/firebase";
import { Eye, EyeOff, TrendingUp, Loader2 } from "lucide-react";

type Mode = "signin" | "register";

// Set a lightweight auth cookie readable by Edge middleware
function setAuthCookie() {
  document.cookie = "ll_authed=1; path=/; SameSite=Strict; Max-Age=86400";
}

function friendlyError(err: AuthError): string {
  switch (err.code) {
    case "auth/invalid-email":        return "Please enter a valid email address.";
    case "auth/user-not-found":
    case "auth/wrong-password":
    case "auth/invalid-credential":   return "Incorrect email or password.";
    case "auth/email-already-in-use": return "An account with this email already exists.";
    case "auth/weak-password":        return "Password must be at least 6 characters.";
    case "auth/too-many-requests":    return "Too many attempts. Please try again later.";
    default:                          return "Something went wrong. Please try again.";
  }
}

export default function AuthForm() {
  const router       = useRouter();
  const searchParams = useSearchParams();
  const next         = searchParams.get("next") ?? "/dashboard";

  const [mode,         setMode]         = useState<Mode>("signin");
  const [displayName,  setDisplayName]  = useState("");
  const [email,        setEmail]        = useState("");
  const [password,     setPassword]     = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState("");

  const inputStyle = {
    background: "hsl(215, 25%, 10%)",
    border:     "1px solid hsl(215, 20%, 22%)",
    color:      "hsl(210, 40%, 92%)",
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      if (mode === "signin") {
        await signInWithEmailAndPassword(auth, email, password);
        setAuthCookie();
        router.push(next);
      } else {
        const { user } = await createUserWithEmailAndPassword(auth, email, password);

        if (displayName.trim()) {
          await updateProfile(user, { displayName: displayName.trim() });
        }

        // Create the Firestore user document with an empty watchlist
        await setDoc(doc(db, "users", user.uid), {
          displayName: displayName.trim() || email.split("@")[0],
          email:       user.email,
          savedStocks: [],
          createdAt:   new Date().toISOString(),
        });

        setAuthCookie();
        router.push("/dashboard");
      }
    } catch (err) {
      setError(friendlyError(err as AuthError));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen grid-bg flex items-center justify-center p-4">
      <div
        className="w-full max-w-sm rounded-2xl p-8"
        style={{ background: "hsl(215, 25%, 13%)", border: "1px solid hsl(215, 20%, 20%)" }}
      >
        {/* Logo */}
        <div className="flex items-center justify-center gap-2 mb-2">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: "hsl(217, 91%, 60%)" }}>
            <TrendingUp className="w-4 h-4 text-white" strokeWidth={2.5} />
          </div>
          <span className="text-xl font-bold" style={{ color: "hsl(217, 91%, 60%)" }}>LagLens</span>
        </div>

        <p className="text-center text-sm mb-7" style={{ color: "hsl(215, 15%, 60%)" }}>
          {mode === "signin" ? "Sign in to view dashboard" : "Create your account"}
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Name — register only */}
          {mode === "register" && (
            <div>
              <label className="block text-sm mb-1.5 font-medium" style={{ color: "hsl(215, 15%, 65%)" }}>
                Name
              </label>
              <input
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="Your name"
                className="w-full px-4 py-2.5 rounded-lg text-sm outline-none transition-all"
                style={inputStyle}
                onFocus={(e) => (e.currentTarget.style.borderColor = "hsl(217, 91%, 60%)")}
                onBlur={(e)  => (e.currentTarget.style.borderColor = "hsl(215, 20%, 22%)")}
              />
            </div>
          )}

          {/* Email */}
          <div>
            <label className="block text-sm mb-1.5 font-medium" style={{ color: "hsl(215, 15%, 65%)" }}>
              Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full px-4 py-2.5 rounded-lg text-sm outline-none transition-all"
              style={inputStyle}
              onFocus={(e) => (e.currentTarget.style.borderColor = "hsl(217, 91%, 60%)")}
              onBlur={(e)  => (e.currentTarget.style.borderColor = "hsl(215, 20%, 22%)")}
            />
          </div>

          {/* Password */}
          <div>
            <label className="block text-sm mb-1.5 font-medium" style={{ color: "hsl(215, 15%, 65%)" }}>
              Password
            </label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
                className="w-full px-4 py-2.5 pr-11 rounded-lg text-sm outline-none transition-all"
                style={inputStyle}
                onFocus={(e) => (e.currentTarget.style.borderColor = "hsl(217, 91%, 60%)")}
                onBlur={(e)  => (e.currentTarget.style.borderColor = "hsl(215, 20%, 22%)")}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2"
                style={{ color: "hsl(215, 15%, 55%)" }}
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {/* Forgot password — sign-in only */}
          {mode === "signin" && (
            <div className="flex justify-end">
              <button type="button" className="text-xs hover:underline" style={{ color: "hsl(215, 15%, 55%)" }}>
                Forgot Password
              </button>
            </div>
          )}

          {/* Error */}
          {error && (
            <p className="text-xs text-center py-2 px-3 rounded-lg"
              style={{ background: "hsl(0,60%,20%)", color: "hsl(0,84%,75%)" }}>
              {error}
            </p>
          )}

          {/* Submit */}
          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 rounded-lg font-semibold text-sm text-white transition-all hover:opacity-90 active:scale-[0.98] flex items-center justify-center gap-2 disabled:opacity-60"
            style={{ background: "hsl(217, 91%, 60%)" }}
          >
            {loading && <Loader2 className="w-4 h-4 animate-spin" />}
            {mode === "signin" ? "Sign In" : "Create Account"}
          </button>
        </form>

        {/* Toggle mode */}
        <p className="mt-5 text-center text-sm" style={{ color: "hsl(215, 15%, 55%)" }}>
          {mode === "signin" ? "Don't have an account? " : "Already have an account? "}
          <button
            onClick={() => { setMode(mode === "signin" ? "register" : "signin"); setError(""); }}
            className="font-medium hover:underline"
            style={{ color: "hsl(217, 91%, 60%)" }}
          >
            {mode === "signin" ? "Create Account" : "Sign In"}
          </button>
        </p>
      </div>
    </div>
  );
}