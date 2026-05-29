// Frontend/components/ui/Authform.tsx
"use client";
import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  sendEmailVerification,
  updateProfile,
  signOut,
  AuthError,
} from "firebase/auth";

import { auth } from "@/src/app/lib/firebase";
import { Eye, EyeOff, TrendingUp, Loader2, CircleAlert, CircleCheck } from "lucide-react";

type Mode = "signin" | "register";

// ── Email validation ──────────────────────────────────────────────────────────

const LABEL_RE = /^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$/;
const TLD_RE   = /^[a-zA-Z]{2,10}$/;

function isValidEmail(raw: string): boolean {
  const email = raw.trim().toLowerCase();
  if (!email || email.length > 254) return false;
  if ((email.match(/@/g) ?? []).length !== 1) return false;

  const atIdx  = email.indexOf("@");
  const local  = email.slice(0, atIdx);
  const domain = email.slice(atIdx + 1);

  // Local part
  if (!local || local.length > 64) return false;
  if (!/^[a-zA-Z0-9.+_-]+$/.test(local)) return false;
  if (!/^[a-zA-Z0-9]/.test(local) || !/[a-zA-Z0-9]$/.test(local)) return false;
  if (local.includes("..")) return false;

  // Domain part
  if (!domain || domain.length > 253) return false;
  if (domain.startsWith(".") || domain.endsWith(".")) return false;
  if (domain.includes("..")) return false;

  const labels = domain.split(".");
  if (labels.length < 2) return false;
  for (const label of labels) {
    if (!label || label.length > 63 || !LABEL_RE.test(label)) return false;
  }

  const tld = labels[labels.length - 1];
  if (!TLD_RE.test(tld)) return false;

  const sld = labels[labels.length - 2];
  if (!sld) return false;

  return true;
}

// ── Firebase error messages ───────────────────────────────────────────────────

function friendlyError(err: AuthError): string {
  switch (err.code) {
    case "auth/invalid-email":        return "Please enter a valid email address.";
    case "auth/user-not-found":
    case "auth/wrong-password":
    case "auth/invalid-credential":   return "Incorrect email or password.";
    case "auth/email-already-in-use": return "An account with this email already exists.";
    case "auth/weak-password":        return "Password must be at least 6 characters.";
    case "auth/too-many-requests":    return "Too many attempts. Please try again later.";
    case "auth/unauthorized-domain":  return "Sign-in is not enabled for this domain. Contact support.";
    default:                          return "Something went wrong. Please try again.";
  }
}

// ── Cookie helper ─────────────────────────────────────────────────────────────

function setAuthCookie() {
  document.cookie = "ll_authed=1; path=/; SameSite=Strict; Max-Age=86400";
}

// ── Main component ────────────────────────────────────────────────────────────

export default function AuthForm() {
  const router       = useRouter();
  const searchParams = useSearchParams();
  const next         = searchParams.get("next") ?? "/dashboard";

  const [mode,          setMode]          = useState<Mode>("signin");
  const [displayName,   setDisplayName]   = useState("");
  const [email,         setEmail]         = useState("");
  const [emailTouched,  setEmailTouched]  = useState(false);
  const [pw,            setPw]            = useState("");
  const [showPassword,  setShowPassword]  = useState(false);
  const [loading,       setLoading]       = useState(false);
  const [error,         setError]         = useState("");

  // Inline email validation state
  const emailValid    = isValidEmail(email);
  const showEmailErr  = emailTouched && email.length > 0 && !emailValid;
  const showEmailOk   = emailTouched && email.length > 0 && emailValid;

  const emailBorder = showEmailErr
    ? "1px solid hsl(0,84%,55%)"
    : showEmailOk
    ? "1px solid hsl(142,71%,45%)"
    : "1px solid hsl(215,20%,22%)";

  const inputBase = {
    background: "hsl(215,25%,10%)",
    color:      "hsl(210,40%,92%)",
  };

  // ── Submit ───────────────────────────────────────────────────────────────────

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!isValidEmail(email)) {
      setEmailTouched(true);
      setError("Please enter a valid email address (e.g. name@example.com).");
      return;
    }

    setLoading(true);

    try {
      if (mode === "signin") {
        const { user } = await signInWithEmailAndPassword(auth, email, pw);
        if (!user.emailVerified) {
          try {
            await sendEmailVerification(user, {
              url: `${window.location.origin}/verify-email?verified=true`,
            });
          } catch {
            // Rate limit or domain error — redirect anyway, user can resend from verify page
          }
          await signOut(auth);
          router.push(`/verify-email?email=${encodeURIComponent(email)}`);
          return;
        }
        setAuthCookie();
        router.push(next);

      } else {
        // Register
        const { user } = await createUserWithEmailAndPassword(auth, email, pw);

        if (displayName.trim()) {
          await updateProfile(user, { displayName: displayName.trim() });
        }

        // Send verification email — user must verify before accessing dashboard
        try {
          await sendEmailVerification(user, {
            url: `${window.location.origin}/verify-email?verified=true`,
          });
        } catch {
          // Domain may not be authorized yet — user can resend from verify page
        }
        router.push(`/verify-email?email=${encodeURIComponent(email)}`);
      }
    } catch (err) {
      setError(friendlyError(err as AuthError));
    } finally {
      setLoading(false);
    }
  };

  const switchMode = () => {
    setMode(m => m === "signin" ? "register" : "signin");
    setError("");
    setEmailTouched(false);
  };

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen grid-bg flex items-center justify-center p-4">
      <div
        className="w-full max-w-sm rounded-2xl p-8"
        style={{ background: "hsl(215,25%,13%)", border: "1px solid hsl(215,20%,20%)" }}
      >
        {/* Logo — always shown */}
        <div className="flex items-center justify-center gap-2 mb-6">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: "hsl(217,91%,60%)" }}
          >
            <TrendingUp className="w-4 h-4 text-white" strokeWidth={2.5} />
          </div>
          <span className="text-xl font-bold" style={{ color: "hsl(217,91%,60%)" }}>LagLens</span>
        </div>

        <p className="text-center text-sm mb-7" style={{ color: "hsl(215,15%,60%)" }}>
          {mode === "signin" ? "Sign in to view dashboard" : "Create your account"}
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">

          {/* Name — register only */}
          {mode === "register" && (
            <div>
              <label className="block text-sm mb-1.5 font-medium" style={{ color: "hsl(215,15%,65%)" }}>
                Name
              </label>
              <input
                type="text"
                value={displayName}
                onChange={e => setDisplayName(e.target.value)}
                placeholder="Your name"
                className="w-full px-4 py-2.5 rounded-lg text-sm outline-none transition-all placeholder:text-[hsl(215,15%,35%)]"
                style={{ ...inputBase, border: "1px solid hsl(215,20%,22%)" }}
                onFocus={e => (e.currentTarget.style.borderColor = "hsl(217,91%,60%)")}
                onBlur={e  => (e.currentTarget.style.borderColor = "hsl(215,20%,22%)")}
              />
            </div>
          )}

          {/* Email */}
          <div>
            <label className="block text-sm mb-1.5 font-medium" style={{ color: "hsl(215,15%,65%)" }}>
              Email
            </label>
            <div className="relative">
              <input
                type="email"
                value={email}
                onChange={e => { setEmail(e.target.value); setError(""); }}
                onBlur={() => setEmailTouched(true)}
                required
                placeholder="name@example.com"
                className="w-full px-4 py-2.5 pr-9 rounded-lg text-sm outline-none transition-all placeholder:text-[hsl(215,15%,35%)]"
                style={{ ...inputBase, border: emailBorder }}
                onFocus={e => {
                  if (!showEmailErr && !showEmailOk) {
                    e.currentTarget.style.borderColor = "hsl(217,91%,60%)";
                  }
                }}
              />
              {showEmailOk && (
                <CircleCheck
                  className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4"
                  style={{ color: "hsl(142,71%,45%)" }}
                />
              )}
              {showEmailErr && (
                <CircleAlert
                  className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4"
                  style={{ color: "hsl(0,84%,60%)" }}
                />
              )}
            </div>
            {showEmailErr && (
              <p className="mt-1.5 text-xs" style={{ color: "hsl(0,84%,65%)" }}>
                Please enter a valid email — e.g. name@example.com
              </p>
            )}
          </div>

          {/* Password */}
          <div>
            <label className="block text-sm mb-1.5 font-medium" style={{ color: "hsl(215,15%,65%)" }}>
              Password
            </label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                value={pw}
                onChange={e => setPw(e.target.value)}
                required
                minLength={6}
                placeholder={mode === "register" ? "At least 6 characters" : "Your password"}
                className="w-full px-4 py-2.5 pr-11 rounded-lg text-sm outline-none transition-all placeholder:text-[hsl(215,15%,35%)]"
                style={{ ...inputBase, border: "1px solid hsl(215,20%,22%)" }}
                onFocus={e => (e.currentTarget.style.borderColor = "hsl(217,91%,60%)")}
                onBlur={e  => (e.currentTarget.style.borderColor = "hsl(215,20%,22%)")}
              />
              <button
                type="button"
                onClick={() => setShowPassword(s => !s)}
                className="absolute right-3 top-1/2 -translate-y-1/2"
                style={{ color: "hsl(215,15%,55%)" }}
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {/* Forgot password — sign-in only */}
          {mode === "signin" && (
            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => router.push("/forgot-password")}
                className="text-xs hover:underline"
                style={{ color: "hsl(215,15%,55%)" }}
              >
                Forgot Password
              </button>
            </div>
          )}

          {/* Error banner */}
          {error && (
            <p
              className="text-xs text-center py-2 px-3 rounded-lg flex items-center justify-center gap-1.5"
              style={{ background: "hsl(0,60%,20%)", color: "hsl(0,84%,75%)" }}
            >
              <CircleAlert className="w-3.5 h-3.5 flex-shrink-0" />
              {error}
            </p>
          )}

          {/* Submit */}
          <button
            type="submit"
            disabled={loading || (emailTouched && !emailValid && email.length > 0)}
            className="w-full py-2.5 rounded-lg font-semibold text-sm text-white transition-all hover:opacity-90 active:scale-[0.98] flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            style={{ background: "hsl(217,91%,60%)" }}
          >
            {loading && <Loader2 className="w-4 h-4 animate-spin" />}
            {mode === "signin" ? "Sign In" : "Create Account"}
          </button>
        </form>

        {/* Toggle mode */}
        <p className="mt-5 text-center text-sm" style={{ color: "hsl(215,15%,55%)" }}>
          {mode === "signin" ? "Don't have an account? " : "Already have an account? "}
          <button
            onClick={switchMode}
            className="font-medium hover:underline"
            style={{ color: "hsl(217,91%,60%)" }}
          >
            {mode === "signin" ? "Create Account" : "Sign In"}
          </button>
        </p>
      </div>
    </div>
  );
}
