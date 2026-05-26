// frontend/src/app/reset-password/page.tsx
//
// Firebase password reset action handler. Firebase's reset email links to its
// hosted action page by default. To use this custom page instead, configure
// the action URL in Firebase Console:
//   Authentication → Templates → Password reset → Customize action URL
//   Set it to: https://your-domain.com/reset-password
//
// Firebase appends ?mode=resetPassword&oobCode=XXX&apiKey=YYY to the URL.
// This page reads the oobCode and calls confirmPasswordReset.
"use client";
import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Eye, EyeOff, TrendingUp } from "lucide-react";
import { confirmPasswordReset } from "firebase/auth";
import { FirebaseError } from "firebase/app";
import { auth } from "@/src/app/lib/firebase";

function ResetPasswordForm() {
  const searchParams = useSearchParams();
  const oobCode = searchParams.get("oobCode") ?? "";
  const router = useRouter();

  const [password, setPassword]         = useState("");
  const [confirm, setConfirm]           = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm]   = useState(false);
  const [error, setError]               = useState("");
  const [success, setSuccess]           = useState(false);
  const [loading, setLoading]           = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (password !== confirm) {
      setError("Passwords do not match.");
      return;
    }
    if (!oobCode) {
      setError("Invalid or expired reset link. Please request a new one.");
      return;
    }

    setLoading(true);
    try {
      await confirmPasswordReset(auth, oobCode, password);
      setSuccess(true);
    } catch (err: unknown) {
      if (err instanceof FirebaseError) {
        switch (err.code) {
          case "auth/expired-action-code":
            setError("This reset link has expired. Please request a new one.");
            break;
          case "auth/invalid-action-code":
            setError("This reset link is invalid or has already been used. Please request a new one.");
            break;
          case "auth/weak-password":
            setError("Password must be at least 6 characters.");
            break;
          default:
            setError(err.message.replace("Firebase: ", "").replace(/\(auth\/.*\)/, "").trim() || "Failed to reset password.");
        }
      } else {
        setError("Failed to reset password.");
      }
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div className="space-y-4">
        <p
          className="text-sm rounded-lg px-4 py-3 text-center leading-relaxed"
          style={{
            color: "hsl(142, 71%, 55%)",
            background: "hsl(142, 71%, 10%)",
            border: "1px solid hsl(142, 71%, 20%)",
          }}
        >
          Your password has been updated successfully.
        </p>
        <button
          onClick={() => router.push("/")}
          className="w-full py-2.5 rounded-lg font-semibold text-sm text-white transition-all hover:opacity-90 active:scale-[0.98]"
          style={{ background: "hsl(217, 91%, 60%)" }}
        >
          Sign In
        </button>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm mb-1.5 font-medium" style={{ color: "hsl(215, 15%, 65%)" }}>
          New Password
        </label>
        <div className="relative">
          <input
            type={showPassword ? "text" : "password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter new password"
            required
            minLength={6}
            className="w-full px-4 py-2.5 pr-11 rounded-lg text-sm transition-all outline-none"
            style={{ background: "hsl(215, 25%, 10%)", border: "1px solid hsl(215, 20%, 22%)", color: "hsl(210, 40%, 92%)" }}
            onFocus={(e) => (e.currentTarget.style.borderColor = "hsl(217, 91%, 60%)")}
            onBlur={(e)  => (e.currentTarget.style.borderColor = "hsl(215, 20%, 22%)")}
          />
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="absolute right-3 top-1/2 -translate-y-1/2 transition-colors"
            style={{ color: "hsl(215, 15%, 55%)" }}
            aria-label={showPassword ? "Hide password" : "Show password"}
          >
            {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
        </div>
      </div>

      <div>
        <label className="block text-sm mb-1.5 font-medium" style={{ color: "hsl(215, 15%, 65%)" }}>
          Confirm Password
        </label>
        <div className="relative">
          <input
            type={showConfirm ? "text" : "password"}
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            placeholder="Confirm new password"
            required
            minLength={6}
            className="w-full px-4 py-2.5 pr-11 rounded-lg text-sm transition-all outline-none"
            style={{ background: "hsl(215, 25%, 10%)", border: "1px solid hsl(215, 20%, 22%)", color: "hsl(210, 40%, 92%)" }}
            onFocus={(e) => (e.currentTarget.style.borderColor = "hsl(217, 91%, 60%)")}
            onBlur={(e)  => (e.currentTarget.style.borderColor = "hsl(215, 20%, 22%)")}
          />
          <button
            type="button"
            onClick={() => setShowConfirm(!showConfirm)}
            className="absolute right-3 top-1/2 -translate-y-1/2 transition-colors"
            style={{ color: "hsl(215, 15%, 55%)" }}
            aria-label={showConfirm ? "Hide password" : "Show password"}
          >
            {showConfirm ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {error && (
        <p
          className="text-xs rounded-lg px-3 py-2"
          style={{ color: "hsl(0, 84%, 65%)", background: "hsl(0, 84%, 10%)", border: "1px solid hsl(0, 84%, 20%)" }}
        >
          {error}
        </p>
      )}

      <button
        type="submit"
        disabled={loading}
        className="w-full py-2.5 rounded-lg font-semibold text-sm text-white transition-all hover:opacity-90 active:scale-[0.98] disabled:opacity-60"
        style={{ background: "hsl(217, 91%, 60%)" }}
      >
        {loading ? "Updating…" : "Update Password"}
      </button>
    </form>
  );
}

export default function ResetPasswordPage() {
  const router = useRouter();

  return (
    <div
      className="min-h-screen flex items-center justify-center p-4 relative overflow-hidden"
      style={{ background: "hsl(213, 27%, 7%)" }}
    >
      {/* Background grid */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(hsl(215, 20%, 14% / 0.6) 1px, transparent 1px),
            linear-gradient(90deg, hsl(215, 20%, 14% / 0.6) 1px, transparent 1px)
          `,
          backgroundSize: "48px 48px",
        }}
      />

      <div
        className="relative w-full max-w-sm rounded-2xl p-8"
        style={{
          background: "hsl(215, 25%, 13%)",
          border: "1px solid hsl(215, 20%, 20%)",
          boxShadow: "0 24px 60px hsl(213, 27%, 3% / 0.7)",
        }}
      >
        {/* Logo */}
        <div className="flex items-center justify-center gap-2 mb-2">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: "hsl(217, 91%, 60%)" }}
          >
            <TrendingUp className="w-4 h-4 text-white" strokeWidth={2.5} />
          </div>
          <span className="text-xl font-bold" style={{ color: "hsl(217, 91%, 60%)" }}>
            LagLens
          </span>
        </div>

        <p className="text-center text-sm mb-7" style={{ color: "hsl(215, 15%, 60%)" }}>
          Create a new password
        </p>

        <Suspense fallback={<p className="text-center text-sm" style={{ color: "hsl(215, 15%, 55%)" }}>Loading…</p>}>
          <ResetPasswordForm />
        </Suspense>

        <div className="mt-4">
          <button
            type="button"
            onClick={() => router.push("/forgot-password")}
            className="text-xs transition-colors hover:underline w-full text-center"
            style={{ color: "hsl(215, 15%, 55%)" }}
          >
            Request a new reset link
          </button>
        </div>
      </div>
    </div>
  );
}
