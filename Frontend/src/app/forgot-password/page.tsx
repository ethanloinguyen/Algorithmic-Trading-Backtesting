// frontend/src/app/forgot-password/page.tsx
"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { TrendingUp } from "lucide-react";
import { sendPasswordResetEmail } from "firebase/auth";
import { auth } from "@/src/app/lib/firebase";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [sent, setSent]   = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleReset = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await sendPasswordResetEmail(auth, email);
      setSent(true);
    } catch (err: unknown) {
      const msg = (err as { message?: string })?.message ?? "Failed to send reset email.";
      setError(msg.replace("Firebase: ", "").replace(/\(auth\/.*\)/, "").trim());
    } finally {
      setLoading(false);
    }
  };

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
          Reset your password
        </p>

        <form onSubmit={handleReset} className="space-y-4">
          <div>
            <label className="block text-sm mb-1.5 font-medium" style={{ color: "hsl(215, 15%, 65%)" }}>
              Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email address"
              required
              className="w-full px-4 py-2.5 rounded-lg text-sm transition-all outline-none"
              style={{ background: "hsl(215, 25%, 10%)", border: "1px solid hsl(215, 20%, 22%)", color: "hsl(210, 40%, 92%)" }}
              onFocus={(e) => (e.currentTarget.style.borderColor = "hsl(217, 91%, 60%)")}
              onBlur={(e)  => (e.currentTarget.style.borderColor = "hsl(215, 20%, 22%)")}
            />
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
            disabled={loading || sent}
            className="w-full py-2.5 rounded-lg font-semibold text-sm text-white transition-all hover:opacity-90 active:scale-[0.98] disabled:opacity-60"
            style={{ background: "hsl(217, 91%, 60%)" }}
          >
            {loading ? "Sending…" : "Reset Password"}
          </button>
        </form>

        <div className="mt-4">
          <button
            type="button"
            onClick={() => router.push("/")}
            className="text-xs transition-colors hover:underline w-full text-center"
            style={{ color: "hsl(215, 15%, 55%)" }}
          >
            Back to Log In
          </button>

          {sent && (
            <p
              className="mt-4 text-center text-sm rounded-lg px-4 py-3 leading-relaxed"
              style={{
                color: "hsl(142, 71%, 55%)",
                background: "hsl(142, 71%, 10%)",
                border: "1px solid hsl(142, 71%, 20%)",
              }}
            >
              An email has been sent to your address. Please click the link inside to reset your password.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}