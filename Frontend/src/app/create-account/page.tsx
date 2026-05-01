// frontend/src/app/create-account/page.tsx
"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Eye, EyeOff, TrendingUp } from "lucide-react";
import { createUserWithEmailAndPassword, updateProfile } from "firebase/auth";
import { doc, setDoc } from "firebase/firestore";
import { auth, db } from "@/src/app/lib/firebase";

export default function CreateAccountPage() {
  const [name, setName]             = useState("");
  const [email, setEmail]           = useState("");
  const [password, setPassword]     = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError]           = useState("");
  const [loading, setLoading]       = useState(false);
  const router = useRouter();

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const cred = await createUserWithEmailAndPassword(auth, email, password);
      await updateProfile(cred.user, { displayName: name });
      // Create Firestore user doc
      await setDoc(doc(db, "users", cred.user.uid), {
        savedStocks:     [],
        clickedStocks:   {},
        savedPortfolios: [],
      });
      document.cookie = "ll_authed=1; path=/; max-age=86400";
      router.push("/dashboard");
    } catch (err: unknown) {
      const msg = (err as { message?: string })?.message ?? "Account creation failed.";
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
          Create your account
        </p>

        <form onSubmit={submit} className="space-y-4">
          <div>
            <label className="block text-sm mb-1.5 font-medium" style={{ color: "hsl(215, 15%, 65%)" }}>
              Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Your full name"
              required
              className="w-full px-4 py-2.5 rounded-lg text-sm transition-all outline-none"
              style={{ background: "hsl(215, 25%, 10%)", border: "1px solid hsl(215, 20%, 22%)", color: "hsl(210, 40%, 92%)" }}
              onFocus={(e) => (e.currentTarget.style.borderColor = "hsl(217, 91%, 60%)")}
              onBlur={(e)  => (e.currentTarget.style.borderColor = "hsl(215, 20%, 22%)")}
            />
          </div>

          <div>
            <label className="block text-sm mb-1.5 font-medium" style={{ color: "hsl(215, 15%, 65%)" }}>
              Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Your email address"
              required
              className="w-full px-4 py-2.5 rounded-lg text-sm transition-all outline-none"
              style={{ background: "hsl(215, 25%, 10%)", border: "1px solid hsl(215, 20%, 22%)", color: "hsl(210, 40%, 92%)" }}
              onFocus={(e) => (e.currentTarget.style.borderColor = "hsl(217, 91%, 60%)")}
              onBlur={(e)  => (e.currentTarget.style.borderColor = "hsl(215, 20%, 22%)")}
            />
          </div>

          <div>
            <label className="block text-sm mb-1.5 font-medium" style={{ color: "hsl(215, 15%, 65%)" }}>
              Password
            </label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Create a password"
                required
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
            {loading ? "Creating account…" : "Create Account"}
          </button>
        </form>

        <p className="mt-5 text-center text-sm" style={{ color: "hsl(215, 15%, 55%)" }}>
          Already have an account?{" "}
          <button
            onClick={() => router.push("/")}
            className="font-medium transition-colors hover:underline"
            style={{ color: "hsl(217, 91%, 60%)" }}
          >
            Sign In
          </button>
        </p>
      </div>
    </div>
  );
}