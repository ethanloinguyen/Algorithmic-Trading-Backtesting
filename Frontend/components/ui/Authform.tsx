"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Eye, EyeOff, TrendingUp } from "lucide-react";

export default function AuthForm() {
  const [email, setEmail] = useState("demo@gmail.com");
  const [password, setPassword] = useState("••••••••••");
  const [showPassword, setShowPassword] = useState(false);
  const router = useRouter();

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    router.push("/dashboard");
  };

  return (
    <div className="min-h-screen grid-bg flex items-center justify-center p-4">
      <div
        className="w-full max-w-sm rounded-2xl p-8"
        style={{ background: "hsl(215, 25%, 13%)", border: "1px solid hsl(215, 20%, 20%)" }}
      >
        {/* Logo */}
        <div className="flex items-center justify-center gap-2 mb-2">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: "hsl(217, 91%, 60%)" }}>
            <TrendingUp className="w-4 h-4 text-white" strokeWidth={2.5} />
          </div>
          <span className="text-xl font-bold" style={{ color: "hsl(217, 91%, 60%)" }}>LagLens</span>
        </div>

        <p className="text-center text-sm mb-7" style={{ color: "hsl(215, 15%, 60%)" }}>
          Sign in to view dashboard
        </p>

        <form onSubmit={submit} className="space-y-4">
          <div>
            <label
              className="block text-sm mb-1.5 font-medium"
              style={{ color: "hsl(215, 15%, 65%)" }}
            >
              Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-2.5 rounded-lg text-sm transition-all outline-none focus:ring-2"
              style={{
                background: "hsl(215, 25%, 10%)",
                border: "1px solid hsl(215, 20%, 22%)",
                color: "hsl(210, 40%, 92%)",
                focusRingColor: "hsl(217, 91%, 60%)",
              }}
              onFocus={e => e.currentTarget.style.borderColor = "hsl(217, 91%, 60%)"}
              onBlur={e => e.currentTarget.style.borderColor = "hsl(215, 20%, 22%)"}
            />
          </div>

          <div>
            <label
              className="block text-sm mb-1.5 font-medium"
              style={{ color: "hsl(215, 15%, 65%)" }}
            >
              Password
            </label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-2.5 pr-11 rounded-lg text-sm transition-all outline-none"
                style={{
                  background: "hsl(215, 25%, 10%)",
                  border: "1px solid hsl(215, 20%, 22%)",
                  color: "hsl(210, 40%, 92%)",
                }}
                onFocus={e => e.currentTarget.style.borderColor = "hsl(217, 91%, 60%)"}
                onBlur={e => e.currentTarget.style.borderColor = "hsl(215, 20%, 22%)"}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 transition-colors"
                style={{ color: "hsl(215, 15%, 55%)" }}
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div className="flex justify-end">
            <button
              type="button"
              className="text-xs transition-colors hover:underline"
              style={{ color: "hsl(215, 15%, 55%)" }}
            >
              Forgot Password
            </button>
          </div>

          <button
            type="submit"
            className="w-full py-2.5 rounded-lg font-semibold text-sm text-white transition-all hover:opacity-90 active:scale-[0.98]"
            style={{ background: "hsl(217, 91%, 60%)" }}
          >
            Sign In
          </button>
        </form>

        <p className="mt-5 text-center text-sm" style={{ color: "hsl(215, 15%, 55%)" }}>
          Don&apos;t have an account?{" "}
          <button
            className="font-medium transition-colors hover:underline"
            style={{ color: "hsl(217, 91%, 60%)" }}
          >
            Create Account
          </button>
        </p>
      </div>
    </div>
  );
}