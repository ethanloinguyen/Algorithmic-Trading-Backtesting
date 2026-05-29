// src/app/auth/action/page.tsx
"use client";
import { Suspense, useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { applyActionCode } from "firebase/auth";
import { auth } from "@/src/app/lib/firebase";
import { TrendingUp, CircleCheck, XCircle, Loader2, MailCheck } from "lucide-react";

type Status = "idle" | "loading" | "success" | "error";

function ActionContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const mode = searchParams.get("mode");
  const oobCode = searchParams.get("oobCode") ?? "";
  const continueUrl = searchParams.get("continueUrl") ?? "/verify-email?verified=true";

  const [status, setStatus] = useState<Status>(() =>
    mode === "verifyEmail" && oobCode ? "idle" : "error"
  );
  const [errorMessage, setErrorMessage] = useState(
    mode !== "verifyEmail" || !oobCode ? "Invalid verification link." : ""
  );

  const handleVerify = async () => {
    setStatus("loading");
    try {
      await applyActionCode(auth, oobCode);
      await auth.currentUser?.reload();
      setStatus("success");
    } catch (err: unknown) {
      const code = (err as { code?: string })?.code;
      setStatus("error");
      if (code === "auth/expired-action-code") {
        setErrorMessage(
          "This verification link has expired. Sign in and request a new one."
        );
      } else if (code === "auth/invalid-action-code") {
        setErrorMessage(
          "This link has already been used or is no longer valid. If your email is already verified, sign in normally."
        );
      } else {
        setErrorMessage(
          "Verification failed. Please sign in and request a new verification email."
        );
      }
    }
  };

  useEffect(() => {
    if (status !== "success") return;
    router.push(continueUrl);
  }, [status, continueUrl, router]);

  return (
    <div
      className="min-h-screen flex items-center justify-center p-4"
      style={{ background: "hsl(213,27%,7%)" }}
    >
      <div
        className="w-full max-w-sm rounded-2xl p-8"
        style={{ background: "hsl(215,25%,13%)", border: "1px solid hsl(215,20%,20%)" }}
      >
        <div className="flex items-center justify-center gap-2 mb-6">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: "hsl(217,91%,60%)" }}
          >
            <TrendingUp className="w-4 h-4 text-white" strokeWidth={2.5} />
          </div>
          <span className="text-xl font-bold" style={{ color: "hsl(217,91%,60%)" }}>
            LagLens
          </span>
        </div>

        {status === "idle" && (
          <div className="text-center">
            <div
              className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-5"
              style={{ background: "hsla(217,91%,60%,0.15)" }}
            >
              <MailCheck className="w-8 h-8" style={{ color: "hsl(217,91%,60%)" }} />
            </div>
            <h2 className="text-lg font-bold mb-2" style={{ color: "hsl(210,40%,92%)" }}>
              Verify your email
            </h2>
            <p className="text-sm mb-8 leading-relaxed" style={{ color: "hsl(215,15%,60%)" }}>
              Click the button below to confirm your email address and activate your account.
            </p>
            <button
              onClick={handleVerify}
              className="w-full py-2.5 rounded-lg font-semibold text-sm text-white transition-all hover:opacity-90 active:scale-[0.98]"
              style={{ background: "hsl(217,91%,60%)" }}
            >
              Verify Email
            </button>
          </div>
        )}

        {status === "loading" && (
          <div className="text-center py-4">
            <Loader2
              className="w-8 h-8 animate-spin mx-auto mb-3"
              style={{ color: "hsl(217,91%,60%)" }}
            />
            <p className="text-sm" style={{ color: "hsl(215,15%,60%)" }}>
              Verifying your email…
            </p>
          </div>
        )}

        {status === "success" && (
          <div className="text-center">
            <div
              className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-5"
              style={{ background: "hsla(142,71%,45%,0.15)" }}
            >
              <CircleCheck className="w-8 h-8" style={{ color: "hsl(142,71%,55%)" }} />
            </div>
            <h2 className="text-lg font-bold mb-2" style={{ color: "hsl(210,40%,92%)" }}>
              Email verified!
            </h2>
            <p className="text-sm" style={{ color: "hsl(215,15%,60%)" }}>
              Redirecting you to sign in…
            </p>
          </div>
        )}

        {status === "error" && (
          <div className="text-center">
            <div
              className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-5"
              style={{ background: "hsla(0,84%,60%,0.15)" }}
            >
              <XCircle className="w-8 h-8" style={{ color: "hsl(0,84%,65%)" }} />
            </div>
            <h2 className="text-lg font-bold mb-2" style={{ color: "hsl(210,40%,92%)" }}>
              Link expired or already used
            </h2>
            <p
              className="text-sm mb-8 leading-relaxed"
              style={{ color: "hsl(215,15%,60%)" }}
            >
              {errorMessage}
            </p>
            <button
              onClick={() => router.push("/")}
              className="w-full py-2.5 rounded-lg font-semibold text-sm text-white transition-all hover:opacity-90 active:scale-[0.98]"
              style={{ background: "hsl(217,91%,60%)" }}
            >
              Sign In
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default function AuthActionPage() {
  return (
    <Suspense>
      <ActionContent />
    </Suspense>
  );
}
