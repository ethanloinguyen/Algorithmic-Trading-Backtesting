// frontend/src/app/verify-email/page.tsx
"use client";
import { useState, useEffect, useRef, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { onAuthStateChanged, sendEmailVerification, User } from "firebase/auth";
import { doc, setDoc } from "firebase/firestore";
import { auth, db } from "@/src/app/lib/firebase";
import {
  TrendingUp,
  MailCheck,
  CircleCheck,
  Loader2,
} from "lucide-react";

function VerifyEmailContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const email = searchParams.get("email") ?? "";
  const isVerified = searchParams.get("verified") === "true";
  const initialSendFailed = searchParams.get("emailSent") === "false";

  const [currentUser, setCurrentUser] = useState<User | null | undefined>(undefined);
  const [resending, setResending] = useState(false);
  const [resent, setResent] = useState(false);
  const [resendError, setResendError] = useState("");
  const docWritten = useRef(false);

  useEffect(() => {
    return onAuthStateChanged(auth, (user) => setCurrentUser(user));
  }, []);

  useEffect(() => {
    if (!isVerified || !currentUser || docWritten.current) return;
    const writeUserDoc = async () => {
      // Reload to get the latest emailVerified status from Firebase
      await currentUser.reload();
      const freshUser = auth.currentUser;
      if (!freshUser?.emailVerified) return;
      docWritten.current = true;
      await setDoc(
        doc(db, "users", freshUser.uid),
        {
          displayName:   freshUser.displayName || freshUser.email?.split("@")[0] || "",
          email:         freshUser.email,
          savedStocks:   [],
          portfolios:    [],
          clickedStocks: {},
          createdAt:     new Date().toISOString(),
        },
        { merge: true }
      );
    };
    writeUserDoc();
  }, [isVerified, currentUser]);

  const handleResend = async () => {
    const user = currentUser ?? auth.currentUser;
    if (resending || resent || !user) return;
    setResending(true);
    setResendError("");
    try {
      await sendEmailVerification(user, {
        url: `${window.location.origin}/verify-email?verified=true`,
      });
      setResent(true);
    } catch (err: unknown) {
      const msg = (err as { message?: string })?.message ?? "Failed to send email.";
      setResendError(msg.replace("Firebase: ", "").replace(/\(auth\/.*\)/, "").trim());
    } finally {
      setResending(false);
    }
  };

  return (
    <div className="min-h-screen grid-bg flex items-center justify-center p-4">
      <div
        className="w-full max-w-sm rounded-2xl p-8"
        style={{ background: "hsl(215,25%,13%)", border: "1px solid hsl(215,20%,20%)" }}
      >
        {/* Logo */}
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

        {isVerified ? (
          /* ── Verified state ── */
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
            <p className="text-sm mb-8 leading-relaxed" style={{ color: "hsl(215,15%,60%)" }}>
              Your account is ready. Sign in with your new email to get started.
            </p>

            <button
              onClick={() => router.push("/")}
              className="w-full py-2.5 rounded-lg font-semibold text-sm text-white transition-all hover:opacity-90 active:scale-[0.98]"
              style={{ background: "hsl(217,91%,60%)" }}
            >
              Sign In
            </button>
          </div>
        ) : (
          /* ── Pending state ── */
          <div className="text-center">
            <div
              className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-5"
              style={{ background: "hsla(217,91%,60%,0.15)" }}
            >
              <MailCheck className="w-8 h-8" style={{ color: "hsl(217,91%,60%)" }} />
            </div>

            <h2 className="text-lg font-bold mb-2" style={{ color: "hsl(210,40%,92%)" }}>
              Check your email
            </h2>
            <p className="text-sm mb-1" style={{ color: "hsl(215,15%,60%)" }}>
              We sent a verification link to
            </p>
            <p className="text-sm font-semibold mb-6" style={{ color: "hsl(217,91%,70%)" }}>
              {email || "your email address"}
            </p>

            {initialSendFailed && (
              <p
                className="text-xs rounded-lg px-3 py-2 mb-4 leading-relaxed"
                style={{ color: "hsl(38,92%,65%)", background: "hsl(38,92%,10%)", border: "1px solid hsl(38,92%,25%)" }}
              >
                The verification email couldn&apos;t be sent automatically. Use the button below to try again.
              </p>
            )}

            <p className="text-xs mb-6 leading-relaxed" style={{ color: "hsl(215,15%,50%)" }}>
              Click the link in the email to verify your account, then come back and sign in.
              Check your spam folder if you don&apos;t see it — institutional emails (.edu, .org) are sometimes filtered.
            </p>

            {resendError && (
              <p
                className="text-xs rounded-lg px-3 py-2 mb-3 leading-relaxed"
                style={{ color: "hsl(0,84%,65%)", background: "hsl(0,84%,10%)", border: "1px solid hsl(0,84%,20%)" }}
              >
                {resendError}
              </p>
            )}

            {/* Resend */}
            <button
              onClick={handleResend}
              disabled={resending || resent || !(currentUser ?? auth.currentUser)}
              className="w-full py-2.5 rounded-lg text-sm font-semibold mb-3 transition-all disabled:opacity-60"
              style={{
                background: resent ? "hsla(142,71%,45%,0.15)" : "hsla(217,91%,60%,0.12)",
                border: `1px solid ${resent ? "hsla(142,71%,45%,0.4)" : "hsla(217,91%,60%,0.3)"}`,
                color: resent ? "hsl(142,71%,55%)" : "hsl(217,91%,70%)",
              }}
            >
              {resending ? (
                <span className="flex items-center justify-center gap-2">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" /> Sending…
                </span>
              ) : resent ? (
                <span className="flex items-center justify-center gap-2">
                  <CircleCheck className="w-3.5 h-3.5" /> Email resent
                </span>
              ) : (
                "Resend verification email"
              )}
            </button>

            <button
              onClick={() => router.push("/")}
              className="w-full py-2.5 rounded-lg text-sm font-medium transition-all hover:opacity-80"
              style={{
                background: "hsl(215,25%,16%)",
                border: "1px solid hsl(215,20%,22%)",
                color: "hsl(215,15%,65%)",
              }}
            >
              Back to sign in
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default function VerifyEmailPage() {
  return (
    <Suspense>
      <VerifyEmailContent />
    </Suspense>
  );
}
