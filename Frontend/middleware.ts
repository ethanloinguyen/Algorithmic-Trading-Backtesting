// frontend/middleware.ts
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// ---------------------------------------------------------------------------
// Route protection middleware
//
// Protected paths: anything under /dashboard
// Public  paths:   / (login)  and  /register
//
// Firebase Auth stores its session token in a cookie named
// "__session" when using Firebase Hosting, or in IndexedDB when
// running client-side only.  Because middleware runs on the Edge
// (no access to IndexedDB), we use a lightweight custom cookie that
// we set manually after a successful login.
//
// How it works:
//   1. After signInWithEmailAndPassword succeeds in AuthForm, we call
//      document.cookie = "ll_authed=1; path=/; SameSite=Strict"
//   2. On logout we clear it:
//      document.cookie = "ll_authed=; Max-Age=0; path=/"
//   3. This middleware checks for that cookie on every request.
//
// For production you should replace this with a Firebase ID-token cookie
// verified via the Firebase Admin SDK in an API route.
// ---------------------------------------------------------------------------

const PROTECTED_PREFIX = "/dashboard";
const LOGIN_PATH       = "/";

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  const isProtected = pathname.startsWith(PROTECTED_PREFIX);
  const isAuthed    = request.cookies.has("ll_authed");

  if (isProtected && !isAuthed) {
    // Redirect to login, preserving the intended destination
    const loginUrl = request.nextUrl.clone();
    loginUrl.pathname = LOGIN_PATH;
    loginUrl.searchParams.set("next", pathname);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  // Run on all routes except Next.js internals and static files
  matcher: ["/((?!_next/static|_next/image|favicon.ico|.*\\.png$).*)"],
};