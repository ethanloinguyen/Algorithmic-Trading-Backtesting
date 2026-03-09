// frontend/src/lib/firebase.ts
import { initializeApp, getApps, getApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

// ---------------------------------------------------------------------------
// Paste your Firebase project config values from:
// Firebase Console → Project Settings → Your apps → Web app → SDK setup
// Then add each value to your .env.local file (see below).
// ---------------------------------------------------------------------------
// .env.local entries needed:
//   NEXT_PUBLIC_FIREBASE_API_KEY=
//   NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=
//   NEXT_PUBLIC_FIREBASE_PROJECT_ID=
//   NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=
//   NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=
//   NEXT_PUBLIC_FIREBASE_APP_ID=
// ---------------------------------------------------------------------------

const firebaseConfig = {
  apiKey:            process.env.NEXT_PUBLIC_FIREBASE_API_KEY!,
  authDomain:        process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN!,
  projectId:         process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID!,
  storageBucket:     process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET!,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID!,
  appId:             process.env.NEXT_PUBLIC_FIREBASE_APP_ID!,
};

// Prevent re-initializing on hot reload in Next.js dev mode
const app  = getApps().length ? getApp() : initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const db   = getFirestore(app);