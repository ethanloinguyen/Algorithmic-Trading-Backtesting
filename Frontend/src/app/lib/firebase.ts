// Frontend/src/app/lib/firebase.ts
"use client";
import { initializeApp, getApps, getApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore, enableIndexedDbPersistence } from "firebase/firestore";

// .env.local entries required:
//   NEXT_PUBLIC_FIREBASE_API_KEY=
//   NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=
//   NEXT_PUBLIC_FIREBASE_PROJECT_ID=
//   NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=
//   NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=
//   NEXT_PUBLIC_FIREBASE_APP_ID=

const firebaseConfig = {
  apiKey:            process.env.NEXT_PUBLIC_FIREBASE_API_KEY!,
  authDomain:        process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN!,
  projectId:         process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID!,
  storageBucket:     process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET!,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID!,
  appId:             process.env.NEXT_PUBLIC_FIREBASE_APP_ID!,
};

const app = getApps().length ? getApp() : initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const db   = getFirestore(app);

// Enable offline persistence (IndexedDB) so cached data survives page refreshes.
// This is what makes Firestore act as a local cache — reads are served from
// IndexedDB when online data hasn't changed, and the app works offline too.
// Wrapped in try/catch because it throws if called more than once (HMR).
if (typeof window !== "undefined") {
  enableIndexedDbPersistence(db).catch((err) => {
    if (err.code === "failed-precondition") {
      // Multiple tabs open — persistence only works in one tab at a time.
      console.warn("Firestore persistence unavailable: multiple tabs open.");
    } else if (err.code === "unimplemented") {
      // Browser doesn't support IndexedDB.
      console.warn("Firestore persistence not supported in this browser.");
    }
  });
}