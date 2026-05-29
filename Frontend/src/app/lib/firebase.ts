// Frontend/src/app/lib/firebase.ts
"use client";
import { initializeApp, getApps, getApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore, initializeFirestore, persistentLocalCache } from "firebase/firestore";

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

const DB_ID = process.env.NEXT_PUBLIC_FIRESTORE_DATABASE_ID ?? "(default)";

// On first load, initializeApp + initializeFirestore with IndexedDB persistence.
// On HMR re-runs the app already exists, so we just call getFirestore().
const isFirstInit = !getApps().length;
const app = isFirstInit ? initializeApp(firebaseConfig) : getApp();

export const auth = getAuth(app);

function initDb() {
  if (!isFirstInit) return getFirestore(app, DB_ID);
  try {
    return initializeFirestore(app, { localCache: persistentLocalCache(), experimentalAutoDetectLongPolling: true }, DB_ID);
  } catch {
    // IndexedDB unavailable (incognito, storage quota) — fall back to memory cache
    return initializeFirestore(app, {}, DB_ID);
  }
}

export const db = initDb();
