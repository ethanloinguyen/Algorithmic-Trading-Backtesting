// Frontend/src/app/page.tsx
"use client"

import { Suspense } from "react";
import AuthForm from "@/components/ui/Authform";

export default function Page() {
  return (
    <Suspense>
      <AuthForm />
    </Suspense>
  );
}