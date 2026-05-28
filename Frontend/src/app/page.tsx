// Frontend/src/app/page.tsx
"use client"

import { useEffect, Suspense } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/src/app/context/AuthContext";
import AuthForm  from "@/components/ui/Authform";

export default function Page() {
  return (
    <Suspense>
      <AuthForm />
    </Suspense>
  );
}