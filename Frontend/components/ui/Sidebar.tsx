"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { TrendingUp, BarChart2, Grid, User, LogOut } from "lucide-react";

export default function Sidebar({ isOpen, onClose }: { isOpen?: boolean; onClose?: () => void }) {
  const pathname = usePathname();
  const router = useRouter();

  const handleLogout = () => {
    // Clear any auth state/tokens here when real auth is implemented
    router.push("/");
  };

  const navItems = [
    { href: "/dashboard", label: "Dashboard", icon: TrendingUp },
    { href: "/dashboard/analysis", label: "Analysis", icon: BarChart2 },
    { href: "/dashboard/heatmap", label: "Heatmap", icon: Grid },
  ];

  return (
    <nav
      className="fixed top-0 left-0 right-0 z-50 flex items-center px-6 h-14 gap-6"
      style={{
        background: "hsl(215, 28%, 9%)",
        borderBottom: "1px solid hsl(215, 20%, 16%)",
      }}
    >
      {/* Logo */}
      <Link href="/dashboard" className="flex items-center gap-2 mr-4">
        <div
          className="w-7 h-7 rounded-md flex items-center justify-center"
          style={{ background: "hsl(217, 91%, 60%)" }}
        >
          <TrendingUp className="w-3.5 h-3.5 text-white" strokeWidth={2.5} />
        </div>
        <span className="font-bold text-base" style={{ color: "hsl(217, 91%, 60%)" }}>
          LagLens
        </span>
      </Link>

      {/* Nav Items */}
      <div className="flex items-center gap-1">
        {navItems.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all"
              style={{
                background: active ? "hsl(217, 91%, 60%)" : "transparent",
                color: active ? "white" : "hsl(215, 15%, 60%)",
              }}
            >
              <Icon className="w-3.5 h-3.5" />
              {label}
            </Link>
          );
        })}
      </div>

      {/* Right side */}
      <div className="ml-auto flex items-center gap-3">
        <div
          className="flex items-center gap-2 px-3 py-1.5 rounded-full text-sm"
          style={{
            background: "hsl(215, 25%, 14%)",
            border: "1px solid hsl(215, 20%, 20%)",
            color: "hsl(215, 15%, 65%)",
          }}
        >
          <User className="w-3.5 h-3.5" />
          <span>Demo User</span>
        </div>
        <button
          onClick={handleLogout}
          className="p-1.5 rounded-md transition-colors hover:opacity-80"
          style={{ color: "hsl(215, 15%, 55%)" }}
          aria-label="Log out"
        >
          <LogOut className="w-4 h-4" />
        </button>
      </div>
    </nav>
  );
}