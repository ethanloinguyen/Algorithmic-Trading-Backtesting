import * as React from "react"
import { cn } from "@/lib/utils"

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "h-9 w-full min-w-0 rounded-lg border px-3 py-1 text-sm outline-none transition-all",
        "bg-[hsl(215,25%,10%)] border-[hsl(215,20%,22%)] text-[hsl(210,40%,92%)] placeholder:text-[hsl(215,15%,40%)]",
        "focus-visible:border-blue-500 focus-visible:ring-2 focus-visible:ring-blue-500/20",
        "disabled:pointer-events-none disabled:opacity-50",
        className
      )}
      {...props}
    />
  )
}

export { Input }