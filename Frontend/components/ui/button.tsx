import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
  {
    variants: {
      variant: {
        default: "bg-blue-600 text-white hover:bg-blue-500 active:scale-[0.98]",
        destructive: "bg-red-600 text-white hover:bg-red-500",
        outline: "border border-[hsl(215,20%,22%)] bg-[hsl(215,25%,14%)] text-[hsl(210,40%,85%)] hover:bg-[hsl(215,25%,18%)]",
        secondary: "bg-[hsl(215,25%,16%)] text-[hsl(210,40%,85%)] hover:bg-[hsl(215,25%,20%)]",
        ghost: "text-[hsl(215,15%,60%)] hover:bg-[hsl(215,25%,14%)] hover:text-[hsl(210,40%,92%)]",
        link: "text-blue-500 underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-10 px-6",
        icon: "size-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant = "default",
  size = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "button"
  return (
    <Comp
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }