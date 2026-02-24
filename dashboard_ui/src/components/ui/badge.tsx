import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-cyan-500/30 bg-cyan-500/15 text-cyan-300",
        secondary:
          "border-white/10 bg-white/[0.06] text-slate-300",
        destructive:
          "border-rose-500/30 bg-rose-500/15 text-rose-400",
        outline: "border-white/15 text-slate-300",
        success:
          "border-emerald-500/30 bg-emerald-500/15 text-emerald-400",
        warning:
          "border-amber-500/30 bg-amber-500/15 text-amber-400",
        error:
          "border-rose-500/30 bg-rose-500/15 text-rose-400",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge }
