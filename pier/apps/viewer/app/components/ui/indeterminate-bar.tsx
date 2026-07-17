import type { CSSProperties } from "react";

import { cn } from "~/lib/utils";

/**
 * Thin indeterminate progress bar. Absolutely positioned over its
 * `position: relative` parent — set `top`/etc. via `className` or `style`.
 *
 * Visually 2px thick during animation but takes up no layout space; pair it
 * with an existing 1px border so the line at rest stays exactly 1px.
 */
export function IndeterminateBar({
  className,
  style,
  label = "Loading",
}: {
  className?: string;
  style?: CSSProperties;
  label?: string;
}) {
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-busy="true"
      className={cn(
        "pointer-events-none absolute inset-x-0 z-50 h-0.5 overflow-hidden",
        className
      )}
      style={style}
    >
      <div className="absolute inset-y-0 w-1/3 bg-foreground/70 animate-indeterminate-bar" />
    </div>
  );
}
