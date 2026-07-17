/**
 * Tool-category → lucide-react icon mapping. We pick icons by *shape*
 * rather than color so the row stays legible without per-category tinting.
 */
import {
  ChevronRight,
  FileText,
  PenLine,
  Terminal,
  Search,
  List,
  Folder,
  Globe,
  Wrench,
  Bot,
  type LucideIcon,
} from "lucide-react";
import type { ToolCategory } from "./tools";

const CATEGORY_ICON: Record<ToolCategory, LucideIcon> = {
  read: FileText,
  write: PenLine,
  execute: Terminal,
  search: Search,
  list: List,
  directory: Folder,
  network: Globe,
  subagent: Bot,
  other: Wrench,
};

export function getToolIcon(category: ToolCategory): LucideIcon {
  return CATEGORY_ICON[category];
}

export function DisclosureChevron({
  open,
  size = 12,
  className,
}: {
  open: boolean;
  size?: number;
  className?: string;
}) {
  return (
    <ChevronRight
      size={size}
      strokeWidth={1.75}
      className={className}
      style={{
        flexShrink: 0,
        transform: open ? "rotate(90deg)" : "rotate(0deg)",
        transition: "transform 100ms",
      }}
    />
  );
}
