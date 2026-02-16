import { useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";

const PAGE_SHORTCUTS: Record<string, string> = {
  "1": "/",
  "2": "/trading",
  "3": "/platform",
  "4": "/risk",
  "5": "/models",
  "6": "/operations",
  "7": "/alerts",
  "8": "/settings",
};

/**
 * Global keyboard shortcut hook.
 * - Ctrl+1..8 → navigate pages
 * - Ctrl+K → command palette (handled by CommandPalette itself)
 * - ? → shows help (dispatches custom event)
 * - R → refresh page data
 */
export function useHotkeys(onRefresh?: () => void) {
  const navigate = useNavigate();

  const handler = useCallback(
    (e: KeyboardEvent) => {
      // Skip if user is typing in an input
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      // Ctrl+1..8 → page nav
      if (e.ctrlKey && PAGE_SHORTCUTS[e.key]) {
        e.preventDefault();
        navigate(PAGE_SHORTCUTS[e.key]);
        return;
      }

      // R → refresh
      if (e.key === "r" && !e.ctrlKey && !e.metaKey) {
        onRefresh?.();
        return;
      }

      // ? → hotkey help
      if (e.key === "?" && !e.ctrlKey) {
        window.dispatchEvent(new CustomEvent("show-hotkey-help"));
        return;
      }
    },
    [navigate, onRefresh],
  );

  useEffect(() => {
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [handler]);
}

export const HOTKEY_LIST = [
  { keys: "Ctrl + K", description: "Command Palette" },
  { keys: "Ctrl + 1-8", description: "Navigate pages" },
  { keys: "R", description: "Refresh data" },
  { keys: "?", description: "Show shortcuts" },
  { keys: "Esc", description: "Close dialogs" },
];
