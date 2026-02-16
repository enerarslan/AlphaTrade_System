import { useEffect, useRef } from "react";
import { useStore } from "@/lib/store";

type DataSlice =
  | "portfolio"
  | "positions"
  | "orders"
  | "signals"
  | "performance"
  | "health"
  | "alerts"
  | "riskMetrics"
  | "tradingStatus";

const FETCH_MAP: Record<DataSlice, (state: ReturnType<typeof useStore.getState>) => () => Promise<void>> = {
  portfolio: (s) => s.fetchPortfolio,
  positions: (s) => s.fetchPositions,
  orders: (s) => s.fetchOrders,
  signals: (s) => s.fetchSignals,
  performance: (s) => s.fetchPerformance,
  health: (s) => s.fetchHealth,
  alerts: (s) => s.fetchAlerts,
  riskMetrics: (s) => s.fetchRiskMetrics,
  tradingStatus: (s) => s.fetchTradingStatus,
};

/**
 * Auto-refresh hook that fetches data on mount and then periodically.
 * - Pauses when the browser tab is hidden (Page Visibility API).
 * - Meant to complement WebSocket live channels, not replace them.
 */
export function useAutoRefresh(slices: DataSlice[], intervalMs = 5000) {
  const visibleRef = useRef(true);

  useEffect(() => {
    const state = useStore.getState();

    // Initial fetch
    for (const slice of slices) {
      const fetcher = FETCH_MAP[slice];
      if (fetcher) void fetcher(state)();
    }

    // Periodic refresh
    const timer = setInterval(() => {
      if (!visibleRef.current) return;
      const s = useStore.getState();
      for (const slice of slices) {
        const fetcher = FETCH_MAP[slice];
        if (fetcher) void fetcher(s)();
      }
    }, intervalMs);

    // Page Visibility
    const handleVisibility = () => {
      visibleRef.current = document.visibilityState === "visible";
    };
    document.addEventListener("visibilitychange", handleVisibility);

    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, [slices.join(","), intervalMs]); // eslint-disable-line react-hooks/exhaustive-deps
}
