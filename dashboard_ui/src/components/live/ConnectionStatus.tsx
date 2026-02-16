import { motion, AnimatePresence } from "framer-motion";
import { Wifi, WifiOff } from "lucide-react";
import { useState } from "react";
import { useStore } from "@/lib/store";

const CHANNEL_LABELS: Record<string, string> = {
  portfolio: "Portfolio",
  orders: "Orders",
  signals: "Signals",
  alerts: "Alerts",
};

/** Compact connection-status badge for the sidebar / header. */
export default function ConnectionStatus() {
  const ws = useStore((s) => s.ws);
  const [expanded, setExpanded] = useState(false);
  const allConnected = Object.values(ws).every(Boolean);
  const anyConnected = Object.values(ws).some(Boolean);

  return (
    <div className="relative">
      <button
        onClick={() => setExpanded((p) => !p)}
        className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-[10px] font-medium tracking-wider uppercase transition-colors hover:bg-white/5"
      >
        {allConnected ? (
          <Wifi className="h-3 w-3 text-emerald-400" />
        ) : (
          <WifiOff className={`h-3 w-3 ${anyConnected ? "text-amber-400" : "text-rose-400"}`} />
        )}
        <span
          className={`${
            allConnected ? "text-emerald-400" : anyConnected ? "text-amber-400" : "text-rose-400"
          }`}
        >
          {allConnected ? "LIVE" : anyConnected ? "PARTIAL" : "OFFLINE"}
        </span>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            className="absolute bottom-full left-0 mb-2 w-44 rounded-xl border border-white/10 bg-slate-900/95 p-3 shadow-xl backdrop-blur-xl"
          >
            <p className="mb-2 text-[10px] font-medium uppercase tracking-widest text-slate-500">
              WebSocket Channels
            </p>
            {Object.entries(CHANNEL_LABELS).map(([key, label]) => (
              <div key={key} className="flex items-center justify-between py-1">
                <span className="text-xs text-slate-300">{label}</span>
                <span className="flex items-center gap-1">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      ws[key as keyof typeof ws]
                        ? "bg-emerald-400 shadow-[0_0_4px_theme(colors.emerald.400)]"
                        : "bg-rose-500"
                    }`}
                  />
                  <span className={`text-[10px] ${ws[key as keyof typeof ws] ? "text-emerald-400" : "text-rose-400"}`}>
                    {ws[key as keyof typeof ws] ? "OK" : "DOWN"}
                  </span>
                </span>
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
