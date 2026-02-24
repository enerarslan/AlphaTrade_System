import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Bell, X, AlertTriangle, CheckCircle2, Info, Flame, Trash2, Filter } from "lucide-react";
import { useStore } from "@/lib/store";
import { useShallow } from "zustand/react/shallow";

type SeverityFilter = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "ALL";

const severityConfig: Record<string, { icon: typeof AlertTriangle; color: string; bg: string; border: string }> = {
  CRITICAL: { icon: Flame, color: "text-rose-400", bg: "bg-rose-500/10", border: "border-rose-500/20" },
  HIGH: { icon: AlertTriangle, color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20" },
  MEDIUM: { icon: Info, color: "text-cyan-400", bg: "bg-cyan-500/10", border: "border-cyan-500/20" },
  LOW: { icon: Info, color: "text-slate-400", bg: "bg-slate-500/10", border: "border-slate-500/20" },
  INFO: { icon: CheckCircle2, color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/20" },
};

export default function NotificationPanel() {
  const [isOpen, setIsOpen] = useState(false);
  const [filter, setFilter] = useState<SeverityFilter>("ALL");
  const { alerts, incidents } = useStore(useShallow((state) => ({ alerts: state.alerts, incidents: state.incidents })));

  const activeAlerts = useMemo(() => alerts.filter((a) => a.status !== "RESOLVED"), [alerts]);

  const filteredAlerts = useMemo(() => {
    if (filter === "ALL") return activeAlerts;
    return activeAlerts.filter((a) => a.severity === filter);
  }, [activeAlerts, filter]);

  const criticalCount = useMemo(
    () => activeAlerts.filter((a) => a.severity === "CRITICAL").length,
    [activeAlerts],
  );

  const totalCount = activeAlerts.length + incidents.length;

  return (
    <>
      {/* Bell Button */}
      <button
        onClick={() => setIsOpen(true)}
        className="relative rounded-lg border border-white/[0.08] bg-white/[0.04] p-2 text-slate-400 transition-all hover:border-cyan-500/30 hover:bg-white/[0.06] hover:text-slate-200"
      >
        <Bell size={16} />
        {totalCount > 0 && (
          <motion.span
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className={`absolute -right-1.5 -top-1.5 flex h-4 min-w-4 items-center justify-center rounded-full px-1 text-[9px] font-bold text-white ${
              criticalCount > 0 ? "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]" : "bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.3)]"
            }`}
          >
            {totalCount}
          </motion.span>
        )}
      </button>

      {/* Panel */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm"
              onClick={() => setIsOpen(false)}
            />

            {/* Slide-out Panel */}
            <motion.div
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "spring", damping: 30, stiffness: 300 }}
              className="fixed right-0 top-0 z-50 flex h-full w-full max-w-md flex-col border-l border-white/[0.08] bg-slate-950/95 shadow-2xl backdrop-blur-2xl"
            >
              {/* Header */}
              <div className="flex items-center justify-between border-b border-white/[0.08] px-5 py-4">
                <div>
                  <h2 className="text-sm font-bold text-slate-100">Notifications</h2>
                  <p className="text-[9px] text-slate-500">{totalCount} active · {criticalCount} critical</p>
                </div>
                <div className="flex items-center gap-2">
                  <button className="rounded-lg border border-white/[0.08] bg-white/[0.03] p-1.5 text-slate-500 transition-colors hover:text-slate-300">
                    <Trash2 size={14} />
                  </button>
                  <button
                    onClick={() => setIsOpen(false)}
                    className="rounded-lg border border-white/[0.08] bg-white/[0.03] p-1.5 text-slate-500 transition-colors hover:text-slate-300"
                  >
                    <X size={14} />
                  </button>
                </div>
              </div>

              {/* Filter Bar */}
              <div className="flex items-center gap-1.5 border-b border-white/[0.06] px-5 py-2">
                <Filter size={12} className="text-slate-500" />
                {(["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"] as const).map((sev) => (
                  <button
                    key={sev}
                    onClick={() => setFilter(sev)}
                    className={`rounded-md px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider transition-all ${
                      filter === sev
                        ? "bg-cyan-500/15 text-cyan-400 border border-cyan-500/30"
                        : "text-slate-500 hover:text-slate-300 border border-transparent"
                    }`}
                  >
                    {sev}
                  </button>
                ))}
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto px-5 py-3 space-y-2">
                {/* Incidents Section */}
                {incidents.length > 0 && filter === "ALL" && (
                  <>
                    <p className="text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500 mb-2">Active Incidents</p>
                    {incidents.slice(0, 5).map((inc) => {
                      const config = severityConfig[inc.severity] ?? severityConfig.MEDIUM;
                      const Icon = config.icon;
                      return (
                        <motion.div
                          key={inc.incident_id}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          className={`rounded-xl border ${config.border} ${config.bg} p-3`}
                        >
                          <div className="flex items-start gap-2">
                            <Icon size={14} className={`mt-0.5 ${config.color}`} />
                            <div className="flex-1 min-w-0">
                              <p className="text-xs font-semibold text-slate-200">{inc.title}</p>
                              {inc.suggested_action && (
                                <p className="mt-0.5 text-[10px] text-slate-400">{inc.suggested_action}</p>
                              )}
                              <p className="mt-1 text-[9px] text-slate-500">{new Date(inc.created_at).toLocaleString()}</p>
                            </div>
                          </div>
                        </motion.div>
                      );
                    })}
                  </>
                )}

                {/* Alerts Section */}
                <p className="text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500 mb-2 mt-3">Alerts</p>
                {filteredAlerts.length === 0 ? (
                  <div className="py-8 text-center">
                    <CheckCircle2 size={24} className="mx-auto text-emerald-500/40" />
                    <p className="mt-2 text-xs text-slate-500">No active alerts</p>
                  </div>
                ) : (
                  filteredAlerts.map((alert, i) => {
                    const config = severityConfig[alert.severity] ?? severityConfig.MEDIUM;
                    const Icon = config.icon;
                    return (
                      <motion.div
                        key={alert.alert_id ?? i}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.02 }}
                        className={`rounded-xl border ${config.border} ${config.bg} p-3 transition-all hover:scale-[1.01]`}
                      >
                        <div className="flex items-start gap-2">
                          <Icon size={14} className={`mt-0.5 shrink-0 ${config.color}`} />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-start justify-between gap-2">
                              <p className="text-xs font-semibold text-slate-200">{alert.title}</p>
                              <span className={`shrink-0 rounded-md px-1.5 py-0.5 text-[8px] font-bold uppercase ${config.bg} ${config.color} border ${config.border}`}>
                                {alert.severity}
                              </span>
                            </div>
                            <p className="mt-0.5 line-clamp-2 text-[10px] text-slate-400">{alert.message}</p>
                            <div className="mt-1 flex items-center gap-2 text-[9px] text-slate-500">
                              <span>{new Date(alert.timestamp).toLocaleTimeString()}</span>
                              <span>•</span>
                              <span className="font-mono text-cyan-400/70">{alert.alert_type}</span>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    );
                  })
                )}
              </div>

              {/* Footer */}
              <div className="border-t border-white/[0.06] px-5 py-3">
                <p className="text-center text-[9px] text-slate-500">
                  Showing {filteredAlerts.length} of {activeAlerts.length} alerts · {incidents.length} incidents
                </p>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}


