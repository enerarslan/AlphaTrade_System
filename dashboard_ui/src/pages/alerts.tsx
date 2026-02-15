import { useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Bell, CheckCircle, ShieldAlert, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore } from "@/lib/store";

const stagger = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.06 } },
};
const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: "easeOut" as const } },
} as const;

function severityColor(severity: string) {
  switch (severity) {
    case "CRITICAL":
      return "border-rose-500/20 bg-rose-500/[0.04] shadow-[0_0_15px_rgba(244,63,94,0.06)]";
    case "HIGH":
      return "border-amber-500/20 bg-amber-500/[0.04]";
    case "MEDIUM":
      return "border-yellow-500/15 bg-yellow-500/[0.03]";
    default:
      return "border-white/[0.08] bg-white/[0.03]";
  }
}

function severityBadgeVariant(severity: string) {
  switch (severity) {
    case "CRITICAL":
      return "error" as const;
    case "HIGH":
      return "warning" as const;
    case "MEDIUM":
      return "warning" as const;
    default:
      return "outline" as const;
  }
}

export default function AlertsPage() {
  const {
    alerts,
    user,
    hasPermission,
    fetchAlerts,
    acknowledgeAlert,
    resolveAlert,
  } = useStore();

  const canManageAlerts = hasPermission("control.alerts.manage");

  useEffect(() => {
    void fetchAlerts();
    const timer = setInterval(() => void fetchAlerts(), 10000);
    return () => clearInterval(timer);
  }, [fetchAlerts]);

  const activeAlerts = useMemo(
    () => alerts.filter((a) => a.status !== "RESOLVED"),
    [alerts],
  );
  const resolvedAlerts = useMemo(
    () => alerts.filter((a) => a.status === "RESOLVED").slice(0, 20),
    [alerts],
  );

  const criticalCount = useMemo(() => activeAlerts.filter((a) => a.severity === "CRITICAL").length, [activeAlerts]);

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-6">
      {/* Header */}
      <motion.section variants={fadeUp} className="rounded-2xl border border-amber-500/10 bg-amber-500/[0.02] p-6 backdrop-blur-sm">
        <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-amber-400/70">Alert Management</p>
        <h1 className="mt-2 text-3xl font-bold text-slate-100">Alert Center</h1>
        <p className="mt-1 text-sm text-slate-400">Active and historical alerts with severity classification.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          <Badge variant={criticalCount > 0 ? "error" : "success"}>
            <ShieldAlert size={12} className="mr-1" />
            {criticalCount} Critical
          </Badge>
          <Badge variant="warning">{activeAlerts.length} Active</Badge>
          <Badge variant="outline">{resolvedAlerts.length} Resolved</Badge>
        </div>
      </motion.section>

      {/* Active Alerts */}
      <motion.div variants={fadeUp}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bell size={18} className="text-amber-400" />
              Active Alerts
            </CardTitle>
            <CardDescription>{activeAlerts.length} unresolved alerts.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <AnimatePresence mode="popLayout">
              {activeAlerts.length === 0 ? (
                <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-2 text-sm text-emerald-400">
                  <CheckCircle size={16} /> All clear — no active alerts.
                </motion.p>
              ) : (
                activeAlerts.map((alert) => (
                  <motion.div
                    key={alert.alert_id}
                    layout
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    className={`rounded-xl border p-4 ${severityColor(alert.severity)}`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <Badge variant={severityBadgeVariant(alert.severity)}>{alert.severity}</Badge>
                          <h3 className="font-medium text-slate-200">{alert.title}</h3>
                        </div>
                        <p className="mt-1 text-sm text-slate-400">{alert.message}</p>
                        <div className="mt-2 flex items-center gap-3 text-xs text-slate-500">
                          <span>{new Date(alert.timestamp).toLocaleString()}</span>
                          <span>Status: {alert.status}</span>
                        </div>
                      </div>
                      {canManageAlerts && (
                        <div className="flex shrink-0 gap-1">
                          {alert.status === "ACTIVE" && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => void acknowledgeAlert(alert.alert_id, user?.username ?? "operator")}
                            >
                              Ack
                            </Button>
                          )}
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => void resolveAlert(alert.alert_id)}
                          >
                            <X size={14} />
                          </Button>
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))
              )}
            </AnimatePresence>
          </CardContent>
        </Card>
      </motion.div>

      {/* Resolved Alerts */}
      <motion.div variants={fadeUp}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle size={18} className="text-emerald-400" />
              Resolved Alerts
            </CardTitle>
            <CardDescription>Recently resolved alert history.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {resolvedAlerts.length === 0 ? (
              <p className="text-sm text-slate-500">No resolved alerts.</p>
            ) : (
              resolvedAlerts.map((alert) => (
                <div key={alert.alert_id} className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-4 py-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">{alert.severity}</Badge>
                      <span className="text-sm text-slate-300">{alert.title}</span>
                    </div>
                    <Badge variant="success">Resolved</Badge>
                  </div>
                  <p className="mt-1 text-xs text-slate-500">{alert.message}</p>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}
