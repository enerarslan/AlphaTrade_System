import { useEffect } from "react";
import { BellRing } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore } from "@/lib/store";

export default function AlertsPage() {
  const { user, alerts, hasPermission, fetchAlerts, acknowledgeAlert, resolveAlert } = useStore();
  const canManageAlerts = hasPermission("alerts.manage");

  useEffect(() => {
    void fetchAlerts();
    const timer = setInterval(() => void fetchAlerts(), 7000);
    return () => clearInterval(timer);
  }, [fetchAlerts]);

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-slate-200 bg-white/90 p-6">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Incident Management</p>
        <h1 className="mt-2 text-3xl font-bold text-slate-900">Alert Center</h1>
        <p className="mt-1 text-sm text-slate-600">Acknowledge and resolve incidents with explicit operator actions.</p>
      </section>

      <Card className="border-slate-200 bg-white/90">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BellRing size={18} />
            Active and Historical Alerts
          </CardTitle>
          <CardDescription>Total {alerts.length} alerts</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {alerts.length === 0 ? (
            <p className="text-sm text-slate-500">No alerts available.</p>
          ) : (
            alerts.map((alert) => (
              <div key={alert.alert_id} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <p className="font-semibold text-slate-900">{alert.title}</p>
                    <p className="text-sm text-slate-700">{alert.message}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={alert.severity === "CRITICAL" ? "error" : alert.severity === "HIGH" ? "warning" : "outline"}>
                      {alert.severity}
                    </Badge>
                    <Badge variant="outline">{alert.status}</Badge>
                  </div>
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => void acknowledgeAlert(alert.alert_id, user?.username ?? "operator")}
                    disabled={!canManageAlerts || alert.status !== "FIRING"}
                  >
                    Acknowledge
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => void resolveAlert(alert.alert_id)} disabled={!canManageAlerts || alert.status === "RESOLVED"}>
                    Resolve
                  </Button>
                  <span className="text-xs text-slate-500">{new Date(alert.timestamp).toLocaleString()}</span>
                </div>
              </div>
            ))
          )}
          {!canManageAlerts ? <p className="text-xs text-rose-700">Read-only role: alert actions are disabled.</p> : null}
        </CardContent>
      </Card>
    </div>
  );
}
