import { useEffect, useMemo, useState, type ElementType } from "react";
import { Cpu, Database, HardDrive, Network, Server } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore } from "@/lib/store";

const iconByComponent: Record<string, ElementType> = {
  database: Database,
  redis: HardDrive,
  broker: Network,
  gpu: Cpu,
};

export default function OperationsPage() {
  const {
    hasPermission,
    mfaStatus,
    health,
    jobs,
    logs,
    auditTrail,
    siemStatus,
    sloStatus,
    incidents,
    incidentTimeline,
    runbooks,
    fetchHealth,
    fetchJobs,
    fetchLogs,
    fetchAuditTrail,
    fetchSiemStatus,
    fetchSloStatus,
    fetchIncidents,
    fetchIncidentTimeline,
    fetchRunbooks,
    createJob,
    executeRunbookAction,
    flushSiemQueue,
    exportAuditTrail,
  } = useStore();
  const [mfaCode, setMfaCode] = useState("");
  const canRunRunbooks = hasPermission("operations.runbooks.execute");
  const canCreateJobs = hasPermission("control.jobs.create");
  const canReadAudit = hasPermission("control.audit.read");
  const canManageAudit = hasPermission("control.audit.manage");
  const requiresMfa = Boolean(mfaStatus?.mfa_enabled);

  useEffect(() => {
    void Promise.all([fetchHealth(), fetchJobs(), fetchLogs(), fetchAuditTrail(), fetchSiemStatus(), fetchSloStatus(), fetchIncidents(), fetchIncidentTimeline(), fetchRunbooks()]);
    const timer = setInterval(
      () => void Promise.all([fetchHealth(), fetchJobs(), fetchLogs(), fetchAuditTrail(), fetchSiemStatus(), fetchSloStatus(), fetchIncidents(), fetchIncidentTimeline(), fetchRunbooks()]),
      10000,
    );
    return () => clearInterval(timer);
  }, [fetchHealth, fetchJobs, fetchLogs, fetchAuditTrail, fetchSiemStatus, fetchSloStatus, fetchIncidents, fetchIncidentTimeline, fetchRunbooks]);

  const recentLogs = useMemo(() => logs.slice(0, 18), [logs]);
  const recentJobs = useMemo(() => jobs.slice(0, 10), [jobs]);
  const recentAudit = useMemo(() => auditTrail.slice(0, 10), [auditTrail]);
  const recentIncidents = useMemo(() => incidents.slice(0, 8), [incidents]);
  const recentTimeline = useMemo(() => incidentTimeline.slice(0, 12), [incidentTimeline]);

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-slate-200 bg-white/90 p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">Infrastructure</p>
            <h1 className="mt-2 text-3xl font-bold text-slate-900">Operations Console</h1>
            <p className="mt-1 text-sm text-slate-600">Service health, command jobs, and runtime telemetry.</p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => void createJob("health", ["check", "--full"])} disabled={!canCreateJobs}>
              Full Health Check
            </Button>
            <Button variant="outline" onClick={() => void createJob("deploy", ["env", "check"])} disabled={!canCreateJobs}>
              Validate Env
            </Button>
            <Button variant="outline" onClick={() => void createJob("deploy", ["gpu", "check"])} disabled={!canCreateJobs}>
              GPU Check
            </Button>
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {health?.checks.map((check) => {
          const Icon = iconByComponent[check.component] ?? Server;
          return (
            <Card key={check.component} className="border-slate-200 bg-white/90">
              <CardHeader className="pb-2">
                <CardDescription className="capitalize">{check.component.replaceAll("_", " ")}</CardDescription>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Icon size={18} />
                  {check.status}
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-slate-600">
                <p>{check.message}</p>
                <p className="mt-1 text-xs">Latency: {check.latency_ms.toFixed(2)}ms</p>
              </CardContent>
            </Card>
          );
        })}
      </section>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>SLO Status</CardDescription>
            <CardTitle className="text-xl">{sloStatus?.status ?? "--"}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Availability</CardDescription>
            <CardTitle className="text-xl">{((sloStatus?.availability ?? 0) * 100).toFixed(2)}%</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Burn Rate 1h</CardDescription>
            <CardTitle className="text-xl">{(sloStatus?.burn_rate_1h ?? 0).toFixed(2)}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Error Budget Left</CardDescription>
            <CardTitle className="text-xl">{(sloStatus?.error_budget_remaining_pct ?? 0).toFixed(2)}%</CardTitle>
          </CardHeader>
        </Card>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>Command Job Queue</CardTitle>
            <CardDescription>Latest asynchronous operations launched from dashboard</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {recentJobs.length === 0 ? (
              <p className="text-sm text-slate-500">No jobs yet.</p>
            ) : (
              recentJobs.map((job) => (
                <div key={job.job_id} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div className="flex items-center justify-between">
                    <p className="font-medium text-slate-800">{job.command}</p>
                    <Badge variant={job.status === "succeeded" ? "success" : "outline"}>{job.status}</Badge>
                  </div>
                  <p className="mt-1 truncate font-mono text-xs text-slate-600">{job.args.join(" ")}</p>
                </div>
              ))
            )}
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>System Logs</CardTitle>
            <CardDescription>Recent logs streamed from backend state</CardDescription>
          </CardHeader>
          <CardContent className="max-h-[360px] space-y-2 overflow-auto">
            {recentLogs.length === 0 ? (
              <p className="text-sm text-slate-500">No logs available.</p>
            ) : (
              recentLogs.map((entry, idx) => (
                <div key={`${entry.timestamp}_${idx}`} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div className="flex items-center justify-between text-xs">
                    <Badge variant="outline">{entry.level}</Badge>
                    <span className="font-mono text-slate-500">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                  </div>
                  <p className="mt-2 text-sm text-slate-700">{entry.message}</p>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>Runbook Automation</CardTitle>
            <CardDescription>Execute operator-safe runbook actions</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {requiresMfa ? (
              <input
                className="h-10 w-full rounded-lg border border-slate-300 bg-white px-3 font-mono"
                value={mfaCode}
                onChange={(e) => setMfaCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                placeholder="MFA code"
              />
            ) : null}
            {runbooks.slice(0, 6).map((rb) => {
              const action = rb.alert_type === "HIGH_LATENCY"
                ? "health_check"
                : rb.alert_type === "BROKER_CONNECTION_LOST"
                  ? "broker_reconnect_dryrun"
                  : rb.alert_type === "DATA_FEED_FAILURE"
                    ? "data_feed_check"
                    : "validate_env";
              return (
                <Button
                  key={rb.alert_type}
                  variant="outline"
                  className="w-full justify-start"
                  onClick={() => void executeRunbookAction(action, mfaCode || undefined)}
                  disabled={!canRunRunbooks || (requiresMfa && mfaCode.length !== 6)}
                >
                  {rb.alert_type.replaceAll("_", " ")}
                </Button>
              );
            })}
            {!canRunRunbooks ? <p className="text-xs text-rose-700">Role lacks runbook execution permission.</p> : null}
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>Recent Incidents</CardTitle>
            <CardDescription>High-severity alerts and active incidents</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {recentIncidents.length === 0 ? (
              <p className="text-sm text-slate-500">No incidents.</p>
            ) : (
              recentIncidents.map((incident) => (
                <div key={incident.incident_id} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium text-slate-800">{incident.title}</p>
                    <Badge variant={incident.severity === "CRITICAL" ? "error" : "warning"}>{incident.severity}</Badge>
                  </div>
                  <p className="mt-1 text-xs text-slate-500">{new Date(incident.created_at).toLocaleString()}</p>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </section>

      <Card className="border-slate-200 bg-white/90">
        <CardHeader>
          <CardTitle>Signed Audit Trail</CardTitle>
          <CardDescription>Tamper-evident privileged action records</CardDescription>
        </CardHeader>
        <CardContent className="max-h-[300px] space-y-2 overflow-auto">
          {recentAudit.length === 0 ? (
            <p className="text-sm text-slate-500">No audit records.</p>
          ) : (
            recentAudit.map((entry) => (
              <div key={entry.record_hash} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div className="flex items-center justify-between text-xs">
                  <Badge variant="outline">{entry.status}</Badge>
                  <span className="font-mono text-slate-500">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                </div>
                <p className="mt-2 text-sm font-medium text-slate-800">{entry.action}</p>
                <p className="mt-1 text-xs font-mono text-slate-500">hash {entry.record_hash.slice(0, 14)}...</p>
              </div>
            ))
          )}
        </CardContent>
      </Card>

      {canReadAudit ? (
        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>SIEM Forwarding</CardTitle>
            <CardDescription>Centralized audit export and downstream SIEM delivery status.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid gap-2 text-sm text-slate-700 sm:grid-cols-2">
              <p>Enabled: <span className="font-semibold">{siemStatus?.enabled ? "yes" : "no"}</span></p>
              <p>Queue depth: <span className="font-semibold">{siemStatus?.queue_depth ?? 0}</span></p>
              <p>Delivered: <span className="font-semibold">{siemStatus?.total_delivered ?? 0}</span></p>
              <p>Failed: <span className="font-semibold">{siemStatus?.total_failed ?? 0}</span></p>
            </div>
            {siemStatus?.last_error ? (
              <p className="rounded-lg border border-rose-200 bg-rose-50 p-2 text-xs text-rose-700">{siemStatus.last_error}</p>
            ) : null}
            <div className="flex flex-wrap gap-2">
              <Button variant="outline" onClick={() => void exportAuditTrail("jsonl")}>
                Export NDJSON
              </Button>
              <Button variant="outline" onClick={() => void exportAuditTrail("json")}>
                Export JSON
              </Button>
              <Button variant="outline" onClick={() => void flushSiemQueue(5)} disabled={!canManageAudit}>
                Flush SIEM Queue
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : null}

      <Card className="border-slate-200 bg-white/90">
        <CardHeader>
          <CardTitle>Incident Timeline</CardTitle>
          <CardDescription>Correlated stream: alerts, audit and runtime logs</CardDescription>
        </CardHeader>
        <CardContent className="max-h-[300px] space-y-2 overflow-auto">
          {recentTimeline.length === 0 ? (
            <p className="text-sm text-slate-500">No incident timeline records.</p>
          ) : (
            recentTimeline.map((event, idx) => (
              <div key={`${event.timestamp}_${idx}`} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div className="flex items-center justify-between text-xs">
                  <Badge variant="outline">{event.source}</Badge>
                  <span className="font-mono text-slate-500">{new Date(event.timestamp).toLocaleTimeString()}</span>
                </div>
                <p className="mt-2 text-sm text-slate-700">{event.message}</p>
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}
