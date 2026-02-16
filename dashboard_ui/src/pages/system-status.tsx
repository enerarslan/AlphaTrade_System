import { useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  BookOpen,
  FileText,
  Gauge,
  HeartPulse,
  Play,
  ScrollText,
  Server,
  ShieldCheck,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore } from "@/lib/store";
import DataGrid from "@/components/ui/data-grid";
import { type ColumnDef } from "@tanstack/react-table";
import PanelLayout from "@/components/layout/PanelLayout";

const stagger = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } },
};
const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" as const } },
} as const;

function StatusDot({ ok }: { ok: boolean }) {
  return (
    <span className={`inline-block h-2.5 w-2.5 rounded-full ${ok ? "bg-emerald-400 shadow-[0_0_6px_rgba(16,185,129,0.5)]" : "bg-rose-400 shadow-[0_0_6px_rgba(244,63,94,0.5)]"}`} />
  );
}

/* ── DataGrid column definitions ── */

type AuditRow = {
  timestamp: string;
  action: string;
  user: string;
  status: string;
  details: Record<string, unknown>;
  prev_hash: string;
  record_hash: string;
};

const auditColumns: ColumnDef<AuditRow, unknown>[] = [
  {
    accessorKey: "timestamp",
    header: "Time",
    size: 130,
    cell: ({ getValue }) => new Date(getValue() as string).toLocaleString(),
  },
  { accessorKey: "action", header: "Action", size: 140 },
  { accessorKey: "user", header: "User", size: 100 },
  {
    accessorKey: "status",
    header: "Status",
    size: 80,
    cell: ({ getValue }) => {
      const v = getValue() as string;
      const variant = v === "SUCCESS" ? "success" : v === "FAILURE" ? "error" : "outline";
      return <Badge variant={variant}>{v}</Badge>;
    },
  },
  {
    accessorKey: "record_hash",
    header: "Hash",
    size: 100,
    cell: ({ getValue }) => (
      <span className="font-mono text-slate-500 text-[10px]">{(getValue() as string).slice(0, 12)}…</span>
    ),
  },
];

type LogRow = {
  timestamp: string;
  level: string;
  category: string;
  message: string;
};

const logColumns: ColumnDef<LogRow, unknown>[] = [
  {
    accessorKey: "timestamp",
    header: "Time",
    size: 100,
    cell: ({ getValue }) => new Date(getValue() as string).toLocaleTimeString(),
  },
  {
    accessorKey: "level",
    header: "Level",
    size: 70,
    cell: ({ getValue }) => {
      const v = getValue() as string;
      const color = v === "ERROR" ? "text-rose-400" : v === "WARNING" ? "text-amber-400" : "text-slate-400";
      return <span className={`font-mono text-xs ${color}`}>{v}</span>;
    },
  },
  { accessorKey: "category", header: "Category", size: 90 },
  { accessorKey: "message", header: "Message" },
];

export default function SystemStatusPage() {
  const {
    health,
    sloStatus,
    jobs,
    logs,
    auditTrail,
    siemStatus,
    incidents,
    runbooks,
    hasPermission,
    fetchHealth,
    fetchSloStatus,
    fetchJobs,
    fetchLogs,
    fetchAuditTrail,
    fetchSiemStatus,
    fetchIncidents,
    fetchIncidentTimeline,
    fetchRunbooks,
    cancelJob,
    executeRunbookAction,
    exportAuditTrail,
  } = useStore();

  const canReadAudit = hasPermission("control.audit.read");
  const canManageAudit = hasPermission("control.audit.manage");
  const canRunbook = hasPermission("operations.runbook.execute");

  useEffect(() => {
    void fetchHealth();
    void fetchSloStatus();
    void fetchJobs();
    void fetchLogs();
    void fetchIncidents();
    void fetchIncidentTimeline();
    void fetchRunbooks();
    if (canReadAudit) {
      void fetchAuditTrail();
      void fetchSiemStatus();
    }
    const timer = setInterval(() => {
      void fetchHealth();
      void fetchJobs();
      void fetchLogs();
    }, 10000);
    return () => clearInterval(timer);
  }, [fetchHealth, fetchSloStatus, fetchJobs, fetchLogs, fetchAuditTrail, fetchSiemStatus, fetchIncidents, fetchIncidentTimeline, fetchRunbooks, canReadAudit]);

  const healthChecks = useMemo(() => {
    if (!health) return [];
    return (health.checks ?? []).map((c) => ({
      name: c.component,
      status: c.status,
      ok: c.status === "HEALTHY",
    }));
  }, [health]);

  const recentJobs = useMemo(() => jobs.slice(0, 8), [jobs]);
  const recentIncidents = useMemo(() => incidents.slice(0, 8), [incidents]);

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-6">
      {/* Header */}
      <motion.section variants={fadeUp} className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-6 backdrop-blur-sm">
        <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-500/70">Site Reliability</p>
        <h1 className="mt-2 text-3xl font-bold text-slate-100">Operations Console</h1>
        <p className="mt-1 text-sm text-slate-400">Infrastructure health, SLOs, logs, audit, incidents, and runbook automation.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          <Badge variant={health?.status === "healthy" ? "success" : "error"}>
            <HeartPulse size={12} className="mr-1" />
            {health?.status ?? "--"}
          </Badge>
          <Badge variant="outline">SLO: {sloStatus?.status ?? "--"}</Badge>
        </div>
      </motion.section>

      {/* Health Checks */}
      <motion.div variants={fadeUp}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server size={18} className="text-cyan-400" />
              Health Checks
            </CardTitle>
            <CardDescription>Component-level health status.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              {healthChecks.length === 0 ? (
                <p className="text-sm text-slate-500 col-span-full">No health data.</p>
              ) : (
                healthChecks.map((c) => (
                  <div key={c.name} className={`flex items-center gap-3 rounded-xl border px-4 py-3 ${c.ok ? "border-emerald-500/15 bg-emerald-500/[0.03]" : "border-rose-500/15 bg-rose-500/[0.03]"}`}>
                    <StatusDot ok={c.ok} />
                    <div>
                      <p className="font-medium text-slate-200">{c.name}</p>
                      <p className="text-xs text-slate-500">{String(c.status)}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* SLO + Jobs — Resizable */}
      <motion.div variants={fadeUp}>
        <PanelLayout
          orientation="horizontal"
          storageKey="ops-slo-jobs"
          panels={[
            {
              id: "slo",
              defaultSize: 50,
              minSize: 30,
              children: (
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Gauge size={18} className="text-emerald-400" />
                      SLO Status
                    </CardTitle>
                    <CardDescription>Service-level objectives.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    {!sloStatus ? (
                      <p className="text-slate-500">No SLO data.</p>
                    ) : (
                      Object.entries(sloStatus).map(([key, val]) => (
                        <div key={key} className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
                          <span className="text-slate-400">{key.replaceAll("_", " ")}</span>
                          <span className="font-mono text-slate-200">{typeof val === "number" ? val.toFixed(3) : String(val)}</span>
                        </div>
                      ))
                    )}
                  </CardContent>
                </Card>
              ),
            },
            {
              id: "jobs",
              defaultSize: 50,
              minSize: 30,
              children: (
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Play size={18} className="text-cyan-400" />
                      Command Jobs
                    </CardTitle>
                    <CardDescription>Orchestrated job queue.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {recentJobs.length === 0 ? (
                      <p className="text-sm text-slate-500">No jobs.</p>
                    ) : (
                      recentJobs.map((job) => (
                        <div key={job.job_id} className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
                          <div>
                            <p className="font-medium text-slate-200">{job.command}</p>
                            <p className="truncate font-mono text-xs text-slate-500">{job.args.join(" ")}</p>
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant={job.status === "succeeded" ? "success" : job.status === "failed" ? "error" : "warning"}>{job.status}</Badge>
                            {(job.status === "queued" || job.status === "running") && (
                              <Button variant="ghost" size="sm" onClick={() => void cancelJob(job.job_id)}>Cancel</Button>
                            )}
                          </div>
                        </div>
                      ))
                    )}
                  </CardContent>
                </Card>
              ),
            },
          ]}
        />
      </motion.div>

      {/* Logs + Audit — Resizable with DataGrids */}
      <motion.div variants={fadeUp}>
        <PanelLayout
          orientation="horizontal"
          storageKey="ops-logs-audit"
          panels={[
            {
              id: "logs",
              defaultSize: canReadAudit ? 50 : 100,
              minSize: 30,
              children: (
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <FileText size={18} className="text-amber-400" />
                      System Logs
                    </CardTitle>
                    <CardDescription>Recent system log entries.</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {logs.length === 0 ? (
                      <p className="text-sm text-slate-500">No logs.</p>
                    ) : (
                      <DataGrid
                        data={logs}
                        columns={logColumns}
                        maxHeight={380}
                        exportFilename="system_logs"
                      />
                    )}
                  </CardContent>
                </Card>
              ),
            },
            ...(canReadAudit
              ? [
                  {
                    id: "audit",
                    defaultSize: 50,
                    minSize: 30,
                    children: (
                      <Card className="h-full">
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            <ScrollText size={18} className="text-cyan-400" />
                            Audit Trail
                          </CardTitle>
                          <CardDescription>Immutable audit log with integrity hashes.</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          {auditTrail.length === 0 ? (
                            <p className="text-sm text-slate-500">No audit entries.</p>
                          ) : (
                            <DataGrid
                              data={auditTrail}
                              columns={auditColumns}
                              maxHeight={340}
                              exportFilename="audit_trail"
                            />
                          )}
                          {canManageAudit && (
                            <Button variant="outline" size="sm" onClick={() => void exportAuditTrail("json")}>
                              Export Audit Trail
                            </Button>
                          )}
                        </CardContent>
                      </Card>
                    ),
                  },
                ]
              : []),
          ]}
        />
      </motion.div>

      {/* Incidents + SIEM + Runbooks */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle size={18} className="text-amber-400" />
              Incidents
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {recentIncidents.length === 0 ? (
              <p className="text-sm text-slate-500">No incidents.</p>
            ) : (
              recentIncidents.map((inc) => (
                <div key={inc.incident_id} className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-slate-200">{inc.title}</span>
                    <Badge variant={inc.severity === "CRITICAL" ? "error" : "warning"}>{inc.severity}</Badge>
                  </div>
                  <p className="text-xs text-slate-500">{new Date(inc.created_at).toLocaleString()}</p>
                </div>
              ))
            )}
          </CardContent>
        </Card>

        {canReadAudit && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ShieldCheck size={18} className="text-emerald-400" />
                SIEM Status
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-slate-300">
              <p>Enabled: <span className="font-semibold">{siemStatus?.enabled ? "yes" : "no"}</span></p>
              <p>Endpoint: <span className="font-mono text-xs text-slate-400">{siemStatus?.endpoint ?? "--"}</span></p>
              <p>Queue: <span className="font-semibold">{siemStatus?.queue_depth ?? 0}</span></p>
              {siemStatus?.last_error && <p className="text-xs text-rose-400">{siemStatus.last_error}</p>}
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BookOpen size={18} className="text-cyan-400" />
              Runbooks
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {runbooks.length === 0 ? (
              <p className="text-sm text-slate-500">No runbooks.</p>
            ) : (
              runbooks.map((rb) => (
                <div key={rb.alert_type} className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
                  <div>
                    <p className="text-sm font-medium text-slate-200">{rb.alert_type}</p>
                    <p className="text-xs text-slate-500">{rb.suggested_action ?? "No action specified"}</p>
                  </div>
                  <Button variant="outline" size="sm" disabled={!canRunbook} onClick={() => void executeRunbookAction(rb.alert_type)}>
                    Run
                  </Button>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}
