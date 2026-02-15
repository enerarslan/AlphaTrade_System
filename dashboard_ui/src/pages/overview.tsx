import { useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import { Activity, AlertTriangle, Brain, BriefcaseBusiness, ShieldAlert, Workflow } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SignalTape } from "@/components/live/SignalTape";
import { useStore } from "@/lib/store";

const stagger = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08 },
  },
};

const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" as const } },
} as const;

function GlowMetric({ label, value, accent = "cyan" }: { label: string; value: string; accent?: "cyan" | "emerald" | "rose" | "amber" }) {
  const colors = {
    cyan: "border-cyan-500/20 shadow-[0_0_15px_rgba(6,182,212,0.08)]",
    emerald: "border-emerald-500/20 shadow-[0_0_15px_rgba(16,185,129,0.08)]",
    rose: "border-rose-500/20 shadow-[0_0_15px_rgba(244,63,94,0.08)]",
    amber: "border-amber-500/20 shadow-[0_0_15px_rgba(245,158,11,0.08)]",
  };
  const textColors = {
    cyan: "text-cyan-300",
    emerald: "text-emerald-300",
    rose: "text-rose-300",
    amber: "text-amber-300",
  };
  return (
    <div className={`rounded-xl border bg-white/[0.03] px-4 py-3 backdrop-blur-sm ${colors[accent]}`}>
      <p className="text-[10px] uppercase tracking-[0.16em] text-slate-500">{label}</p>
      <p className={`mt-1 text-lg font-bold font-mono ${textColors[accent]}`}>{value}</p>
    </div>
  );
}

export default function OverviewPage() {
  const {
    hasPermission,
    fetchSnapshot,
    fetchSystemCoverage,
    portfolio,
    performance,
    riskMetrics,
    riskConcentration,
    riskCorrelation,
    riskStress,
    riskAttribution,
    tca,
    executionQuality,
    ws,
    tradingStatus,
    alerts,
    incidents,
    jobs,
    signals,
    modelStatuses,
    modelRegistry,
    modelDrift,
    sloStatus,
    systemCoverage,
    lastRefreshAt,
  } = useStore();

  useEffect(() => {
    void fetchSnapshot();
    const snapshotTimer = setInterval(() => void fetchSnapshot(), 12000);
    let coverageTimer: ReturnType<typeof setInterval> | undefined;
    if (hasPermission("read.basic")) {
      void fetchSystemCoverage();
      coverageTimer = setInterval(() => void fetchSystemCoverage(), 60000);
    }
    return () => {
      clearInterval(snapshotTimer);
      if (coverageTimer) {
        clearInterval(coverageTimer);
      }
    };
  }, [fetchSnapshot, fetchSystemCoverage, hasPermission]);

  const wsConnectedCount = useMemo(
    () => [ws.portfolio, ws.orders, ws.signals, ws.alerts].filter(Boolean).length,
    [ws],
  );
  const unresolvedAlerts = useMemo(() => alerts.filter((x) => x.status !== "RESOLVED"), [alerts]);
  const criticalAlerts = useMemo(
    () => unresolvedAlerts.filter((x) => x.severity === "CRITICAL").length,
    [unresolvedAlerts],
  );
  const activeJobs = useMemo(
    () => jobs.filter((x) => x.status === "queued" || x.status === "running").slice(0, 6),
    [jobs],
  );
  const modelHealth = useMemo(
    () => modelStatuses.filter((x) => x.status.toLowerCase() === "healthy").length,
    [modelStatuses],
  );
  const riskSignals = useMemo(
    () =>
      Object.entries(riskStress?.scenarios ?? {})
        .sort((a, b) => Number(a[1]) - Number(b[1]))
        .slice(0, 5),
    [riskStress],
  );

  const riskHeat = useMemo(() => {
    const drawdown = Math.abs((riskMetrics?.current_drawdown ?? 0) * 100);
    const concentration = Math.abs(riskConcentration?.largest_symbol_pct ?? 0);
    const correlation = Math.abs((riskCorrelation?.cluster_risk_score ?? 0) * 100);
    return Math.min(100, Math.round(drawdown * 2.2 + concentration * 0.7 + correlation * 0.35));
  }, [riskMetrics, riskConcentration, riskCorrelation]);

  const latestOrders = useMemo(() => jobs.slice(0, 5), [jobs]);
  const recentIncidents = useMemo(() => incidents.slice(0, 5), [incidents]);
  const dailyPnl = performance?.daily_pnl ?? 0;
  const pnlPositive = dailyPnl >= 0;

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-6">
      {/* Hero */}
      <motion.section variants={fadeUp} className="mission-gradient mission-grid relative overflow-hidden rounded-2xl border border-white/[0.08] p-6 shadow-lg backdrop-blur-sm">
        <div className="relative z-10 flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.24em] text-cyan-500/70">AlphaTrade Mission Control</p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-100 sm:text-4xl">
              Live Trading <span className="text-glow-cyan text-cyan-400">Cockpit</span>
            </h1>
            <p className="mt-2 max-w-3xl text-sm text-slate-400">
              Execution, risk, model governance, incidents, and infrastructure coverage in a single real-time command surface.
            </p>
          </div>
          <div className="space-y-2">
            <div className="relative status-ping">
              <Badge variant={tradingStatus?.running ? "success" : "warning"}>
                <Activity size={12} className="mr-1" />
                {tradingStatus?.running ? "Engine Live" : "Engine Idle"}
              </Badge>
            </div>
            <Badge variant={wsConnectedCount === 4 ? "success" : "warning"}>
              Streams {wsConnectedCount}/4
            </Badge>
            <Badge variant={criticalAlerts > 0 ? "error" : "outline"}>{criticalAlerts} Critical</Badge>
          </div>
        </div>

        <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <GlowMetric label="Equity" value={`$${(portfolio?.equity ?? 0).toLocaleString()}`} accent="cyan" />
          <GlowMetric
            label="Daily PnL"
            value={`${pnlPositive ? "+" : ""}$${dailyPnl.toLocaleString()}`}
            accent={pnlPositive ? "emerald" : "rose"}
          />
          <GlowMetric label="VaR 95" value={`$${(riskMetrics?.portfolio_var_95 ?? 0).toLocaleString()}`} accent="amber" />
          <GlowMetric label="Refresh" value={lastRefreshAt ? new Date(lastRefreshAt).toLocaleTimeString() : "--"} accent="cyan" />
        </div>
      </motion.section>

      {/* Signal Tape */}
      <motion.div variants={fadeUp}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl">
              <Activity size={18} className="text-cyan-400" />
              Live Signal Tape
            </CardTitle>
            <CardDescription>Real-time strategy signals with strength and confidence spread.</CardDescription>
          </CardHeader>
          <CardContent>
            <SignalTape signals={signals} />
          </CardContent>
        </Card>
      </motion.div>

      {/* Three-column grid */}
      <motion.section variants={fadeUp} className="grid gap-4 lg:grid-cols-3">
        {/* Execution Intelligence */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BriefcaseBusiness size={18} className="text-cyan-400" />
              Execution Intelligence
            </CardTitle>
            <CardDescription>Latency, slippage, and operational job flow.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="grid grid-cols-2 gap-2">
              <GlowMetric label="Fill Prob" value={`${((tca?.fill_probability ?? 0) * 100).toFixed(1)}%`} accent="emerald" />
              <GlowMetric label="Slippage" value={`${(tca?.slippage_bps ?? 0).toFixed(2)} bps`} accent="amber" />
              <GlowMetric label="Arrival Delta" value={`${(executionQuality?.arrival_price_delta_bps ?? 0).toFixed(2)} bps`} accent="cyan" />
              <GlowMetric label="Reject Rate" value={`${((executionQuality?.rejection_rate ?? 0) * 100).toFixed(2)}%`} accent="rose" />
            </div>
            <div className="space-y-2">
              {activeJobs.length === 0 ? (
                <p className="text-slate-500">No active command job.</p>
              ) : (
                activeJobs.map((job) => (
                  <div key={job.job_id} className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-slate-200">{job.command}</span>
                      <Badge variant="outline">{job.status}</Badge>
                    </div>
                    <p className="truncate font-mono text-xs text-slate-500">{job.args.join(" ")}</p>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        {/* Risk Matrix */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ShieldAlert size={18} className="text-rose-400" />
              Risk Matrix
            </CardTitle>
            <CardDescription>Drawdown, concentration, correlation, stress and breaches.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-3">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-[10px] uppercase tracking-wider text-slate-500">Risk Heat</span>
                <span className="font-mono font-semibold text-slate-200">{riskHeat}/100</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-white/[0.06]">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${riskHeat}%` }}
                  transition={{ duration: 1, ease: "easeOut" }}
                  className={riskHeat >= 70 ? "h-full bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.4)]" : riskHeat >= 40 ? "h-full bg-amber-500 shadow-[0_0_10px_rgba(245,158,11,0.3)]" : "h-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.3)]"}
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <GlowMetric label="Drawdown" value={`${((riskMetrics?.current_drawdown ?? 0) * 100).toFixed(2)}%`} accent="rose" />
              <GlowMetric label="Largest Pos" value={`${(riskConcentration?.largest_symbol_pct ?? 0).toFixed(2)}%`} accent="amber" />
              <GlowMetric label="Cluster Corr" value={(riskCorrelation?.cluster_risk_score ?? 0).toFixed(3)} accent="cyan" />
              <GlowMetric label="Breaches" value={String(riskAttribution?.breaches_count ?? 0)} accent="rose" />
            </div>
            <div className="space-y-2">
              {riskSignals.length === 0 ? (
                <p className="text-slate-500">No stress scenarios available.</p>
              ) : (
                riskSignals.map(([name, value]) => (
                  <div key={name} className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
                    <span className="text-slate-300">{name.replaceAll("_", " ")}</span>
                    <span className="font-mono text-rose-400">${Number(value).toLocaleString()}</span>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        {/* Model & SRE Pulse */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain size={18} className="text-indigo-400" />
              Model & SRE Pulse
            </CardTitle>
            <CardDescription>Model health, drift signal, SLO and incident pressure.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="grid grid-cols-2 gap-2">
              <GlowMetric label="Healthy Models" value={`${modelHealth}/${Math.max(modelStatuses.length, 1)}`} accent="emerald" />
              <GlowMetric label="Registry Versions" value={String(modelRegistry?.versions_count ?? 0)} accent="cyan" />
              <GlowMetric label="Drift Score" value={(modelDrift?.drift_score ?? 0).toFixed(3)} accent="amber" />
              <GlowMetric label="SLO Status" value={String(sloStatus?.status ?? "--")} accent="cyan" />
            </div>
            <div className="space-y-2">
              {recentIncidents.length === 0 ? (
                <p className="text-slate-500">No incident pressure.</p>
              ) : (
                recentIncidents.map((incident) => (
                  <div key={incident.incident_id} className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-slate-200">{incident.title}</span>
                      <Badge variant={incident.severity === "CRITICAL" ? "error" : "warning"}>{incident.severity}</Badge>
                    </div>
                    <p className="text-xs text-slate-500">{new Date(incident.created_at).toLocaleString()}</p>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </motion.section>

      {/* Bottom two-column */}
      <motion.section variants={fadeUp} className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Workflow size={18} className="text-cyan-400" />
              Platform Coverage
            </CardTitle>
            <CardDescription>Main CLI, script fleet, domains and datasets coverage.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-2 text-sm">
              <GlowMetric label="Main" value={systemCoverage?.main_entrypoint_exists ? "online" : "missing"} accent={systemCoverage?.main_entrypoint_exists ? "emerald" : "rose"} />
              <GlowMetric label="Scripts" value={String(systemCoverage?.scripts.length ?? 0)} accent="cyan" />
              <GlowMetric label="Domains" value={String(systemCoverage?.domains.length ?? 0)} accent="cyan" />
              <GlowMetric label="Data Files" value={String(systemCoverage?.data_assets.total_files ?? 0)} accent="amber" />
            </div>
            <div className="flex flex-wrap gap-2">
              {(systemCoverage?.domains ?? []).slice(0, 10).map((domain) => (
                <span key={domain.domain} className="rounded-full border border-white/[0.08] bg-white/[0.04] px-2.5 py-1 text-xs text-slate-300">
                  {domain.domain} ({domain.module_count})
                </span>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle size={18} className="text-amber-400" />
              Incident & Audit Pressure
            </CardTitle>
            <CardDescription>Open alerts, incident flow and control activity snapshot.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-2 text-sm">
              <GlowMetric label="Open Alerts" value={String(unresolvedAlerts.length)} accent={unresolvedAlerts.length > 0 ? "amber" : "emerald"} />
              <GlowMetric label="Critical" value={String(criticalAlerts)} accent={criticalAlerts > 0 ? "rose" : "emerald"} />
              <GlowMetric label="Incidents" value={String(incidents.length)} accent="amber" />
              <GlowMetric label="Jobs (Recent)" value={String(jobs.length)} accent="cyan" />
            </div>
            <div className="space-y-2">
              {latestOrders.length === 0 ? (
                <p className="text-sm text-slate-500">No recent control jobs.</p>
              ) : (
                latestOrders.map((job) => (
                  <div key={job.job_id} className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-slate-200">{job.command}</span>
                      <Badge variant={job.status === "succeeded" ? "success" : "outline"}>{job.status}</Badge>
                    </div>
                    <p className="truncate font-mono text-xs text-slate-500">{job.args.join(" ")}</p>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </motion.section>
    </motion.div>
  );
}
