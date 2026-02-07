import { useEffect, useMemo } from "react";
import { Activity, AlertTriangle, Brain, BriefcaseBusiness, ShieldAlert, Workflow } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SignalTape } from "@/components/live/SignalTape";
import { useStore } from "@/lib/store";

function MetricChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white/80 px-3 py-2 shadow-sm">
      <p className="text-[11px] uppercase tracking-[0.16em] text-slate-500">{label}</p>
      <p className="mt-1 text-sm font-semibold text-slate-900">{value}</p>
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

  return (
    <div className="space-y-6">
      <section className="mission-gradient mission-grid relative overflow-hidden rounded-3xl border border-slate-200 p-6 shadow-sm">
        <div className="relative z-10 flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-600">AlphaTrade Mission Control</p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
              JPMorgan-Class Live Trading Cockpit
            </h1>
            <p className="mt-2 max-w-3xl text-sm text-slate-700">
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
            <Badge variant={criticalAlerts > 0 ? "error" : "outline"}>{criticalAlerts} Critical Alerts</Badge>
          </div>
        </div>

        <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <MetricChip label="Equity" value={`$${(portfolio?.equity ?? 0).toLocaleString()}`} />
          <MetricChip label="Daily PnL" value={`$${(performance?.daily_pnl ?? 0).toLocaleString()}`} />
          <MetricChip label="VaR 95" value={`$${(riskMetrics?.portfolio_var_95 ?? 0).toLocaleString()}`} />
          <MetricChip label="Refresh" value={lastRefreshAt ? new Date(lastRefreshAt).toLocaleTimeString() : "--"} />
        </div>
      </section>

      <Card className="border-slate-200 bg-white/95">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <Activity size={18} />
            Live Signal Tape
          </CardTitle>
          <CardDescription>Real-time strategy signals with strength and confidence spread.</CardDescription>
        </CardHeader>
        <CardContent>
          <SignalTape signals={signals} />
        </CardContent>
      </Card>

      <section className="grid gap-4 lg:grid-cols-3">
        <Card className="border-slate-200 bg-white/95">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BriefcaseBusiness size={18} />
              Execution Intelligence
            </CardTitle>
            <CardDescription>Latency, slippage, and operational job flow.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="grid grid-cols-2 gap-2">
              <MetricChip label="Fill Prob" value={`${((tca?.fill_probability ?? 0) * 100).toFixed(1)}%`} />
              <MetricChip label="Slippage" value={`${(tca?.slippage_bps ?? 0).toFixed(2)} bps`} />
              <MetricChip label="Arrival Delta" value={`${(executionQuality?.arrival_price_delta_bps ?? 0).toFixed(2)} bps`} />
              <MetricChip label="Reject Rate" value={`${((executionQuality?.rejection_rate ?? 0) * 100).toFixed(2)}%`} />
            </div>
            <div className="space-y-2">
              {activeJobs.length === 0 ? (
                <p className="text-slate-500">No active command job.</p>
              ) : (
                activeJobs.map((job) => (
                  <div key={job.job_id} className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-slate-800">{job.command}</span>
                      <Badge variant="outline">{job.status}</Badge>
                    </div>
                    <p className="truncate font-mono text-xs text-slate-500">{job.args.join(" ")}</p>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/95">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ShieldAlert size={18} />
              Risk Matrix
            </CardTitle>
            <CardDescription>Drawdown, concentration, correlation, stress and breaches.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-xs uppercase tracking-wider text-slate-500">Risk Heat</span>
                <span className="font-mono font-semibold text-slate-800">{riskHeat}/100</span>
              </div>
              <div className="h-2 overflow-hidden rounded bg-slate-200">
                <div
                  className={riskHeat >= 70 ? "h-full bg-rose-600" : riskHeat >= 40 ? "h-full bg-amber-500" : "h-full bg-emerald-500"}
                  style={{ width: `${riskHeat}%` }}
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <MetricChip label="Drawdown" value={`${((riskMetrics?.current_drawdown ?? 0) * 100).toFixed(2)}%`} />
              <MetricChip label="Largest Pos" value={`${(riskConcentration?.largest_symbol_pct ?? 0).toFixed(2)}%`} />
              <MetricChip label="Cluster Corr" value={(riskCorrelation?.cluster_risk_score ?? 0).toFixed(3)} />
              <MetricChip label="Breaches" value={String(riskAttribution?.breaches_count ?? 0)} />
            </div>
            <div className="space-y-2">
              {riskSignals.length === 0 ? (
                <p className="text-slate-500">No stress scenarios available.</p>
              ) : (
                riskSignals.map(([name, value]) => (
                  <div key={name} className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                    <span className="text-slate-700">{name.replaceAll("_", " ")}</span>
                    <span className="font-mono text-rose-700">${Number(value).toLocaleString()}</span>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/95">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain size={18} />
              Model & SRE Pulse
            </CardTitle>
            <CardDescription>Model health, drift signal, SLO and incident pressure.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="grid grid-cols-2 gap-2">
              <MetricChip label="Healthy Models" value={`${modelHealth}/${Math.max(modelStatuses.length, 1)}`} />
              <MetricChip label="Registry Versions" value={String(modelRegistry?.versions_count ?? 0)} />
              <MetricChip label="Drift Score" value={(modelDrift?.drift_score ?? 0).toFixed(3)} />
              <MetricChip label="SLO Status" value={String(sloStatus?.status ?? "--")} />
            </div>
            <div className="space-y-2">
              {recentIncidents.length === 0 ? (
                <p className="text-slate-500">No incident pressure.</p>
              ) : (
                recentIncidents.map((incident) => (
                  <div key={incident.incident_id} className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-slate-800">{incident.title}</span>
                      <Badge variant={incident.severity === "CRITICAL" ? "error" : "warning"}>{incident.severity}</Badge>
                    </div>
                    <p className="text-xs text-slate-500">{new Date(incident.created_at).toLocaleString()}</p>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card className="border-slate-200 bg-white/95">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Workflow size={18} />
              Platform Coverage
            </CardTitle>
            <CardDescription>Main CLI, script fleet, domains and datasets coverage.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-2 text-sm">
              <MetricChip label="Main" value={systemCoverage?.main_entrypoint_exists ? "online" : "missing"} />
              <MetricChip label="Scripts" value={String(systemCoverage?.scripts.length ?? 0)} />
              <MetricChip label="Domains" value={String(systemCoverage?.domains.length ?? 0)} />
              <MetricChip label="Data Files" value={String(systemCoverage?.data_assets.total_files ?? 0)} />
            </div>
            <div className="flex flex-wrap gap-2">
              {(systemCoverage?.domains ?? []).slice(0, 10).map((domain) => (
                <span key={domain.domain} className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-700">
                  {domain.domain} ({domain.module_count})
                </span>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/95">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle size={18} />
              Incident & Audit Pressure
            </CardTitle>
            <CardDescription>Open alerts, incident flow and control activity snapshot.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-2 text-sm">
              <MetricChip label="Open Alerts" value={String(unresolvedAlerts.length)} />
              <MetricChip label="Critical" value={String(criticalAlerts)} />
              <MetricChip label="Incidents" value={String(incidents.length)} />
              <MetricChip label="Jobs (Recent)" value={String(jobs.length)} />
            </div>
            <div className="space-y-2">
              {latestOrders.length === 0 ? (
                <p className="text-sm text-slate-500">No recent control jobs.</p>
              ) : (
                latestOrders.map((job) => (
                  <div key={job.job_id} className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-slate-800">{job.command}</span>
                      <Badge variant={job.status === "succeeded" ? "success" : "outline"}>{job.status}</Badge>
                    </div>
                    <p className="truncate font-mono text-xs text-slate-500">{job.args.join(" ")}</p>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}

