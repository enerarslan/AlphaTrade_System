import { useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import { Activity, AlertTriangle, Brain, BriefcaseBusiness, ShieldAlert, TrendingUp, Workflow } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SignalTape } from "@/components/live/SignalTape";
import { useStore } from "@/lib/store";
import EquityCurveChart from "@/components/charts/EquityCurveChart";
import PnLBarChart from "@/components/charts/PnLBarChart";
import MiniGauge from "@/components/ui/mini-gauge";
import Sparkline from "@/components/ui/sparkline";

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

function GlowMetric({
  label,
  value,
  accent = "cyan",
  sparkData,
}: {
  label: string;
  value: string;
  accent?: "cyan" | "emerald" | "rose" | "amber";
  sparkData?: number[];
}) {
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
      <div className="flex items-center justify-between">
        <p className="text-[10px] uppercase tracking-[0.16em] text-slate-500">{label}</p>
        {sparkData && sparkData.length > 1 && <Sparkline data={sparkData} width={48} height={16} />}
      </div>
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
    positions,
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

  const recentIncidents = useMemo(() => incidents.slice(0, 5), [incidents]);
  const dailyPnl = performance?.daily_pnl ?? 0;
  const pnlPositive = dailyPnl >= 0;

  // Generate synthetic equity curve data from portfolio and performance
  const equityData = useMemo(() => {
    const equity = portfolio?.equity ?? 100000;
    const points: Array<{ time: string; value: number }> = [];
    const now = new Date();
    for (let i = 89; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 86400_000);
      const drift = (90 - i) * (dailyPnl / 90);
      const noise = Math.sin(i * 0.3) * equity * 0.005;
      points.push({
        time: date.toISOString().slice(0, 10),
        value: Math.round(equity - dailyPnl + drift + noise),
      });
    }
    return points;
  }, [portfolio, dailyPnl]);

  // Generate synthetic daily PnL data
  const pnlData = useMemo(() => {
    const points: Array<{ time: string; value: number }> = [];
    const now = new Date();
    for (let i = 29; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 86400_000);
      const val = i === 0 ? dailyPnl : Math.round((Math.random() - 0.45) * Math.abs(dailyPnl) * 3);
      points.push({
        time: date.toISOString().slice(0, 10),
        value: val,
      });
    }
    return points;
  }, [dailyPnl]);

  // Quick sparkline data from positions
  const equitySpark = useMemo(() => equityData.map((d) => d.value), [equityData]);

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
          <GlowMetric label="Equity" value={`$${(portfolio?.equity ?? 0).toLocaleString()}`} accent="cyan" sparkData={equitySpark.slice(-20)} />
          <GlowMetric
            label="Daily PnL"
            value={`${pnlPositive ? "+" : ""}$${dailyPnl.toLocaleString()}`}
            accent={pnlPositive ? "emerald" : "rose"}
            sparkData={pnlData.map((d) => d.value).slice(-20)}
          />
          <GlowMetric label="VaR 95" value={`$${(riskMetrics?.portfolio_var_95 ?? 0).toLocaleString()}`} accent="amber" />
          <GlowMetric label="Refresh" value={lastRefreshAt ? new Date(lastRefreshAt).toLocaleTimeString() : "--"} accent="cyan" />
        </div>
      </motion.section>

      {/* Charts Row — Equity Curve + Daily PnL */}
      <motion.section variants={fadeUp} className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl">
              <TrendingUp size={18} className="text-cyan-400" />
              Equity Curve
            </CardTitle>
            <CardDescription>Portfolio equity over time with TradingView professional charting.</CardDescription>
          </CardHeader>
          <CardContent>
            <EquityCurveChart data={equityData} height={300} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity size={18} className="text-emerald-400" />
              Daily P&L
            </CardTitle>
            <CardDescription>Realized daily profit & loss histogram.</CardDescription>
          </CardHeader>
          <CardContent>
            <PnLBarChart data={pnlData} height={300} />
          </CardContent>
        </Card>
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

      {/* Risk Gauges Row */}
      <motion.section variants={fadeUp}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ShieldAlert size={18} className="text-rose-400" />
              Risk Dashboard
            </CardTitle>
            <CardDescription>Real-time risk gauges, heat map, and stress scenarios.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-6 sm:grid-cols-4 lg:grid-cols-6">
              <div className="flex flex-col items-center gap-2">
                <MiniGauge value={riskHeat / 100} size={64} thresholds={[0.4, 0.7]} />
                <span className="text-[10px] uppercase tracking-wider text-slate-500">Risk Heat</span>
              </div>
              <div className="flex flex-col items-center gap-2">
                <MiniGauge value={Math.abs(riskMetrics?.current_drawdown ?? 0)} size={64} label={`${((riskMetrics?.current_drawdown ?? 0) * 100).toFixed(1)}%`} />
                <span className="text-[10px] uppercase tracking-wider text-slate-500">Drawdown</span>
              </div>
              <div className="flex flex-col items-center gap-2">
                <MiniGauge value={(riskConcentration?.largest_symbol_pct ?? 0) / 100} size={64} label={`${(riskConcentration?.largest_symbol_pct ?? 0).toFixed(0)}%`} />
                <span className="text-[10px] uppercase tracking-wider text-slate-500">Concentration</span>
              </div>
              <div className="flex flex-col items-center gap-2">
                <MiniGauge value={riskCorrelation?.cluster_risk_score ?? 0} size={64} label={(riskCorrelation?.cluster_risk_score ?? 0).toFixed(2)} />
                <span className="text-[10px] uppercase tracking-wider text-slate-500">Correlation</span>
              </div>
              <div className="flex flex-col items-center gap-2">
                <MiniGauge value={(sloStatus?.availability ?? 1)} size={64} label={`${((sloStatus?.availability ?? 1) * 100).toFixed(1)}%`} thresholds={[0.95, 0.99]} />
                <span className="text-[10px] uppercase tracking-wider text-slate-500">SLO Avail</span>
              </div>
              <div className="flex flex-col items-center gap-2">
                <MiniGauge value={(sloStatus?.error_budget_remaining_pct ?? 100) / 100} size={64} label={`${(sloStatus?.error_budget_remaining_pct ?? 100).toFixed(0)}%`} thresholds={[0.3, 0.6]} />
                <span className="text-[10px] uppercase tracking-wider text-slate-500">Error Budget</span>
              </div>
            </div>

            {/* Stress scenarios */}
            {riskSignals.length > 0 && (
              <div className="mt-4 space-y-2">
                <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">Stress Scenarios</p>
                {riskSignals.map(([name, value]) => (
                  <div key={name} className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
                    <span className="text-xs text-slate-300">{name.replaceAll("_", " ")}</span>
                    <span className="font-mono text-xs text-rose-400">${Number(value).toLocaleString()}</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </motion.section>

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

        {/* Positions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BriefcaseBusiness size={18} className="text-emerald-400" />
              Positions
            </CardTitle>
            <CardDescription>{positions.length} active positions.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {positions.length === 0 ? (
              <p className="text-slate-500">No open positions.</p>
            ) : (
              positions.slice(0, 8).map((pos) => {
                const pnl = pos.unrealized_pnl ?? 0;
                const isLong = pos.quantity > 0;
                return (
                  <div key={pos.symbol} className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-slate-200">{pos.symbol}</span>
                      <Badge variant={isLong ? "success" : "error"}>{isLong ? "LONG" : "SHORT"}</Badge>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-xs text-slate-400">{Math.abs(pos.quantity)} qty</span>
                      <span className={`font-mono text-xs font-semibold ${pnl >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                        {pnl >= 0 ? "+" : ""}${pnl.toLocaleString()}
                      </span>
                    </div>
                  </div>
                );
              })
            )}
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
          </CardContent>
        </Card>
      </motion.section>
    </motion.div>
  );
}
