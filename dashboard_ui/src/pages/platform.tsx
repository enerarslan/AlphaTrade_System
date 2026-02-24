import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { useShallow } from "zustand/react/shallow";
import { CircuitBoard, Database, Hash, Play, Terminal } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore } from "@/lib/store";

const stagger = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } },
};
const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" as const } },
} as const;

export default function PlatformPage() {
  const {
    hasPermission,
    health,
    performance,
    tradingStatus,
    signals,
    modelStatuses,
    systemCoverage,
    jobCatalog,
    fetchHealth,
    fetchPerformance,
    fetchTradingStatus,
    fetchSignals,
    fetchModelStatuses,
    fetchSystemCoverage,
    fetchJobCatalog,
    createJob,
  } = useStore(useShallow((state) => ({
      hasPermission: state.hasPermission,
      health: state.health,
      performance: state.performance,
      tradingStatus: state.tradingStatus,
      signals: state.signals,
      modelStatuses: state.modelStatuses,
      systemCoverage: state.systemCoverage,
      jobCatalog: state.jobCatalog,
      fetchHealth: state.fetchHealth,
      fetchPerformance: state.fetchPerformance,
      fetchTradingStatus: state.fetchTradingStatus,
      fetchSignals: state.fetchSignals,
      fetchModelStatuses: state.fetchModelStatuses,
      fetchSystemCoverage: state.fetchSystemCoverage,
      fetchJobCatalog: state.fetchJobCatalog,
      createJob: state.createJob,
    })));

  const [runCommand, setRunCommand] = useState("");
  const [runArgs, setRunArgs] = useState("");
  const canCreateJob = hasPermission("control.jobs.create");

  useEffect(() => {
    void fetchHealth();
    void fetchPerformance();
    void fetchTradingStatus();
    void fetchSignals();
    void fetchModelStatuses();
    void fetchSystemCoverage();
    void fetchJobCatalog();
    const timer = setInterval(() => {
      if (typeof document !== "undefined" && document.visibilityState !== "visible") return;
      void fetchHealth();
      void fetchPerformance();
      void fetchSignals();
    }, 15000);
    return () => clearInterval(timer);
  }, [fetchHealth, fetchPerformance, fetchTradingStatus, fetchSignals, fetchModelStatuses, fetchSystemCoverage, fetchJobCatalog]);

  const sigQueue = useMemo(() => {
    const buy = signals.filter((s) => (s.direction || "").toUpperCase().includes("LONG") || (s.direction || "").toUpperCase().includes("BUY")).length;
    const sell = signals.length - buy;
    return { total: signals.length, buy, sell };
  }, [signals]);

  const healthyModels = useMemo(
    () => modelStatuses.filter((m) => m.status.toLowerCase() === "healthy").length,
    [modelStatuses],
  );

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-6">
      {/* Header */}
      <motion.section variants={fadeUp} className="rounded-2xl border border-cyan-500/10 bg-cyan-500/[0.02] p-6 backdrop-blur-sm">
        <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-500/70">Infrastructure</p>
        <h1 className="mt-2 text-3xl font-bold text-slate-100">Platform Intelligence</h1>
        <p className="mt-1 text-sm text-slate-400">System performance, signals, models, and infrastructure coverage at a glance.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          <Badge variant={health?.status === "healthy" ? "success" : "warning"}>
            Health: {health?.status ?? "--"}
          </Badge>
          <Badge variant="outline">Engine: {tradingStatus?.running ? "live" : "idle"}</Badge>
        </div>
      </motion.section>

      {/* KPIs */}
      <motion.div variants={fadeUp} className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { label: "Total PnL", value: `$${(performance?.total_pnl ?? 0).toLocaleString()}`, accent: "emerald" },
          { label: "Sharpe", value: (performance?.sharpe_ratio_30d ?? 0)?.toFixed(3) ?? "--", accent: "cyan" },
          { label: "Signal Queue", value: `${sigQueue.total} (${sigQueue.buy}B / ${sigQueue.sell}S)`, accent: "amber" },
          { label: "Healthy Models", value: `${healthyModels}/${Math.max(modelStatuses.length, 1)}`, accent: "emerald" },
        ].map((m) => (
          <div key={m.label} className={`rounded-xl border bg-white/[0.03] px-4 py-3 ${m.accent === "emerald" ? "border-emerald-500/20" : m.accent === "cyan" ? "border-cyan-500/20" : "border-amber-500/20"}`}>
            <p className="text-[10px] uppercase tracking-[0.16em] text-slate-500">{m.label}</p>
            <p className={`mt-1 text-xl font-bold font-mono ${m.accent === "emerald" ? "text-emerald-300" : m.accent === "cyan" ? "text-cyan-300" : "text-amber-300"}`}>
              {m.value}
            </p>
          </div>
        ))}
      </motion.div>

      {/* Command Catalog + Run */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Terminal size={18} className="text-cyan-400" />
              Command Catalog
            </CardTitle>
            <CardDescription>Available system commands. Click to populate and run.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {jobCatalog.length === 0 ? (
              <p className="text-sm text-slate-500">No commands registered.</p>
            ) : (
              jobCatalog.map((cmd) => (
                <button
                  key={cmd.command}
                  className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] px-4 py-3 text-left transition-colors hover:border-cyan-500/20 hover:bg-white/[0.04]"
                  onClick={() => setRunCommand(cmd.command)}
                >
                  <p className="font-medium text-slate-200">{cmd.command}</p>
                  <p className="text-xs text-slate-500">{cmd.summary}</p>
                </button>
              ))
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Play size={18} className="text-emerald-400" />
              Run Command
            </CardTitle>
            <CardDescription>Execute a command from the catalog.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <label className="block text-sm">
              <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">Command</span>
              <input className="glass-input h-10 w-full" value={runCommand} onChange={(e) => setRunCommand(e.target.value)} placeholder="e.g. trade" />
            </label>
            <label className="block text-sm">
              <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">Arguments</span>
              <input className="glass-input h-10 w-full" value={runArgs} onChange={(e) => setRunArgs(e.target.value)} placeholder="--mode paper --symbols AAPL" />
            </label>
            <Button disabled={!canCreateJob || !runCommand} onClick={() => void createJob(runCommand, runArgs.split(/\s+/).filter(Boolean))}>
              Execute
            </Button>
          </CardContent>
        </Card>
      </motion.div>

      {/* Coverage Grid */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CircuitBoard size={18} className="text-cyan-400" />
              Domains
            </CardTitle>
            <CardDescription>Covered domain modules.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {(systemCoverage?.domains ?? []).length === 0 ? (
              <p className="text-sm text-slate-500">No domain data.</p>
            ) : (
              (systemCoverage?.domains ?? []).map((d) => (
                <div key={d.domain} className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
                  <span className="text-sm text-slate-300">{d.domain}</span>
                  <Badge variant="outline">{d.module_count} modules</Badge>
                </div>
              ))
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Hash size={18} className="text-amber-400" />
              Scripts
            </CardTitle>
            <CardDescription>Script fleet inventory.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {(systemCoverage?.scripts ?? []).length === 0 ? (
              <p className="text-sm text-slate-500">No scripts found.</p>
            ) : (
              (systemCoverage?.scripts ?? []).slice(0, 12).map((s) => (
                <div key={s.script_name} className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2 text-sm font-mono text-slate-400">
                  {s.script_name}
                </div>
              ))
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database size={18} className="text-emerald-400" />
              Data Assets
            </CardTitle>
            <CardDescription>Dataset inventory coverage.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3 text-center">
                <p className="text-[10px] uppercase text-slate-500">Files</p>
                <p className="text-xl font-bold font-mono text-emerald-300">{systemCoverage?.data_assets.total_files ?? 0}</p>
              </div>
              <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3 text-center">
                <p className="text-[10px] uppercase text-slate-500">Size (CSV)</p>
                <p className="text-xl font-bold font-mono text-cyan-300">{(systemCoverage?.data_assets.csv_files ?? 0)} files</p>
              </div>
            </div>
            <div className="space-y-1">
              {[
                ["CSV", systemCoverage?.data_assets.csv_files ?? 0],
                ["Parquet", systemCoverage?.data_assets.parquet_files ?? 0],
                ["JSON", systemCoverage?.data_assets.json_files ?? 0],
                ["Other", systemCoverage?.data_assets.other_files ?? 0],
              ].filter(([, count]) => Number(count) > 0).map(([ext, count]) => (
                <div key={String(ext)} className="flex items-center justify-between">
                  <span className="font-mono text-slate-400">{String(ext)}</span>
                  <span className="font-mono text-slate-300">{String(count)}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}


