import { useEffect, useMemo } from "react";
import { Activity, Boxes, Brain, Database, TerminalSquare } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore } from "@/lib/store";

export default function PlatformPage() {
  const {
    hasPermission,
    performance,
    signals,
    modelStatuses,
    jobs,
    jobCatalog,
    systemCoverage,
    fetchPerformance,
    fetchSignals,
    fetchModelStatuses,
    fetchJobs,
    fetchJobCatalog,
    fetchSystemCoverage,
    createJob,
  } = useStore();

  const canCreateJobs = hasPermission("control.jobs.create");
  const canReadJobs = hasPermission("control.jobs.read");

  useEffect(() => {
    void Promise.all([
      fetchPerformance(),
      fetchSignals(),
      fetchModelStatuses(),
      fetchJobs(),
      fetchJobCatalog(),
      fetchSystemCoverage(),
    ]);
    const timer = setInterval(
      () =>
        void Promise.all([
          fetchPerformance(),
          fetchSignals(),
          fetchModelStatuses(),
          fetchJobs(),
        ]),
      15000,
    );
    return () => clearInterval(timer);
  }, [fetchPerformance, fetchSignals, fetchModelStatuses, fetchJobs, fetchJobCatalog, fetchSystemCoverage]);

  const recentSignals = useMemo(() => signals.slice(0, 12), [signals]);
  const recentJobs = useMemo(() => jobs.slice(0, 8), [jobs]);
  const healthyModelCount = useMemo(
    () => modelStatuses.filter((entry) => entry.status.toLowerCase() === "healthy").length,
    [modelStatuses],
  );

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-slate-200 bg-white/90 p-6">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">Platform Coverage</p>
        <h1 className="mt-2 text-3xl font-bold text-slate-900">Main + Scripts + Runtime Intelligence</h1>
        <p className="mt-1 text-sm text-slate-600">
          Unified view for `main.py` orchestration, `scripts/*` controls, and `quant_trading_system` domain coverage.
        </p>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Daily PnL</CardDescription>
            <CardTitle className="text-3xl">${(performance?.daily_pnl ?? 0).toLocaleString()}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Sharpe 30d</CardDescription>
            <CardTitle className="text-3xl">{(performance?.sharpe_ratio_30d ?? 0).toFixed(3)}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Signal Queue</CardDescription>
            <CardTitle className="text-3xl">{signals.length}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Healthy Models</CardDescription>
            <CardTitle className="text-3xl">{healthyModelCount}</CardTitle>
          </CardHeader>
        </Card>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TerminalSquare size={18} />
              Command Catalog
            </CardTitle>
            <CardDescription>Allowlisted commands wired to `main.py` and `scripts/*`.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {!canReadJobs ? (
              <p className="text-sm text-rose-700">Role lacks command catalog permission.</p>
            ) : jobCatalog.length === 0 ? (
              <p className="text-sm text-slate-500">No command catalog entries available.</p>
            ) : (
              jobCatalog.map((entry) => (
                <div key={entry.command} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="font-semibold text-slate-800">{entry.command}</p>
                    <Badge variant={entry.risk_level === "high" ? "warning" : "outline"}>{entry.risk_level}</Badge>
                  </div>
                  <p className="mt-1 text-sm text-slate-600">{entry.summary}</p>
                  <p className="mt-1 font-mono text-xs text-slate-500">{entry.script_path}</p>
                  <div className="mt-2 flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={!canCreateJobs}
                      onClick={() => void createJob(entry.command, entry.sample_args)}
                    >
                      Run Template
                    </Button>
                    <span className="truncate font-mono text-xs text-slate-500">{entry.sample_args.join(" ")}</span>
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Boxes size={18} />
              System Coverage
            </CardTitle>
            <CardDescription>Inventory summary for core project surfaces.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
              <p>
                Main entrypoint:
                <span className="ml-1 font-mono">{systemCoverage?.main_entrypoint ?? "main.py"}</span>
              </p>
              <p>
                Exists:
                <span className="ml-1 font-semibold">{systemCoverage?.main_entrypoint_exists ? "yes" : "no"}</span>
              </p>
              <p>
                Scripts tracked:
                <span className="ml-1 font-semibold">{systemCoverage?.scripts.length ?? 0}</span>
              </p>
              <p>
                Domains tracked:
                <span className="ml-1 font-semibold">{systemCoverage?.domains.length ?? 0}</span>
              </p>
            </div>

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
              <p>Data root: <span className="font-mono">{systemCoverage?.data_assets.root ?? "data"}</span></p>
              <p>Total files: <span className="font-semibold">{systemCoverage?.data_assets.total_files ?? 0}</span></p>
              <p>CSV/TSV: <span className="font-semibold">{systemCoverage?.data_assets.csv_files ?? 0}</span></p>
              <p>Parquet: <span className="font-semibold">{systemCoverage?.data_assets.parquet_files ?? 0}</span></p>
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              {(systemCoverage?.domains ?? []).slice(0, 8).map((entry) => (
                <div key={entry.domain} className="rounded-lg border border-slate-200 bg-slate-50 p-2">
                  <p className="text-sm font-semibold text-slate-800">{entry.domain}</p>
                  <p className="text-xs text-slate-600">{entry.module_count} modules</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity size={18} />
              Recent Signals
            </CardTitle>
            <CardDescription>Latest signal payloads from event bus stream.</CardDescription>
          </CardHeader>
          <CardContent className="max-h-[360px] overflow-auto">
            {recentSignals.length === 0 ? (
              <p className="text-sm text-slate-500">No signal activity yet.</p>
            ) : (
              <table className="w-full text-left text-sm">
                <thead className="sticky top-0 bg-white">
                  <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                    <th className="py-2">Symbol</th>
                    <th className="py-2">Dir</th>
                    <th className="py-2">Strength</th>
                    <th className="py-2">Model</th>
                  </tr>
                </thead>
                <tbody>
                  {recentSignals.map((signal, idx) => (
                    <tr key={`${signal.signal_id || signal.timestamp}_${idx}`} className="border-b border-slate-100">
                      <td className="py-2 font-medium text-slate-800">{signal.symbol}</td>
                      <td className="py-2">{signal.direction}</td>
                      <td className="py-2">{Number(signal.strength ?? 0).toFixed(3)}</td>
                      <td className="py-2 font-mono text-xs">{signal.model_source || "--"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain size={18} />
              Model Runtime Status
            </CardTitle>
            <CardDescription>Live model health and prediction counters.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {modelStatuses.length === 0 ? (
              <p className="text-sm text-slate-500">No model status reported yet.</p>
            ) : (
              modelStatuses.slice(0, 10).map((entry) => (
                <div key={entry.model_name} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div className="flex items-center justify-between">
                    <p className="font-semibold text-slate-800">{entry.model_name}</p>
                    <Badge variant={entry.status.toLowerCase() === "healthy" ? "success" : "warning"}>
                      {entry.status}
                    </Badge>
                  </div>
                  <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-slate-600">
                    <p>predictions: {entry.prediction_count}</p>
                    <p>errors: {entry.error_count}</p>
                    <p>accuracy: {entry.accuracy?.toFixed(3) ?? "--"}</p>
                    <p>auc: {entry.auc?.toFixed(3) ?? "--"}</p>
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </section>

      <Card className="border-slate-200 bg-white/90">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database size={18} />
            Recent Command Jobs
          </CardTitle>
          <CardDescription>Most recent asynchronous runs triggered from dashboard controls.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {recentJobs.length === 0 ? (
            <p className="text-sm text-slate-500">No jobs available.</p>
          ) : (
            recentJobs.map((job) => (
              <div key={job.job_id} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div className="flex items-center justify-between">
                  <p className="font-semibold text-slate-800">{job.command}</p>
                  <Badge variant={job.status === "succeeded" ? "success" : "outline"}>{job.status}</Badge>
                </div>
                <p className="mt-1 truncate font-mono text-xs text-slate-600">{job.args.join(" ")}</p>
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}
