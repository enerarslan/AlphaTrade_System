import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { useShallow } from "zustand/react/shallow";
import { FlaskConical, Play, Square, Clock, Terminal, ChevronDown, ChevronUp } from "lucide-react";
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

function statusVariant(
  status: string,
): "success" | "error" | "warning" | "default" | "outline" | "secondary" {
  switch (status) {
    case "completed":
      return "success";
    case "failed":
      return "error";
    case "running":
      return "warning";
    case "queued":
      return "outline";
    case "cancelled":
      return "secondary";
    default:
      return "outline";
  }
}

type ParsedMetrics = Record<string, number | string>;

export default function BacktestPage() {
  const { hasPermission, jobs, fetchJobs, createJob, cancelJob } = useStore(
    useShallow((state) => ({
      hasPermission: state.hasPermission,
      jobs: state.jobs,
      fetchJobs: state.fetchJobs,
      createJob: state.createJob,
      cancelJob: state.cancelJob,
    })),
  );

  const canCreateJob = hasPermission("control.jobs.create");

  // Form state
  const today = new Date().toISOString().slice(0, 10);
  const sixMonthsAgo = new Date(Date.now() - 182 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
  const [startDate, setStartDate] = useState(sixMonthsAgo);
  const [endDate, setEndDate] = useState(today);
  const [symbols, setSymbols] = useState("AAPL,MSFT,GOOGL");
  const [strategy, setStrategy] = useState("default");
  const [capital, setCapital] = useState(100000);
  const [launching, setLaunching] = useState(false);

  // UI state
  const [expandedJob, setExpandedJob] = useState<string | null>(null);

  useEffect(() => {
    void fetchJobs();
    const timer = setInterval(() => {
      if (typeof document !== "undefined" && document.visibilityState !== "visible") return;
      void fetchJobs();
    }, 5000);
    return () => clearInterval(timer);
  }, [fetchJobs]);

  const backtestJobs = useMemo(
    () =>
      jobs
        .filter((j) => j.command === "backtest")
        .sort(
          (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
        ),
    [jobs],
  );

  const runningCount = useMemo(
    () => backtestJobs.filter((j) => j.status === "running").length,
    [backtestJobs],
  );

  const handleLaunch = async () => {
    if (!startDate || !endDate) return;
    setLaunching(true);
    try {
      await createJob("backtest", [
        "--start", startDate,
        "--end", endDate,
        "--symbols", symbols,
        "--strategy", strategy,
        "--capital", String(capital),
      ]);
    } finally {
      setLaunching(false);
    }
  };

  const selectedJobOutput = useMemo<ParsedMetrics | null>(() => {
    if (!expandedJob) return null;
    const job = backtestJobs.find((j) => j.job_id === expandedJob);
    if (!job?.output) return null;

    try {
      const jsonMatch = job.output.match(/\{[\s\S]*"sharpe"[\s\S]*\}/);
      if (jsonMatch) return JSON.parse(jsonMatch[0]) as ParsedMetrics;
    } catch {
      // output does not contain parseable JSON metrics — silently ignore
    }
    return null;
  }, [expandedJob, backtestJobs]);

  const metricLabel = (key: string) =>
    key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

  const metricColor = (key: string, value: number | string): string => {
    const v = typeof value === "number" ? value : parseFloat(String(value));
    if (isNaN(v)) return "text-slate-300";
    if (key.includes("drawdown") || key.includes("loss"))
      return v < 0 ? "text-rose-300" : "text-emerald-300";
    if (key.includes("return") || key.includes("pnl") || key.includes("profit"))
      return v >= 0 ? "text-emerald-300" : "text-rose-300";
    if (key.includes("sharpe") || key.includes("sortino"))
      return v >= 1 ? "text-emerald-300" : v >= 0 ? "text-amber-300" : "text-rose-300";
    return "text-cyan-300";
  };

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-6">
      {/* Header */}
      <motion.section
        variants={fadeUp}
        className="rounded-2xl border border-teal-500/10 bg-teal-500/[0.02] p-6 backdrop-blur-sm"
      >
        <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-teal-400/70">
          Strategy Research
        </p>
        <h1 className="mt-2 text-3xl font-bold text-slate-100">Backtest Lab</h1>
        <p className="mt-1 text-sm text-slate-400">
          Launch historical simulations, monitor job progress, and inspect parsed results.
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          <Badge variant="outline">
            <FlaskConical size={11} className="mr-1 inline text-teal-400" />
            {backtestJobs.length} total backtests
          </Badge>
          {runningCount > 0 && (
            <Badge variant="warning">
              <span className="mr-1 animate-pulse">&#9679;</span>
              {runningCount} running
            </Badge>
          )}
        </div>
      </motion.section>

      {/* Launch New Backtest */}
      <motion.div variants={fadeUp}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Play size={18} className="text-teal-400" />
              Launch New Backtest
            </CardTitle>
            <CardDescription>
              Configure and submit a new backtest job to the async queue.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!canCreateJob ? (
              <p className="text-sm text-slate-500">
                You do not have permission to launch backtest jobs (control.jobs.create required).
              </p>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                <label className="block text-sm">
                  <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">
                    Start Date
                  </span>
                  <input
                    type="date"
                    className="glass-input h-10 w-full"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                  />
                </label>

                <label className="block text-sm">
                  <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">
                    End Date
                  </span>
                  <input
                    type="date"
                    className="glass-input h-10 w-full"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                  />
                </label>

                <label className="block text-sm">
                  <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">
                    Symbols (comma separated)
                  </span>
                  <input
                    type="text"
                    className="glass-input h-10 w-full font-mono"
                    value={symbols}
                    onChange={(e) => setSymbols(e.target.value)}
                    placeholder="AAPL,MSFT,GOOGL"
                  />
                </label>

                <label className="block text-sm">
                  <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">
                    Strategy
                  </span>
                  <select
                    className="glass-input h-10 w-full"
                    value={strategy}
                    onChange={(e) => setStrategy(e.target.value)}
                  >
                    <option value="default">Default</option>
                    <option value="momentum">Momentum</option>
                    <option value="mean_reversion">Mean Reversion</option>
                    <option value="ml_alpha">ML Alpha</option>
                  </select>
                </label>

                <label className="block text-sm">
                  <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">
                    Initial Capital ($)
                  </span>
                  <input
                    type="number"
                    className="glass-input h-10 w-full font-mono"
                    value={capital}
                    min={1000}
                    step={1000}
                    onChange={(e) => setCapital(Number(e.target.value))}
                  />
                </label>

                <div className="flex items-end">
                  <Button
                    className="h-10 w-full"
                    disabled={launching || !startDate || !endDate || !symbols}
                    onClick={() => void handleLaunch()}
                  >
                    <Play size={14} className="mr-2" />
                    {launching ? "Launching..." : "Launch Backtest"}
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Backtest History */}
      <motion.div variants={fadeUp}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock size={18} className="text-cyan-400" />
              Backtest History
            </CardTitle>
            <CardDescription>
              All submitted backtest jobs, newest first. Expand a job to inspect its output.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {backtestJobs.length === 0 ? (
              <p className="text-sm text-slate-500">
                No backtest jobs found. Launch one above.
              </p>
            ) : (
              backtestJobs.map((job) => {
                const isExpanded = expandedJob === job.job_id;
                const durationMs =
                  job.ended_at && job.started_at
                    ? new Date(job.ended_at).getTime() - new Date(job.started_at).getTime()
                    : null;

                return (
                  <div
                    key={job.job_id}
                    className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4"
                  >
                    <div className="flex items-center justify-between">
                      <div className="min-w-0 flex-1">
                        <p className="font-medium text-slate-200">
                          Backtest #{job.job_id.slice(0, 8)}
                        </p>
                        <p className="mt-0.5 truncate text-xs text-slate-500">
                          {job.args.join(" ")}
                        </p>
                      </div>
                      <div className="ml-3 flex flex-shrink-0 items-center gap-2">
                        <Badge variant={statusVariant(job.status)}>{job.status}</Badge>
                        {job.status === "running" && (
                          <span className="animate-pulse text-xs text-cyan-400">
                            Running...
                          </span>
                        )}
                        {(job.status === "queued" || job.status === "running") && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 px-2 text-xs text-rose-400 hover:text-rose-300"
                            onClick={() => void cancelJob(job.job_id)}
                          >
                            <Square size={12} className="mr-1" />
                            Cancel
                          </Button>
                        )}
                      </div>
                    </div>

                    {/* Timestamps */}
                    <div className="mt-2 flex flex-wrap gap-4 text-xs text-slate-500">
                      <span>Created: {new Date(job.created_at).toLocaleString()}</span>
                      {job.started_at && (
                        <span>Started: {new Date(job.started_at).toLocaleString()}</span>
                      )}
                      {job.ended_at && (
                        <span>Completed: {new Date(job.ended_at).toLocaleString()}</span>
                      )}
                      {durationMs !== null && (
                        <span>Duration: {(durationMs / 1000).toFixed(0)}s</span>
                      )}
                      {job.exit_code !== null && job.exit_code !== undefined && (
                        <span
                          className={
                            job.exit_code === 0 ? "text-emerald-500" : "text-rose-500"
                          }
                        >
                          Exit: {job.exit_code}
                        </span>
                      )}
                    </div>

                    {/* Expandable output */}
                    {job.output && isExpanded && (
                      <div className="mt-3 max-h-[400px] overflow-auto rounded-lg border border-white/[0.06] bg-black/40 p-3 font-mono text-xs text-slate-300 whitespace-pre-wrap">
                        {job.output}
                      </div>
                    )}

                    {job.output && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="mt-2 h-7 px-2 text-xs text-slate-400 hover:text-slate-200"
                        onClick={() =>
                          setExpandedJob(isExpanded ? null : job.job_id)
                        }
                      >
                        <Terminal size={12} className="mr-1" />
                        {isExpanded ? (
                          <>
                            <ChevronUp size={12} className="mr-1" />
                            Hide Output
                          </>
                        ) : (
                          <>
                            <ChevronDown size={12} className="mr-1" />
                            Show Output
                          </>
                        )}
                      </Button>
                    )}
                  </div>
                );
              })
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Parsed Results */}
      {selectedJobOutput && (
        <motion.div variants={fadeUp}>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FlaskConical size={18} className="text-teal-400" />
                Parsed Results
                <span className="text-xs font-normal text-slate-500">
                  — Backtest #{expandedJob?.slice(0, 8)}
                </span>
              </CardTitle>
              <CardDescription>
                Structured performance metrics extracted from the backtest output JSON.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {Object.entries(selectedJobOutput).map(([key, value]) => {
                  const numVal =
                    typeof value === "number" ? value : parseFloat(String(value));
                  const displayValue = isNaN(numVal)
                    ? String(value)
                    : Number.isInteger(numVal)
                    ? numVal.toLocaleString()
                    : numVal.toFixed(4);

                  return (
                    <div
                      key={key}
                      className="rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3"
                    >
                      <p className="text-[10px] uppercase tracking-[0.16em] text-slate-500">
                        {metricLabel(key)}
                      </p>
                      <p
                        className={`mt-1 font-mono text-xl font-bold ${metricColor(key, value)}`}
                      >
                        {displayValue}
                      </p>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </motion.div>
  );
}
