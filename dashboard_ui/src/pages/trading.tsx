import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Activity, AlertOctagon, Gauge, Play, RefreshCcw, Square, Timer, Zap } from "lucide-react";
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

const modeOptions = ["paper", "live", "dry-run"] as const;

export default function TradingPage() {
  const {
    tradingStatus,
    tca,
    executionQuality,
    orders,
    jobs,
    mfaStatus,
    hasPermission,
    fetchTradingStatus,
    fetchTCA,
    fetchExecutionQuality,
    fetchOrders,
    fetchJobs,
    startTrading,
    stopTrading,
    restartTrading,
    activateKillSwitch,
    cancelJob,
  } = useStore();

  const [mode, setMode] = useState<(typeof modeOptions)[number]>("paper");
  const [symbols, setSymbols] = useState("AAPL,MSFT");
  const [killReason, setKillReason] = useState("");
  const [mfaCode, setMfaCode] = useState("");
  const requiresMfa = Boolean(mfaStatus?.mfa_enabled);

  useEffect(() => {
    void fetchTradingStatus();
    void fetchTCA();
    void fetchExecutionQuality();
    void fetchOrders();
    void fetchJobs();
    const timer = setInterval(() => {
      void fetchTradingStatus();
      void fetchOrders();
      void fetchJobs();
    }, 8000);
    return () => clearInterval(timer);
  }, [fetchTradingStatus, fetchTCA, fetchExecutionQuality, fetchOrders, fetchJobs]);

  const canControl = hasPermission("control.trading.start");
  const canKill = hasPermission("control.risk.kill_switch.activate");
  const activeOrders = useMemo(() => orders.filter((o) => o.status?.toLowerCase() !== "filled" && o.status?.toLowerCase() !== "cancelled").slice(0, 15), [orders]);
  const recentJobs = useMemo(() => jobs.slice(0, 8), [jobs]);

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-6">
      {/* Header */}
      <motion.section variants={fadeUp} className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-6 backdrop-blur-sm">
        <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-500/70">Execution Terminal</p>
        <h1 className="mt-2 text-3xl font-bold text-slate-100">Trading Desk</h1>
        <p className="mt-1 text-sm text-slate-400">Engine control, execution quality, active orders, and command orchestration.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          <Badge variant={tradingStatus?.running ? "success" : "outline"}>
            <Activity size={12} className="mr-1" />
            {tradingStatus?.running ? "Engine Live" : "Engine Idle"}
          </Badge>
          {tradingStatus?.running && (
            <>
              <Badge variant="outline">PID {tradingStatus.pid}</Badge>
              <Badge variant="outline">
                <Timer size={12} className="mr-1" />
                Since {tradingStatus.started_at ? new Date(tradingStatus.started_at).toLocaleTimeString() : "--"}
              </Badge>
            </>
          )}
        </div>
      </motion.section>

      {/* Engine Control + Kill Switch */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Gauge size={18} className="text-cyan-400" />
              Engine Controls
            </CardTitle>
            <CardDescription>Start, stop, or restart the trading engine.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-3">
              <label className="block flex-1 text-sm">
                <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">Mode</span>
                <select className="glass-select h-10 w-full" value={mode} onChange={(e) => setMode(e.target.value as typeof mode)}>
                  {modeOptions.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </label>
              <label className="block flex-[2] text-sm">
                <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">Symbols</span>
                <input className="glass-input h-10 w-full" value={symbols} onChange={(e) => setSymbols(e.target.value)} placeholder="AAPL,MSFT,..." />
              </label>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button
                className="gap-2"
                disabled={!canControl || tradingStatus?.running === true}
                onClick={() => void startTrading({ mode, symbols: symbols.split(",").map((s) => s.trim()).filter(Boolean), strategy: "default", capital: 100000 })}
              >
                <Play size={16} /> Start
              </Button>
              <Button
                variant="secondary"
                className="gap-2"
                disabled={!canControl || !tradingStatus?.running}
                onClick={() => void restartTrading({ mode, symbols: symbols.split(",").map((s) => s.trim()).filter(Boolean), strategy: "default", capital: 100000 })}
              >
                <RefreshCcw size={16} /> Restart
              </Button>
              <Button
                variant="destructive"
                className="gap-2"
                disabled={!canControl || !tradingStatus?.running}
                onClick={() => void stopTrading()}
              >
                <Square size={16} /> Stop
              </Button>
            </div>
            {!canControl && <p className="text-xs text-rose-400">Role lacks trading control permission.</p>}
          </CardContent>
        </Card>

        <Card className="danger-zone">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-rose-400">
              <AlertOctagon size={18} />
              Kill Switch
            </CardTitle>
            <CardDescription>Emergency halt — cancels all orders and stops engine immediately.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <label className="block text-sm">
              <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">Reason</span>
              <input className="glass-input h-10 w-full" value={killReason} onChange={(e) => setKillReason(e.target.value)} placeholder="e.g. Flash crash detected" />
            </label>
            {requiresMfa && (
              <label className="block text-sm">
                <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">MFA Code</span>
                <input className="glass-input h-10 w-full font-mono" value={mfaCode} onChange={(e) => setMfaCode(e.target.value.replace(/\D/g, "").slice(0, 6))} placeholder="6-digit TOTP" />
              </label>
            )}
            <Button
              variant="destructive"
              className="w-full gap-2"
              disabled={!canKill || !killReason || (requiresMfa && mfaCode.length !== 6)}
              onClick={() => void activateKillSwitch(killReason, mfaCode || undefined)}
            >
              <Zap size={16} /> Activate Kill Switch
            </Button>
          </CardContent>
        </Card>
      </motion.div>

      {/* Execution Quality */}
      <motion.div variants={fadeUp}>
        <Card>
          <CardHeader>
            <CardTitle>Execution Quality</CardTitle>
            <CardDescription>TCA metrics and execution analytics.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
              {[
                { label: "Fill Prob", value: `${((tca?.fill_probability ?? 0) * 100).toFixed(1)}%`, accent: "emerald" as const },
                { label: "Slippage", value: `${(tca?.slippage_bps ?? 0).toFixed(2)} bps`, accent: "amber" as const },
                { label: "Market Impact", value: `${(tca?.market_impact_bps ?? 0).toFixed(2)} bps`, accent: "amber" as const },
                { label: "Arrival Δ", value: `${(executionQuality?.arrival_price_delta_bps ?? 0).toFixed(2)} bps`, accent: "cyan" as const },
                { label: "Latency", value: `${Object.values(executionQuality?.latency_distribution_ms ?? {})[0]?.toFixed(1) ?? "--"}ms`, accent: "cyan" as const },
                { label: "Reject Rate", value: `${((executionQuality?.rejection_rate ?? 0) * 100).toFixed(2)}%`, accent: "rose" as const },
              ].map((m) => (
                <div key={m.label} className={`rounded-xl border bg-white/[0.03] px-4 py-3 ${m.accent === "emerald" ? "border-emerald-500/20" : m.accent === "amber" ? "border-amber-500/20" : m.accent === "cyan" ? "border-cyan-500/20" : "border-rose-500/20"}`}>
                  <p className="text-[10px] uppercase tracking-[0.16em] text-slate-500">{m.label}</p>
                  <p className={`mt-1 font-mono text-lg font-bold ${m.accent === "emerald" ? "text-emerald-300" : m.accent === "amber" ? "text-amber-300" : m.accent === "cyan" ? "text-cyan-300" : "text-rose-300"}`}>{m.value}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Order Blotter + Command Jobs */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Active Orders</CardTitle>
            <CardDescription>{activeOrders.length} open orders on blotter.</CardDescription>
          </CardHeader>
          <CardContent>
            {activeOrders.length === 0 ? (
              <p className="text-sm text-slate-500">No active orders.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-white/[0.06] text-xs uppercase tracking-wider text-slate-500">
                      <th className="pb-2 pr-4">Symbol</th>
                      <th className="pb-2 pr-4">Side</th>
                      <th className="pb-2 pr-4">Qty</th>
                      <th className="pb-2 pr-4">Price</th>
                      <th className="pb-2">Status</th>
                    </tr>
                  </thead>
                  <tbody className="text-slate-300">
                    {activeOrders.map((o) => (
                      <tr key={o.order_id} className="border-b border-white/[0.04] hover:bg-white/[0.02]">
                        <td className="py-2 pr-4 font-semibold text-slate-100">{o.symbol}</td>
                        <td className={`py-2 pr-4 font-mono ${o.side === "BUY" ? "text-emerald-400" : "text-rose-400"}`}>{o.side}</td>
                        <td className="py-2 pr-4 font-mono">{o.quantity}</td>
                        <td className="py-2 pr-4 font-mono">${Number(o.limit_price ?? 0).toFixed(2)}</td>
                        <td className="py-2"><Badge variant="outline">{o.status}</Badge></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Command Jobs</CardTitle>
            <CardDescription>Recent command orchestration jobs.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {recentJobs.length === 0 ? (
              <p className="text-sm text-slate-500">No jobs found.</p>
            ) : (
              recentJobs.map((job) => (
                <div key={job.job_id} className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
                  <div>
                    <p className="font-medium text-slate-200">{job.command}</p>
                    <p className="truncate font-mono text-xs text-slate-500">{job.args.join(" ")}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={job.status === "succeeded" ? "success" : job.status === "failed" ? "error" : "warning"}>
                      {job.status}
                    </Badge>
                    {(job.status === "queued" || job.status === "running") && (
                      <Button variant="ghost" size="sm" onClick={() => void cancelJob(job.job_id)}>Cancel</Button>
                    )}
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}
