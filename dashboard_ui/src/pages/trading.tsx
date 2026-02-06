import { useEffect, useMemo, useState } from "react";
import { Play, RotateCcw, Square, Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore } from "@/lib/store";

const defaultPayload = {
  mode: "paper" as const,
  symbols: ["AAPL", "MSFT", "NVDA"],
  strategy: "momentum",
  capital: 100000,
};

export default function TradingTerminalPage() {
  const {
    hasPermission,
    mfaStatus,
    orders,
    tca,
    executionQuality,
    tradingStatus,
    startTrading,
    stopTrading,
    restartTrading,
    activateKillSwitch,
    createJob,
    fetchOrders,
    fetchTCA,
    fetchExecutionQuality,
    fetchTradingStatus,
    fetchJobs,
    jobs,
  } = useStore();

  const [mode, setMode] = useState<"live" | "paper" | "dry-run">(defaultPayload.mode);
  const [symbols, setSymbols] = useState(defaultPayload.symbols.join(","));
  const [strategy, setStrategy] = useState(defaultPayload.strategy);
  const [capital, setCapital] = useState(defaultPayload.capital);
  const [killReason, setKillReason] = useState("manual_activation");
  const [mfaCode, setMfaCode] = useState("");

  useEffect(() => {
    void Promise.all([fetchOrders(), fetchTCA(), fetchExecutionQuality(), fetchTradingStatus(), fetchJobs()]);
  }, [fetchOrders, fetchTCA, fetchExecutionQuality, fetchTradingStatus, fetchJobs]);

  const topOrders = useMemo(() => orders.slice(0, 40), [orders]);
  const runningJobs = jobs.filter((x) => x.status === "running" || x.status === "queued").slice(0, 6);
  const canControlTrading = hasPermission("control.trading.start") || hasPermission("control.trading.restart") || hasPermission("control.trading.stop");
  const canStart = hasPermission("control.trading.start");
  const canRestart = hasPermission("control.trading.restart");
  const canStop = hasPermission("control.trading.stop");
  const canKill = hasPermission("control.risk.kill_switch.activate");
  const canCreateJobs = hasPermission("control.jobs.create");
  const requiresMfa = Boolean(mfaStatus?.mfa_enabled);

  const tradingPayload = {
    mode,
    symbols: symbols.split(",").map((x) => x.trim().toUpperCase()).filter(Boolean),
    strategy,
    capital,
    ...(mfaCode ? { mfa_code: mfaCode } : {}),
  };

  return (
    <div className="space-y-6">
      <section className="grid gap-4 lg:grid-cols-3">
        <Card className="border-slate-200 bg-white/90 lg:col-span-2">
          <CardHeader>
            <CardTitle>Trading Engine Control</CardTitle>
            <CardDescription>Managed process orchestration for live/paper trading runtime.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 md:grid-cols-2">
              <label className="text-sm">
                Mode
                <select
                  className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3"
                  value={mode}
                  onChange={(e) => setMode(e.target.value as "live" | "paper" | "dry-run")}
                >
                  <option value="paper">paper</option>
                  <option value="dry-run">dry-run</option>
                  <option value="live">live</option>
                </select>
              </label>
              <label className="text-sm">
                Strategy
                <input
                  className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3"
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value)}
                />
              </label>
              <label className="text-sm md:col-span-2">
                Symbols (comma-separated)
                <input
                  className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3"
                  value={symbols}
                  onChange={(e) => setSymbols(e.target.value)}
                />
              </label>
              <label className="text-sm">
                Capital
                <input
                  className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3"
                  type="number"
                  value={capital}
                  onChange={(e) => setCapital(Number(e.target.value))}
                />
              </label>
              {requiresMfa ? (
                <label className="text-sm">
                  MFA Code
                  <input
                    className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3 font-mono"
                    value={mfaCode}
                    onChange={(e) => setMfaCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                    placeholder="6-digit TOTP"
                  />
                </label>
              ) : null}
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button className="gap-2 bg-emerald-700 text-white hover:bg-emerald-600" onClick={() => void startTrading(tradingPayload)} disabled={!canStart || (mode === "live" && requiresMfa && mfaCode.length !== 6)}>
                <Play size={16} />
                Start
              </Button>
              <Button variant="outline" className="gap-2" onClick={() => void restartTrading(tradingPayload)} disabled={!canRestart || (mode === "live" && requiresMfa && mfaCode.length !== 6)}>
                <RotateCcw size={16} />
                Restart
              </Button>
              <Button variant="destructive" className="gap-2" onClick={() => void stopTrading()} disabled={!canStop}>
                <Square size={16} />
                Stop
              </Button>
              {!canControlTrading ? <p className="text-xs text-rose-700">Role lacks trading control permissions.</p> : null}
            </div>

            <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
              <p className="text-sm font-medium text-amber-900">Emergency Kill Switch</p>
              <div className="mt-2 flex flex-wrap gap-2">
                <input
                  className="h-10 min-w-[240px] flex-1 rounded-lg border border-amber-300 bg-white px-3"
                  value={killReason}
                  onChange={(e) => setKillReason(e.target.value)}
                />
                <Button className="gap-2 bg-rose-700 text-white hover:bg-rose-600" onClick={() => void activateKillSwitch(killReason, mfaCode || undefined)} disabled={!canKill || (requiresMfa && mfaCode.length !== 6)}>
                  <Zap size={16} />
                  Trigger Kill Switch
                </Button>
              </div>
              {!canKill ? <p className="mt-2 text-xs text-rose-700">Role lacks kill switch permission.</p> : null}
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>Runtime State</CardTitle>
            <CardDescription>Current process and execution quality</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex items-center justify-between">
              <span>Engine</span>
              <Badge variant={tradingStatus?.running ? "success" : "outline"}>
                {tradingStatus?.running ? "Running" : "Stopped"}
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span>PID</span>
              <span className="font-mono">{tradingStatus?.pid ?? "--"}</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Fill Probability</span>
              <span>{((tca?.fill_probability ?? 0) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Execution Speed</span>
              <span>{(tca?.execution_speed_ms ?? 0).toFixed(1)} ms</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Slippage</span>
              <span>{(tca?.slippage_bps ?? 0).toFixed(2)} bps</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Arrival Delta</span>
              <span>{(executionQuality?.arrival_price_delta_bps ?? 0).toFixed(2)} bps</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Reject Rate</span>
              <span>{((executionQuality?.rejection_rate ?? 0) * 100).toFixed(2)}%</span>
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>Command Jobs</CardTitle>
            <CardDescription>Asynchronous operational workloads</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap gap-2">
              <Button variant="outline" onClick={() => void createJob("health", ["check", "--full"])} disabled={!canCreateJobs}>
                Run Health Check
              </Button>
              <Button variant="outline" onClick={() => void createJob("data", ["db-status"])} disabled={!canCreateJobs}>
                DB Status
              </Button>
              <Button variant="outline" onClick={() => void createJob("features", ["compute", "--symbols", "AAPL", "MSFT"])} disabled={!canCreateJobs}>
                Feature Compute
              </Button>
            </div>
            {runningJobs.length === 0 ? (
              <p className="text-sm text-slate-500">No running/queued job.</p>
            ) : (
              runningJobs.map((job) => (
                <div key={job.job_id} className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{job.command}</span>
                    <Badge variant="outline">{job.status}</Badge>
                  </div>
                  <p className="mt-1 truncate font-mono text-xs text-slate-600">{job.args.join(" ")}</p>
                </div>
              ))
            )}
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>Active Blotter</CardTitle>
            <CardDescription>Latest 40 orders</CardDescription>
          </CardHeader>
          <CardContent className="max-h-[420px] overflow-auto">
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 bg-white">
                <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                  <th className="py-2">Symbol</th>
                  <th className="py-2">Side</th>
                  <th className="py-2">Qty</th>
                  <th className="py-2">Status</th>
                  <th className="py-2">Updated</th>
                </tr>
              </thead>
              <tbody>
                {topOrders.map((order) => (
                  <tr key={order.order_id} className="border-b border-slate-100">
                    <td className="py-2 font-medium text-slate-800">{order.symbol}</td>
                    <td className="py-2">{order.side}</td>
                    <td className="py-2">{order.quantity}</td>
                    <td className="py-2">
                      <Badge variant={order.status === "FILLED" ? "success" : "outline"}>{order.status}</Badge>
                    </td>
                    <td className="py-2 text-xs text-slate-500">
                      {order.updated_at ? new Date(order.updated_at).toLocaleTimeString() : "--"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
