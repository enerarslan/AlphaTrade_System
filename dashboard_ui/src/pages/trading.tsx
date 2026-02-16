import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Activity, AlertOctagon, Gauge, Play, RefreshCcw, Square, Timer, Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore } from "@/lib/store";
import DataGrid from "@/components/ui/data-grid";
import { type ColumnDef } from "@tanstack/react-table";
import MiniGauge from "@/components/ui/mini-gauge";
import PanelLayout from "@/components/layout/PanelLayout";

const stagger = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } },
};
const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" as const } },
} as const;

const modeOptions = ["paper", "live", "dry-run"] as const;

type OrderRow = {
  order_id: string;
  symbol: string;
  side: string;
  order_type: string;
  quantity: number;
  filled_qty: number;
  limit_price: number | null;
  status: string;
  created_at: string;
};

const orderColumns: ColumnDef<OrderRow, unknown>[] = [
  { accessorKey: "symbol", header: "Symbol", size: 80 },
  {
    accessorKey: "side",
    header: "Side",
    size: 60,
    cell: ({ getValue }) => {
      const v = getValue() as string;
      return <span className={v === "BUY" ? "text-emerald-400" : "text-rose-400"}>{v}</span>;
    },
  },
  { accessorKey: "quantity", header: "Qty", size: 60 },
  {
    accessorKey: "limit_price",
    header: "Price",
    size: 70,
    cell: ({ getValue }) => {
      const v = getValue() as number | null;
      return v != null ? `$${v.toFixed(2)}` : "MKT";
    },
  },
  { accessorKey: "order_type", header: "Type", size: 60 },
  {
    accessorKey: "status",
    header: "Status",
    size: 70,
    cell: ({ getValue }) => {
      const v = getValue() as string;
      const variant = v === "FILLED" ? "success" : v === "CANCELLED" ? "error" : "outline";
      return <Badge variant={variant}>{v}</Badge>;
    },
  },
  { accessorKey: "created_at", header: "Time", size: 100,
    cell: ({ getValue }) => new Date(getValue() as string).toLocaleTimeString(),
  },
];

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

      {/* Execution Quality — with MiniGauges */}
      <motion.div variants={fadeUp}>
        <Card>
          <CardHeader>
            <CardTitle>Execution Quality</CardTitle>
            <CardDescription>TCA metrics and execution analytics.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
              <div className="flex flex-col items-center gap-2 rounded-xl border border-emerald-500/20 bg-white/[0.02] p-3">
                <MiniGauge value={tca?.fill_probability ?? 0} size={48} label={`${((tca?.fill_probability ?? 0) * 100).toFixed(0)}%`} thresholds={[0.7, 0.9]} />
                <span className="text-[10px] uppercase tracking-wider text-slate-500">Fill Prob</span>
              </div>
              <div className="flex flex-col items-center gap-2 rounded-xl border border-amber-500/20 bg-white/[0.02] p-3">
                <MiniGauge value={Math.min(1, (tca?.slippage_bps ?? 0) / 10)} size={48} label={`${(tca?.slippage_bps ?? 0).toFixed(1)}`} thresholds={[0.3, 0.6]} />
                <span className="text-[10px] uppercase tracking-wider text-slate-500">Slippage bps</span>
              </div>
              <div className="flex flex-col items-center gap-2 rounded-xl border border-amber-500/20 bg-white/[0.02] p-3">
                <MiniGauge value={Math.min(1, (tca?.market_impact_bps ?? 0) / 10)} size={48} label={`${(tca?.market_impact_bps ?? 0).toFixed(1)}`} thresholds={[0.3, 0.6]} />
                <span className="text-[10px] uppercase tracking-wider text-slate-500">Mkt Impact</span>
              </div>
              <div className="flex flex-col items-center gap-2 rounded-xl border border-cyan-500/20 bg-white/[0.02] p-3">
                <MiniGauge value={Math.min(1, Math.abs(executionQuality?.arrival_price_delta_bps ?? 0) / 5)} size={48} label={`${(executionQuality?.arrival_price_delta_bps ?? 0).toFixed(1)}`} thresholds={[0.4, 0.7]} />
                <span className="text-[10px] uppercase tracking-wider text-slate-500">Arrival Δ</span>
              </div>
              <div className="flex flex-col items-center gap-2 rounded-xl border border-cyan-500/20 bg-white/[0.02] p-3">
                <MiniGauge value={Math.min(1, (Object.values(executionQuality?.latency_distribution_ms ?? {})[0] ?? 0) / 100)} size={48} label={`${(Object.values(executionQuality?.latency_distribution_ms ?? {})[0]?.toFixed(0) ?? "--")}ms`} thresholds={[0.5, 0.8]} />
                <span className="text-[10px] uppercase tracking-wider text-slate-500">Latency</span>
              </div>
              <div className="flex flex-col items-center gap-2 rounded-xl border border-rose-500/20 bg-white/[0.02] p-3">
                <MiniGauge value={executionQuality?.rejection_rate ?? 0} size={48} label={`${((executionQuality?.rejection_rate ?? 0) * 100).toFixed(1)}%`} thresholds={[0.05, 0.15]} />
                <span className="text-[10px] uppercase tracking-wider text-slate-500">Reject Rate</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Order Blotter + Command Jobs — Resizable Panels */}
      <motion.div variants={fadeUp}>
        <PanelLayout
          orientation="horizontal"
          storageKey="trading-blotter-jobs"
          panels={[
            {
              id: "blotter",
              defaultSize: 60,
              minSize: 35,
              children: (
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle>Order Blotter</CardTitle>
                    <CardDescription>{orders.length} total orders. Sortable, filterable, exportable.</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {orders.length === 0 ? (
                      <p className="text-sm text-slate-500">No orders.</p>
                    ) : (
                      <DataGrid
                        data={orders}
                        columns={orderColumns}
                        maxHeight={400}
                        exportFilename="orders"
                      />
                    )}
                  </CardContent>
                </Card>
              ),
            },
            {
              id: "jobs",
              defaultSize: 40,
              minSize: 25,
              children: (
                <Card className="h-full">
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
              ),
            },
          ]}
        />
      </motion.div>
    </motion.div>
  );
}
