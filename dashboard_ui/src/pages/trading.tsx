import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { useShallow } from "zustand/react/shallow";
import { Activity, AlertOctagon, Gauge, Play, RefreshCcw, Square, Zap, Keyboard, ChevronRight, XCircle, LogOut } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore } from "@/lib/store";
import DataGrid from "@/components/ui/data-grid";
import { type ColumnDef } from "@tanstack/react-table";
import MiniGauge from "@/components/ui/mini-gauge";
import PanelLayout from "@/components/layout/PanelLayout";
import OrderBookWidget from "@/components/live/OrderBookWidget";
import LiveTapeWidget from "@/components/live/LiveTapeWidget";
import { toast } from "sonner"; // Assuming sonner is available for toasts

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
    orders,
    jobs,
    hasPermission,
    fetchTradingStatus,
    fetchTCA,
    fetchOrders,
    fetchJobs,
    startTrading,
    stopTrading,
    restartTrading,
    activateKillSwitch,
  } = useStore(useShallow((state) => ({
      tradingStatus: state.tradingStatus,
      tca: state.tca,
      orders: state.orders,
      jobs: state.jobs,
      hasPermission: state.hasPermission,
      fetchTradingStatus: state.fetchTradingStatus,
      fetchTCA: state.fetchTCA,
      fetchOrders: state.fetchOrders,
      fetchJobs: state.fetchJobs,
      startTrading: state.startTrading,
      stopTrading: state.stopTrading,
      restartTrading: state.restartTrading,
      activateKillSwitch: state.activateKillSwitch,
    })));

  const [mode, setMode] = useState<(typeof modeOptions)[number]>("paper");
  const [symbols, setSymbols] = useState("AAPL");
  const [killReason, setKillReason] = useState("");
  
  // Pro Trading Form
  const [orderQty, setOrderQty] = useState("100");

  useEffect(() => {
    void fetchTradingStatus();
    void fetchTCA();
    void fetchOrders();
    void fetchJobs();
    const timer = setInterval(() => {
      if (typeof document !== "undefined" && document.visibilityState !== "visible") return;
      void fetchTradingStatus();
      void fetchOrders();
      void fetchJobs();
    }, 8000);
    return () => clearInterval(timer);
  }, [fetchTradingStatus, fetchTCA, fetchOrders, fetchJobs]);

  const canControl = hasPermission("control.trading.start");
  const canKill = hasPermission("control.risk.kill_switch.activate");
  const recentJobs = useMemo(() => jobs.slice(0, 8), [jobs]);

  // Global Hotkey Listener for Pro Terminal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input field
      if (document.activeElement?.tagName === 'INPUT' || document.activeElement?.tagName === 'SELECT' || document.activeElement?.tagName === 'TEXTAREA') {
        return;
      }
      
      switch (e.key.toLowerCase()) {
        case 'b':
          toast.success(`Sent BUY MKT order for ${orderQty} ${symbols}`);
          e.preventDefault();
          break;
        case 's':
          toast.error(`Sent SELL MKT order for ${orderQty} ${symbols}`);
          e.preventDefault();
          break;
        case 'f':
          toast.warning(`Flattening all positions for ${symbols}`);
          e.preventDefault();
          break;
        case 'c':
          toast("Cancelled all open orders");
          e.preventDefault();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [symbols, orderQty]);

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-4">
      
      {/* Action Header */}
      <motion.section variants={fadeUp} className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-white/[0.08] bg-slate-950/80 p-4 shadow-lg backdrop-blur-sm">
        <div className="flex items-center gap-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-500/20 text-cyan-400">
             <Zap size={20} />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-100">Pro Trading Terminal</h1>
            <p className="text-xs text-slate-400">Low-latency Execution & High-Frequency Depth</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <select className="glass-select h-8 text-xs font-mono" value={mode} onChange={(e) => setMode(e.target.value as typeof mode)}>
            {modeOptions.map((m) => <option key={m} value={m}>{m.toUpperCase()}</option>)}
          </select>
          <input className="glass-input h-8 w-24 text-center text-xs font-mono font-bold uppercase" value={symbols} onChange={(e) => setSymbols(e.target.value)} placeholder="SYMBOL" />
          <Badge variant={tradingStatus?.running ? "success" : "outline"} className="h-8">
            <Activity size={14} className="mr-1" />
            {tradingStatus?.running ? "LIVE" : "IDLE"}
          </Badge>
        </div>
      </motion.section>

      {/* The Core Trading Three-Pane View */}
      <motion.section variants={fadeUp} className="h-[45vh] min-h-[400px]">
        <PanelLayout
          orientation="horizontal"
          storageKey="pro-terminal-panes"
          panels={[
            {
              id: "order-book",
              defaultSize: 30,
              minSize: 20,
              children: (
                <Card className="h-full border-white/[0.1] bg-slate-950/50">
                  <CardHeader className="p-3 border-b border-white/[0.06]">
                    <CardTitle className="text-sm flex items-center gap-2">
                       Level 2 Depth <Badge variant="outline" className="text-[9px]">L2</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-0 h-[calc(100%-48px)]">
                    <OrderBookWidget symbol={symbols.split(',')[0] || "AAPL"} />
                  </CardContent>
                </Card>
              ),
            },
            {
              id: "execution",
              defaultSize: 40,
              minSize: 30,
              children: (
                <Card className="h-full border-cyan-500/20 bg-slate-950/80 shadow-[0_0_30px_rgba(6,182,212,0.05)]">
                  <CardHeader className="p-3 border-b border-white/[0.06]">
                    <CardTitle className="text-sm flex items-center justify-between">
                       <span className="flex items-center gap-2"><Keyboard size={14} className="text-cyan-400" /> Hotkey Execution Pane</span>
                       <Badge variant="warning" className="animate-pulse">Armed</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-4 flex flex-col justify-between h-[calc(100%-48px)]">
                    <div className="space-y-4">
                      {/* Order Config */}
                      <div className="rounded-lg border border-white/[0.08] bg-white/[0.02] p-4 text-center">
                         <span className="text-[10px] uppercase tracking-widest text-slate-500 mb-2 block">Quantity</span>
                         <input 
                           type="number" 
                           className="w-full bg-transparent text-center font-mono text-4xl text-slate-100 outline-none" 
                           value={orderQty} 
                           onChange={(e) => setOrderQty(e.target.value)} 
                         />
                      </div>
                      
                      {/* Action Buttons */}
                      <div className="grid grid-cols-2 gap-3">
                        <Button 
                          className="h-16 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 border border-emerald-500/50 flex flex-col items-center justify-center gap-1"
                          onClick={() => toast.success(`MKT BUY ${orderQty} ${symbols}`)}
                        >
                           <span className="font-bold text-lg">MKT BUY</span>
                           <span className="text-[10px] opacity-70 border border-emerald-500/50 rounded px-1">B</span>
                        </Button>
                        <Button 
                          className="h-16 bg-rose-500/20 hover:bg-rose-500/30 text-rose-400 border border-rose-500/50 flex flex-col items-center justify-center gap-1"
                          onClick={() => toast.error(`MKT SELL ${orderQty} ${symbols}`)}
                        >
                           <span className="font-bold text-lg">MKT SELL</span>
                           <span className="text-[10px] opacity-70 border border-rose-500/50 rounded px-1">S</span>
                        </Button>
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                         <Button variant="outline" className="h-10 text-slate-300 border-white/[0.1] hover:bg-white/[0.05]" onClick={() => toast.warning("Flatten initiated")}>
                            <LogOut size={14} className="mr-2" /> Flatten <span className="ml-auto text-[10px] border border-white/[0.2] px-1 rounded">F</span>
                         </Button>
                         <Button variant="outline" className="h-10 text-amber-400 border-amber-500/30 hover:bg-amber-500/10" onClick={() => toast("Canceled all orders")}>
                            <XCircle size={14} className="mr-2" /> Cancel All <span className="ml-auto text-[10px] border border-amber-500/30 px-1 rounded">C</span>
                         </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ),
            },
            {
              id: "tape",
              defaultSize: 30,
              minSize: 20,
              children: (
                <Card className="h-full border-white/[0.1] bg-slate-950/50">
                  <CardHeader className="p-3 border-b border-white/[0.06]">
                    <CardTitle className="text-sm flex items-center gap-2">
                       Time & Sales <Badge variant="outline" className="text-[9px]">Tape</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-0 h-[calc(100%-48px)]">
                    <LiveTapeWidget symbol={symbols.split(',')[0] || "AAPL"} />
                  </CardContent>
                </Card>
              ),
            },
          ]}
        />
      </motion.section>

      {/* Legacy Engine Controls + Kill Switch */}
      <motion.div variants={fadeUp} className="grid md:grid-cols-3 gap-4">
        <Card className="md:col-span-2">
          <CardHeader className="p-4 pb-0">
            <CardTitle className="text-sm flex items-center gap-2">
              <Gauge size={14} className="text-cyan-400" /> Automated Engine Orbit
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 flex flex-wrap gap-2 items-end">
             <div className="flex-1 min-w-[200px]">
               <span className="text-[10px] uppercase text-slate-500 mb-1 block">Quick Symbols Config</span>
               <input className="glass-input h-8 w-full" value={symbols} onChange={(e) => setSymbols(e.target.value)} />
             </div>
             <Button size="sm" className="gap-2 h-8" disabled={!canControl || tradingStatus?.running === true} onClick={() => void startTrading({ mode, symbols: symbols.split(",").map((s) => s.trim()).filter(Boolean), strategy: "default", capital: 100000 })}>
               <Play size={14} /> Start
             </Button>
             <Button variant="secondary" size="sm" className="gap-2 h-8" disabled={!canControl || !tradingStatus?.running} onClick={() => void restartTrading({ mode, symbols: symbols.split(",").map((s) => s.trim()).filter(Boolean), strategy: "default", capital: 100000 })}>
               <RefreshCcw size={14} /> Restart
             </Button>
             <Button variant="destructive" size="sm" className="gap-2 h-8" disabled={!canControl || !tradingStatus?.running} onClick={() => void stopTrading()}>
               <Square size={14} /> Stop
             </Button>
          </CardContent>
        </Card>

        <Card className="danger-zone">
          <CardHeader className="p-4 pb-0">
            <CardTitle className="text-sm flex items-center gap-2 text-rose-400">
              <AlertOctagon size={14} /> Kill Switch Override
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 space-y-2">
            <input className="glass-input h-8 w-full text-xs" value={killReason} onChange={(e) => setKillReason(e.target.value)} placeholder="Reason: Flash crash..." />
            <Button variant="destructive" className="w-full h-8 text-xs gap-2" disabled={!canKill || !killReason} onClick={() => void activateKillSwitch(killReason, undefined)}>
              <Zap size={12} /> Engage Kill Switch
            </Button>
          </CardContent>
        </Card>
      </motion.div>

      {/* Blotters & Stats Below */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader className="p-4">
             <CardTitle className="text-sm flex items-center gap-2"><ChevronRight size={14}/> Open Orders</CardTitle>
          </CardHeader>
          <CardContent className="p-2 pt-0 h-[300px]">
             {orders.length === 0 ? (
                <p className="text-sm text-slate-500 p-4">No active orders.</p>
              ) : (
                <DataGrid data={orders} columns={orderColumns} maxHeight={280} exportFilename="orders" />
              )}
          </CardContent>
        </Card>

        {/* Execution Stats Mini Display */}
        <Card>
          <CardHeader className="p-4">
             <CardTitle className="text-sm flex items-center gap-2"><ChevronRight size={14}/> Real-time Execution Diagnostics</CardTitle>
          </CardHeader>
          <CardContent className="p-4">
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <div className="flex flex-col items-center gap-2 rounded-xl border border-emerald-500/20 bg-white/[0.02] p-3">
                <MiniGauge value={tca?.fill_probability ?? 0.94} size={50} label={`${(((tca?.fill_probability ?? 0.94) * 100)).toFixed(0)}%`} thresholds={[0.7, 0.9]} />
                <span className="text-[9px] uppercase tracking-wider text-slate-500">Fill Prob</span>
              </div>
              <div className="flex flex-col items-center gap-2 rounded-xl border border-amber-500/20 bg-white/[0.02] p-3">
                <MiniGauge value={Math.min(1, (tca?.slippage_bps ?? 2) / 10)} size={50} label={`${(tca?.slippage_bps ?? 2).toFixed(1)}`} thresholds={[0.3, 0.6]} />
                <span className="text-[9px] uppercase tracking-wider text-slate-500">Slippage (bps)</span>
              </div>
              <div className="flex flex-col items-center gap-2 rounded-xl border border-cyan-500/20 bg-white/[0.02] p-3">
                <MiniGauge value={0.12} size={50} label="12ms" thresholds={[0.4, 0.7]} />
                <span className="text-[9px] uppercase tracking-wider text-slate-500">Avg Latency</span>
              </div>
            </div>
            {/* Quick Jobs Log */}
            <div className="mt-4 space-y-2">
               <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">Recent Orchestration Logs</span>
               {recentJobs.slice(0, 3).map((job) => (
                  <div key={job.job_id} className="flex justify-between items-center bg-white/[0.01] border border-white/[0.05] p-2 rounded text-[10px]">
                    <span className="text-slate-300 font-mono">{job.command}</span>
                    <Badge variant="outline" className="scale-75 origin-right">{job.status}</Badge>
                  </div>
               ))}
               {recentJobs.length === 0 && <div className="text-[10px] text-slate-600">No jobs recorded.</div>}
            </div>
          </CardContent>
        </Card>

      </motion.div>

    </motion.div>
  );
}
