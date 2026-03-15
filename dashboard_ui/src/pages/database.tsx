import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  Database,
  BarChart3,
  TrendingUp,
  ArrowLeftRight,
  Zap,
  ShieldAlert,
  BookOpen,
  DollarSign,
  RefreshCw,
} from "lucide-react";
import { type ColumnDef } from "@tanstack/react-table";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import DataGrid from "@/components/ui/data-grid";
import { api } from "@/lib/api";

// ─── Motion variants ────────────────────────────────────────────────────────

const stagger = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.07 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: "easeOut" as const } },
} as const;

const darkTooltipStyle = {
  contentStyle: {
    background: "rgba(15,23,42,0.95)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: "8px",
    color: "#e2e8f0",
    fontSize: 12,
  },
  itemStyle: { color: "#e2e8f0" },
  labelStyle: { color: "#94a3b8" },
};

// ─── Tab definitions ─────────────────────────────────────────────────────────

const tabs = [
  { id: "overview", label: "Overview", icon: Database },
  { id: "market", label: "Market Data", icon: BarChart3 },
  { id: "performance", label: "Performance", icon: TrendingUp },
  { id: "trades", label: "Trade Log", icon: ArrowLeftRight },
  { id: "signals", label: "Signals", icon: Zap },
  { id: "risk", label: "Risk Events", icon: ShieldAlert },
  { id: "reference", label: "Reference Data", icon: BookOpen },
  { id: "earnings", label: "Earnings", icon: DollarSign },
] as const;

type TabId = (typeof tabs)[number]["id"];

// ─── Table category mapping ───────────────────────────────────────────────────

const categories: Record<string, string[]> = {
  "Market Data": ["ohlcv_bars", "features", "stock_quotes", "stock_trades"],
  Reference: [
    "security_master",
    "corporate_actions",
    "fundamental_snapshots",
    "earnings_events",
    "sec_filings",
    "macro_observations",
    "macro_vintage_observations",
    "news_articles",
  ],
  Microstructure: ["short_sale_volumes", "fails_to_deliver"],
  Trading: ["orders", "trades", "positions", "position_history"],
  Signals: ["signals", "model_predictions"],
  Performance: ["daily_performance", "trade_log"],
  System: ["system_logs", "alerts", "risk_events"],
};

const categoryColors: Record<string, string> = {
  "Market Data": "text-cyan-400 border-cyan-500/20 bg-cyan-500/10",
  Reference: "text-amber-400 border-amber-500/20 bg-amber-500/10",
  Microstructure: "text-violet-400 border-violet-500/20 bg-violet-500/10",
  Trading: "text-emerald-400 border-emerald-500/20 bg-emerald-500/10",
  Signals: "text-blue-400 border-blue-500/20 bg-blue-500/10",
  Performance: "text-teal-400 border-teal-500/20 bg-teal-500/10",
  System: "text-slate-400 border-slate-500/20 bg-slate-500/10",
};

function getCategoryForTable(tableName: string): string {
  for (const [cat, tables] of Object.entries(categories)) {
    if (tables.includes(tableName)) return cat;
  }
  return "System";
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

function fmt(n: number | null | undefined, decimals = 2): string {
  if (n == null || isNaN(n)) return "—";
  return n.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

function fmtPct(n: number | null | undefined): string {
  if (n == null || isNaN(n)) return "—";
  return `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`;
}

function PnlCell({ value }: { value: number | null }) {
  const color =
    value == null ? "text-slate-500" : value >= 0 ? "text-emerald-400" : "text-rose-400";
  return <span className={color}>{value != null ? `$${fmt(value)}` : "—"}</span>;
}

function PctCell({ value }: { value: number | null }) {
  const color =
    value == null ? "text-slate-500" : value >= 0 ? "text-emerald-400" : "text-rose-400";
  return <span className={color}>{value != null ? fmtPct(value) : "—"}</span>;
}

// ─── Inline filter bar ────────────────────────────────────────────────────────

function FilterBar({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-4 flex flex-wrap items-center gap-2">
      {children}
    </div>
  );
}

function TextInput({
  value,
  onChange,
  placeholder,
  className = "",
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  className?: string;
}) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className={`rounded-lg border border-white/[0.08] bg-white/[0.03] px-3 py-1.5 text-xs text-slate-300 outline-none placeholder:text-slate-600 focus:border-teal-500/40 focus:bg-white/[0.05] transition-colors ${className}`}
    />
  );
}

function SelectInput({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  placeholder?: string;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-lg border border-white/[0.08] bg-slate-900 px-3 py-1.5 text-xs text-slate-300 outline-none focus:border-teal-500/40 transition-colors"
    >
      {placeholder && <option value="">{placeholder}</option>}
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

// ─── Tab 1: Overview ──────────────────────────────────────────────────────────

type TableInfo = {
  table_name: string;
  total_size: string;
  size_bytes: number;
  row_count: number;
};

function OverviewTab() {
  const [tables, setTables] = useState<TableInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get<{ tables: TableInfo[]; total_tables: number }>("/db/tables");
      const sorted = [...(res.data.tables ?? [])].sort((a, b) => b.size_bytes - a.size_bytes);
      setTables(sorted);
    } catch {
      setError("Failed to load table inventory");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetch();
  }, [fetch]);

  const totalSize = tables.reduce((sum, t) => sum + t.size_bytes, 0);
  const totalRows = tables.reduce((sum, t) => sum + t.row_count, 0);

  const formatBytes = (bytes: number) => {
    if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
    if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
    return `${bytes} B`;
  };

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-4">
      {/* Summary row */}
      <motion.div variants={fadeUp} className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { label: "Total Tables", value: String(tables.length) },
          { label: "Total Rows", value: totalRows.toLocaleString() },
          { label: "Total Size", value: formatBytes(totalSize) },
          { label: "Categories", value: String(Object.keys(categories).length) },
        ].map((kpi) => (
          <div
            key={kpi.label}
            className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-3"
          >
            <p className="text-[9px] font-semibold uppercase tracking-[0.2em] text-slate-500">
              {kpi.label}
            </p>
            <p className="mt-1 text-lg font-bold text-slate-100">{kpi.value}</p>
          </div>
        ))}
      </motion.div>

      {/* Action row */}
      <motion.div variants={fadeUp} className="flex items-center justify-between">
        <p className="text-xs text-slate-500">
          {loading ? "Loading table inventory..." : error ? error : `${tables.length} tables in TimescaleDB`}
        </p>
        <Button size="sm" variant="outline" onClick={() => void fetch()} disabled={loading}>
          <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
          Refresh
        </Button>
      </motion.div>

      {/* Table cards grid */}
      <motion.div variants={fadeUp} className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {tables.map((t) => {
          const cat = getCategoryForTable(t.table_name);
          const colorClass = categoryColors[cat] ?? categoryColors["System"];
          return (
            <div
              key={t.table_name}
              className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-3 transition-colors hover:bg-white/[0.04]"
            >
              <div className="flex items-start justify-between gap-2">
                <p className="font-mono text-xs font-semibold text-slate-200">{t.table_name}</p>
                <span
                  className={`shrink-0 rounded-md border px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide ${colorClass}`}
                >
                  {cat}
                </span>
              </div>
              <div className="mt-2 flex items-center gap-3">
                <span className="text-xs text-slate-400">{t.row_count.toLocaleString()} rows</span>
                <span className="text-slate-600">·</span>
                <span className="text-xs text-slate-500">{t.total_size}</span>
              </div>
            </div>
          );
        })}
      </motion.div>
    </motion.div>
  );
}

// ─── Tab 2: Market Data ────────────────────────────────────────────────────────

type OhlcvRow = {
  symbol: string;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

const ohlcvColumns: ColumnDef<OhlcvRow, unknown>[] = [
  { accessorKey: "timestamp", header: "Timestamp", cell: (i) => new Date(i.getValue() as string).toLocaleString() },
  { accessorKey: "symbol", header: "Symbol" },
  { accessorKey: "open", header: "Open", cell: (i) => fmt(i.getValue() as number) },
  { accessorKey: "high", header: "High", cell: (i) => fmt(i.getValue() as number) },
  { accessorKey: "low", header: "Low", cell: (i) => fmt(i.getValue() as number) },
  { accessorKey: "close", header: "Close", cell: (i) => fmt(i.getValue() as number) },
  {
    accessorKey: "volume",
    header: "Volume",
    cell: (i) => (i.getValue() as number).toLocaleString(),
  },
];

const timeframeOptions = [
  { value: "1Min", label: "1 Min" },
  { value: "5Min", label: "5 Min" },
  { value: "15Min", label: "15 Min" },
  { value: "1Hour", label: "1 Hour" },
  { value: "1Day", label: "1 Day" },
];

function MarketDataTab() {
  const [symbol, setSymbol] = useState("AAPL");
  const [timeframe, setTimeframe] = useState("15Min");
  const [limit, setLimit] = useState("200");
  const [data, setData] = useState<OhlcvRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    if (!symbol.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.get<{ data: OhlcvRow[]; count: number }>("/db/ohlcv", {
        params: { symbol: symbol.trim().toUpperCase(), timeframe, limit: Number(limit) || 200 },
      });
      setData(res.data.data ?? []);
    } catch {
      setError("Failed to fetch OHLCV data");
    } finally {
      setLoading(false);
    }
  }, [symbol, timeframe, limit]);

  const chartData = data.slice(-100).map((row) => ({
    time: new Date(row.timestamp).toLocaleDateString(),
    close: row.close,
  }));

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-4">
      <motion.div variants={fadeUp}>
        <FilterBar>
          <TextInput value={symbol} onChange={setSymbol} placeholder="Symbol (e.g. AAPL)" className="w-36" />
          <SelectInput value={timeframe} onChange={setTimeframe} options={timeframeOptions} />
          <TextInput value={limit} onChange={setLimit} placeholder="Limit" className="w-20" />
          <Button size="sm" onClick={() => void fetchData()} disabled={loading}>
            {loading ? <RefreshCw size={12} className="animate-spin" /> : null}
            Fetch
          </Button>
        </FilterBar>
        {error && <p className="mb-3 text-xs text-rose-400">{error}</p>}
      </motion.div>

      {chartData.length > 0 && (
        <motion.div variants={fadeUp}>
          <Card className="border-white/[0.06] bg-white/[0.02]">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-teal-400">
                {symbol.toUpperCase()} Close Price — {timeframe}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis
                    dataKey="time"
                    tick={{ fill: "#64748b", fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fill: "#64748b", fontSize: 10 }}
                    tickLine={false}
                    axisLine={false}
                    domain={["auto", "auto"]}
                    tickFormatter={(v: number) => `$${v.toFixed(0)}`}
                  />
                  <Tooltip {...darkTooltipStyle} formatter={(v: number | undefined) => [`$${(v ?? 0).toFixed(2)}`, "Close"]} />
                  <Line
                    type="monotone"
                    dataKey="close"
                    stroke="#2dd4bf"
                    strokeWidth={1.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>
      )}

      <motion.div variants={fadeUp}>
        <DataGrid
          data={data}
          columns={ohlcvColumns}
          maxHeight={420}
          exportFilename={`ohlcv_${symbol}_${timeframe}`}
        />
      </motion.div>
    </motion.div>
  );
}

// ─── Tab 3: Performance ────────────────────────────────────────────────────────

type DailyPerfRow = {
  date: string;
  starting_equity: number;
  ending_equity: number;
  pnl: number;
  pnl_percent: number;
  trades_count: number;
  win_count: number;
  loss_count: number;
};

function PerformanceTab() {
  const [data, setData] = useState<DailyPerfRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get<{ data: DailyPerfRow[]; count: number }>("/db/daily-performance", {
        params: { limit: 365 },
      });
      setData(res.data.data ?? []);
    } catch {
      setError("Failed to fetch performance data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchData();
  }, [fetchData]);

  const totalPnl = data.reduce((s, d) => s + (d.pnl ?? 0), 0);
  const winDays = data.filter((d) => d.pnl > 0).length;
  const avgDailyPnl = data.length > 0 ? totalPnl / data.length : 0;
  const bestDay = data.length > 0 ? Math.max(...data.map((d) => d.pnl)) : 0;
  const worstDay = data.length > 0 ? Math.min(...data.map((d) => d.pnl)) : 0;

  const barData = data.slice(-60).map((d) => ({ date: d.date, pnl: d.pnl }));

  const equityData = data.slice(-90).map((d) => ({ date: d.date, equity: d.ending_equity }));

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-4">
      <motion.div variants={fadeUp} className="flex items-center justify-between">
        <p className="text-xs text-slate-500">
          {loading ? "Loading..." : error ?? `${data.length} trading days`}
        </p>
        <Button size="sm" variant="outline" onClick={() => void fetchData()} disabled={loading}>
          <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
          Refresh
        </Button>
      </motion.div>

      {/* KPI row */}
      <motion.div variants={fadeUp} className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {[
          { label: "Total Days", value: String(data.length) },
          { label: "Total PnL", value: `$${fmt(totalPnl)}`, pos: totalPnl >= 0 },
          { label: "Avg Daily", value: `$${fmt(avgDailyPnl)}`, pos: avgDailyPnl >= 0 },
          { label: "Win Days", value: String(winDays), pos: true },
          { label: "Best Day", value: `$${fmt(bestDay)}`, pos: true },
          { label: "Worst Day", value: `$${fmt(worstDay)}`, pos: false },
        ].map((kpi) => (
          <div
            key={kpi.label}
            className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-3"
          >
            <p className="text-[9px] font-semibold uppercase tracking-[0.2em] text-slate-500">
              {kpi.label}
            </p>
            <p
              className={`mt-1 text-sm font-bold ${
                kpi.pos === undefined
                  ? "text-slate-100"
                  : kpi.pos
                    ? "text-emerald-400"
                    : "text-rose-400"
              }`}
            >
              {kpi.value}
            </p>
          </div>
        ))}
      </motion.div>

      {/* Daily PnL bar chart */}
      {barData.length > 0 && (
        <motion.div variants={fadeUp}>
          <Card className="border-white/[0.06] bg-white/[0.02]">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-teal-400">Daily PnL (last 60 days)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={barData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 9 }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                  <YAxis tick={{ fill: "#64748b", fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
                  <Tooltip {...darkTooltipStyle} formatter={(v: number | undefined) => [`$${fmt(v ?? 0)}`, "PnL"]} />
                  <Bar dataKey="pnl" radius={[2, 2, 0, 0]}>
                    {barData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.pnl >= 0 ? "#10b981" : "#f43f5e"} fillOpacity={0.75} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Cumulative equity chart */}
      {equityData.length > 0 && (
        <motion.div variants={fadeUp}>
          <Card className="border-white/[0.06] bg-white/[0.02]">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-teal-400">Cumulative Equity (last 90 days)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={equityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 9 }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                  <YAxis tick={{ fill: "#64748b", fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
                  <Tooltip {...darkTooltipStyle} formatter={(v: number | undefined) => [`$${fmt(v ?? 0)}`, "Equity"]} />
                  <Line type="monotone" dataKey="equity" stroke="#2dd4bf" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {error && <motion.div variants={fadeUp}><p className="text-xs text-rose-400">{error}</p></motion.div>}
    </motion.div>
  );
}

// ─── Tab 4: Trade Log ─────────────────────────────────────────────────────────

type TradeLogRow = {
  trade_id: string;
  symbol: string;
  side: string;
  entry_time: string;
  exit_time: string;
  entry_price: number;
  exit_price: number;
  pnl: number;
  pnl_percent: number;
  strategy: string;
};

const tradeColumns: ColumnDef<TradeLogRow, unknown>[] = [
  { accessorKey: "symbol", header: "Symbol" },
  {
    accessorKey: "side",
    header: "Side",
    cell: (i) => {
      const v = i.getValue() as string;
      return (
        <span className={v === "BUY" || v === "LONG" ? "text-emerald-400" : "text-rose-400"}>
          {v}
        </span>
      );
    },
  },
  { accessorKey: "entry_time", header: "Entry Time", cell: (i) => new Date(i.getValue() as string).toLocaleString() },
  { accessorKey: "exit_time", header: "Exit Time", cell: (i) => i.getValue() ? new Date(i.getValue() as string).toLocaleString() : "—" },
  { accessorKey: "entry_price", header: "Entry $", cell: (i) => `$${fmt(i.getValue() as number)}` },
  { accessorKey: "exit_price", header: "Exit $", cell: (i) => i.getValue() ? `$${fmt(i.getValue() as number)}` : "—" },
  { accessorKey: "pnl", header: "PnL", cell: (i) => <PnlCell value={i.getValue() as number} /> },
  { accessorKey: "pnl_percent", header: "PnL%", cell: (i) => <PctCell value={i.getValue() as number} /> },
  { accessorKey: "strategy", header: "Strategy" },
];

function TradeLogTab() {
  const [symbolFilter, setSymbolFilter] = useState("");
  const [strategyFilter, setStrategyFilter] = useState("");
  const [data, setData] = useState<TradeLogRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get<{ data: TradeLogRow[]; count: number }>("/db/trade-log", {
        params: {
          symbol: symbolFilter.trim().toUpperCase() || undefined,
          strategy: strategyFilter.trim() || undefined,
          limit: 200,
        },
      });
      setData(res.data.data ?? []);
    } catch {
      setError("Failed to fetch trade log");
    } finally {
      setLoading(false);
    }
  }, [symbolFilter, strategyFilter]);

  useEffect(() => {
    void fetchData();
  }, [fetchData]);

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-4">
      <motion.div variants={fadeUp}>
        <FilterBar>
          <TextInput value={symbolFilter} onChange={setSymbolFilter} placeholder="Symbol" className="w-28" />
          <TextInput value={strategyFilter} onChange={setStrategyFilter} placeholder="Strategy" className="w-32" />
          <Button size="sm" onClick={() => void fetchData()} disabled={loading}>
            {loading ? <RefreshCw size={12} className="animate-spin" /> : null}
            Fetch
          </Button>
        </FilterBar>
        {error && <p className="mb-3 text-xs text-rose-400">{error}</p>}
      </motion.div>
      <motion.div variants={fadeUp}>
        <DataGrid data={data} columns={tradeColumns} maxHeight={480} exportFilename="trade_log" />
      </motion.div>
    </motion.div>
  );
}

// ─── Tab 5: Signals History ───────────────────────────────────────────────────

type SignalRow = {
  signal_id: string;
  timestamp: string;
  symbol: string;
  direction: string;
  strength: number;
  confidence: number;
  model_source: string;
};

const signalColumns: ColumnDef<SignalRow, unknown>[] = [
  { accessorKey: "timestamp", header: "Timestamp", cell: (i) => new Date(i.getValue() as string).toLocaleString() },
  { accessorKey: "symbol", header: "Symbol" },
  {
    accessorKey: "direction",
    header: "Direction",
    cell: (i) => {
      const v = i.getValue() as string;
      return (
        <span className={v === "LONG" || v === "BUY" ? "text-emerald-400" : v === "SHORT" || v === "SELL" ? "text-rose-400" : "text-slate-400"}>
          {v}
        </span>
      );
    },
  },
  { accessorKey: "strength", header: "Strength", cell: (i) => fmt(i.getValue() as number, 4) },
  { accessorKey: "confidence", header: "Confidence", cell: (i) => `${((i.getValue() as number) * 100).toFixed(1)}%` },
  { accessorKey: "model_source", header: "Model Source" },
];

function SignalsTab() {
  const [symbolFilter, setSymbolFilter] = useState("");
  const [modelFilter, setModelFilter] = useState("");
  const [data, setData] = useState<SignalRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get<{ data: SignalRow[]; count: number }>("/db/signals-history", {
        params: {
          symbol: symbolFilter.trim().toUpperCase() || undefined,
          model_source: modelFilter.trim() || undefined,
          limit: 200,
        },
      });
      setData(res.data.data ?? []);
    } catch {
      setError("Failed to fetch signals history");
    } finally {
      setLoading(false);
    }
  }, [symbolFilter, modelFilter]);

  useEffect(() => {
    void fetchData();
  }, [fetchData]);

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-4">
      <motion.div variants={fadeUp}>
        <FilterBar>
          <TextInput value={symbolFilter} onChange={setSymbolFilter} placeholder="Symbol" className="w-28" />
          <TextInput value={modelFilter} onChange={setModelFilter} placeholder="Model Source" className="w-36" />
          <Button size="sm" onClick={() => void fetchData()} disabled={loading}>
            {loading ? <RefreshCw size={12} className="animate-spin" /> : null}
            Fetch
          </Button>
        </FilterBar>
        {error && <p className="mb-3 text-xs text-rose-400">{error}</p>}
      </motion.div>
      <motion.div variants={fadeUp}>
        <DataGrid data={data} columns={signalColumns} maxHeight={480} exportFilename="signals_history" />
      </motion.div>
    </motion.div>
  );
}

// ─── Tab 6: Risk Events ───────────────────────────────────────────────────────

type RiskEventRow = {
  id: string;
  timestamp: string;
  event_type: string;
  severity: string;
  symbol: string;
  description: string;
  resolved: boolean;
};

function severityVariant(s: string): "error" | "warning" | "outline" | "success" {
  switch (s?.toUpperCase()) {
    case "CRITICAL": return "error";
    case "HIGH": return "error";
    case "MEDIUM": return "warning";
    case "LOW": return "outline";
    default: return "outline";
  }
}

const eventTypeOptions = [
  { value: "POSITION_LIMIT", label: "Position Limit" },
  { value: "DRAWDOWN", label: "Drawdown" },
  { value: "KILL_SWITCH", label: "Kill Switch" },
  { value: "VAR_BREACH", label: "VaR Breach" },
  { value: "MARGIN_CALL", label: "Margin Call" },
];

const severityOptions = [
  { value: "CRITICAL", label: "Critical" },
  { value: "HIGH", label: "High" },
  { value: "MEDIUM", label: "Medium" },
  { value: "LOW", label: "Low" },
];

function RiskEventsTab() {
  const [eventTypeFilter, setEventTypeFilter] = useState("");
  const [severityFilter, setSeverityFilter] = useState("");
  const [data, setData] = useState<RiskEventRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get<{ data: RiskEventRow[]; count: number }>("/db/risk-events", {
        params: {
          event_type: eventTypeFilter || undefined,
          severity: severityFilter || undefined,
          limit: 200,
        },
      });
      setData(res.data.data ?? []);
    } catch {
      setError("Failed to fetch risk events");
    } finally {
      setLoading(false);
    }
  }, [eventTypeFilter, severityFilter]);

  useEffect(() => {
    void fetchData();
  }, [fetchData]);

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-4">
      <motion.div variants={fadeUp}>
        <FilterBar>
          <SelectInput
            value={eventTypeFilter}
            onChange={setEventTypeFilter}
            options={eventTypeOptions}
            placeholder="All Event Types"
          />
          <SelectInput
            value={severityFilter}
            onChange={setSeverityFilter}
            options={severityOptions}
            placeholder="All Severities"
          />
          <Button size="sm" onClick={() => void fetchData()} disabled={loading}>
            {loading ? <RefreshCw size={12} className="animate-spin" /> : null}
            Fetch
          </Button>
        </FilterBar>
        {error && <p className="mb-3 text-xs text-rose-400">{error}</p>}
      </motion.div>

      <motion.div variants={fadeUp} className="space-y-2">
        {data.length === 0 && !loading && (
          <p className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-6 text-center text-xs text-slate-500">
            No risk events found.
          </p>
        )}
        {data.map((evt) => (
          <div
            key={evt.id}
            className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-3 transition-colors hover:bg-white/[0.04]"
          >
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div className="flex items-center gap-2">
                <Badge variant={severityVariant(evt.severity)}>{evt.severity}</Badge>
                <span className="font-mono text-xs font-semibold text-slate-200">{evt.event_type}</span>
                {evt.symbol && (
                  <span className="rounded bg-white/[0.04] px-1.5 py-0.5 font-mono text-[10px] text-slate-400">
                    {evt.symbol}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Badge variant={evt.resolved ? "success" : "warning"}>
                  {evt.resolved ? "Resolved" : "Active"}
                </Badge>
                <span className="text-[10px] text-slate-500">
                  {new Date(evt.timestamp).toLocaleString()}
                </span>
              </div>
            </div>
            {evt.description && (
              <p className="mt-1.5 text-xs text-slate-400">{evt.description}</p>
            )}
          </div>
        ))}
      </motion.div>
    </motion.div>
  );
}

// ─── Tab 7: Reference Data ────────────────────────────────────────────────────

type SecurityRow = {
  symbol: string;
  name: string;
  exchange: string;
  asset_type: string;
  status: string;
  sector: string;
  industry: string;
  market_cap: number;
};

const securityColumns: ColumnDef<SecurityRow, unknown>[] = [
  { accessorKey: "symbol", header: "Symbol" },
  { accessorKey: "name", header: "Name" },
  { accessorKey: "exchange", header: "Exchange" },
  { accessorKey: "sector", header: "Sector" },
  { accessorKey: "industry", header: "Industry" },
  {
    accessorKey: "market_cap",
    header: "Market Cap",
    cell: (i) => {
      const v = i.getValue() as number;
      if (!v) return "—";
      if (v >= 1e12) return `$${(v / 1e12).toFixed(1)}T`;
      if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
      if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
      return `$${v.toLocaleString()}`;
    },
  },
  {
    accessorKey: "status",
    header: "Status",
    cell: (i) => {
      const v = i.getValue() as string;
      return (
        <Badge variant={v === "active" || v === "ACTIVE" ? "success" : "outline"}>{v}</Badge>
      );
    },
  },
];

const sectorOptions = [
  { value: "Technology", label: "Technology" },
  { value: "Financials", label: "Financials" },
  { value: "Healthcare", label: "Healthcare" },
  { value: "Consumer Discretionary", label: "Consumer Disc." },
  { value: "Communication Services", label: "Communication" },
  { value: "Industrials", label: "Industrials" },
  { value: "Energy", label: "Energy" },
  { value: "Materials", label: "Materials" },
  { value: "Real Estate", label: "Real Estate" },
  { value: "Utilities", label: "Utilities" },
];

function ReferenceDataTab() {
  const [sectorFilter, setSectorFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [data, setData] = useState<SecurityRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get<{ data: SecurityRow[]; count: number }>("/db/security-master", {
        params: {
          sector: sectorFilter || undefined,
          status: statusFilter || undefined,
          limit: 100,
        },
      });
      setData(res.data.data ?? []);
    } catch {
      setError("Failed to fetch security master");
    } finally {
      setLoading(false);
    }
  }, [sectorFilter, statusFilter]);

  useEffect(() => {
    void fetchData();
  }, [fetchData]);

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-4">
      <motion.div variants={fadeUp}>
        <FilterBar>
          <SelectInput
            value={sectorFilter}
            onChange={setSectorFilter}
            options={sectorOptions}
            placeholder="All Sectors"
          />
          <SelectInput
            value={statusFilter}
            onChange={setStatusFilter}
            options={[{ value: "active", label: "Active" }, { value: "inactive", label: "Inactive" }]}
            placeholder="All Statuses"
          />
          <Button size="sm" onClick={() => void fetchData()} disabled={loading}>
            {loading ? <RefreshCw size={12} className="animate-spin" /> : null}
            Fetch
          </Button>
        </FilterBar>
        {error && <p className="mb-3 text-xs text-rose-400">{error}</p>}
      </motion.div>
      <motion.div variants={fadeUp}>
        <DataGrid data={data} columns={securityColumns} maxHeight={480} exportFilename="security_master" />
      </motion.div>
    </motion.div>
  );
}

// ─── Tab 8: Earnings ──────────────────────────────────────────────────────────

type EarningsRow = {
  symbol: string;
  fiscal_date_ending: string;
  reported_date: string;
  reported_eps: number;
  estimated_eps: number;
  surprise: number;
  surprise_pct: number;
};

const earningsColumns: ColumnDef<EarningsRow, unknown>[] = [
  { accessorKey: "symbol", header: "Symbol" },
  { accessorKey: "fiscal_date_ending", header: "Fiscal Date" },
  { accessorKey: "reported_date", header: "Report Date" },
  { accessorKey: "reported_eps", header: "Reported EPS", cell: (i) => `$${fmt(i.getValue() as number)}` },
  { accessorKey: "estimated_eps", header: "Est. EPS", cell: (i) => `$${fmt(i.getValue() as number)}` },
  {
    accessorKey: "surprise",
    header: "Surprise",
    cell: (i) => <PnlCell value={i.getValue() as number} />,
  },
  {
    accessorKey: "surprise_pct",
    header: "Surprise%",
    cell: (i) => <PctCell value={i.getValue() as number} />,
  },
];

function EarningsTab() {
  const [symbolFilter, setSymbolFilter] = useState("");
  const [data, setData] = useState<EarningsRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get<{ data: EarningsRow[]; count: number }>("/db/earnings", {
        params: {
          symbol: symbolFilter.trim().toUpperCase() || undefined,
          limit: 100,
        },
      });
      setData(res.data.data ?? []);
    } catch {
      setError("Failed to fetch earnings data");
    } finally {
      setLoading(false);
    }
  }, [symbolFilter]);

  useEffect(() => {
    void fetchData();
  }, [fetchData]);

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-4">
      <motion.div variants={fadeUp}>
        <FilterBar>
          <TextInput value={symbolFilter} onChange={setSymbolFilter} placeholder="Symbol (optional)" className="w-36" />
          <Button size="sm" onClick={() => void fetchData()} disabled={loading}>
            {loading ? <RefreshCw size={12} className="animate-spin" /> : null}
            Fetch
          </Button>
        </FilterBar>
        {error && <p className="mb-3 text-xs text-rose-400">{error}</p>}
      </motion.div>
      <motion.div variants={fadeUp}>
        <DataGrid data={data} columns={earningsColumns} maxHeight={480} exportFilename="earnings" />
      </motion.div>
    </motion.div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

export default function DatabasePage() {
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  const renderTab = () => {
    switch (activeTab) {
      case "overview": return <OverviewTab />;
      case "market": return <MarketDataTab />;
      case "performance": return <PerformanceTab />;
      case "trades": return <TradeLogTab />;
      case "signals": return <SignalsTab />;
      case "risk": return <RiskEventsTab />;
      case "reference": return <ReferenceDataTab />;
      case "earnings": return <EarningsTab />;
    }
  };

  return (
    <motion.div
      variants={stagger}
      initial="hidden"
      animate="show"
      className="space-y-5"
    >
      {/* Page header */}
      <motion.div variants={fadeUp} className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-teal-500/10 bg-teal-500/10">
          <Database size={18} className="text-teal-400" />
        </div>
        <div>
          <h1 className="text-base font-bold tracking-tight text-slate-100">Database Explorer</h1>
          <p className="text-[11px] text-slate-500">
            TimescaleDB — 23 tables across market data, trading, signals, risk, and reference data
          </p>
        </div>
      </motion.div>

      {/* Tab navigation */}
      <motion.div variants={fadeUp}>
        <div className="flex gap-1 overflow-x-auto rounded-xl border border-white/[0.06] bg-white/[0.02] p-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs font-medium transition-all whitespace-nowrap ${
                activeTab === tab.id
                  ? "bg-teal-500/15 text-teal-300 border border-teal-500/20"
                  : "text-slate-500 hover:bg-white/[0.04] hover:text-slate-300 border border-transparent"
              }`}
            >
              <tab.icon size={14} />
              {tab.label}
            </button>
          ))}
        </div>
      </motion.div>

      {/* Tab content */}
      <motion.div variants={fadeUp} key={activeTab}>
        {renderTab()}
      </motion.div>
    </motion.div>
  );
}
