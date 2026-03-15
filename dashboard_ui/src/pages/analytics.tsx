import { useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import { useShallow } from "zustand/react/shallow";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
} from "recharts";
import {
  Activity,
  BarChart2,
  TrendingUp,
  TrendingDown,
  Layers,
  Target,
  Zap,
  Award,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore } from "@/lib/store";
import MiniGauge from "@/components/ui/mini-gauge";
import PanelLayout from "@/components/layout/PanelLayout";

const stagger = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.07 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" as const } },
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

// KPI card with violet accent theme
function KpiCard({
  label,
  value,
  sub,
  positive,
  icon: Icon,
}: {
  label: string;
  value: string;
  sub?: string;
  positive?: boolean;
  icon?: React.ElementType;
}) {
  const accent =
    positive === undefined
      ? "border-violet-500/20 shadow-[0_0_14px_rgba(139,92,246,0.07)]"
      : positive
        ? "border-emerald-500/20 shadow-[0_0_14px_rgba(16,185,129,0.07)]"
        : "border-rose-500/20 shadow-[0_0_14px_rgba(244,63,94,0.07)]";
  const textColor =
    positive === undefined
      ? "text-violet-300"
      : positive
        ? "text-emerald-300"
        : "text-rose-300";

  return (
    <div
      className={`rounded-xl border bg-white/[0.03] px-4 py-3.5 backdrop-blur-sm transition-all hover:bg-white/[0.05] ${accent}`}
    >
      <div className="flex items-center justify-between">
        <p className="text-[10px] uppercase tracking-[0.16em] text-slate-500">{label}</p>
        {Icon && <Icon size={14} className="text-slate-600" />}
      </div>
      <p className={`mt-1.5 font-mono text-xl font-bold ${textColor}`}>{value}</p>
      {sub && <p className="mt-0.5 text-[10px] text-slate-600">{sub}</p>}
    </div>
  );
}

// Animated exposure bar row
function ExposureRow({
  label,
  value,
  pct,
  color,
}: {
  label: string;
  value: string;
  pct: number;
  color: string;
}) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono text-slate-300">{value}</span>
      </div>
      <div className="h-3 overflow-hidden rounded-full bg-white/[0.06]">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
          transition={{ duration: 1.0, ease: "easeOut" }}
          className={`h-full rounded-full ${color}`}
        />
      </div>
    </div>
  );
}

export default function AnalyticsPage() {
  const {
    performance,
    portfolio,
    positions,
    orders,
    riskMetrics,
    tca,
    riskConcentration,
    fetchSnapshot,
    fetchRiskConcentration,
  } = useStore(
    useShallow((state) => ({
      performance: state.performance,
      portfolio: state.portfolio,
      positions: state.positions,
      orders: state.orders,
      riskMetrics: state.riskMetrics,
      tca: state.tca,
      riskConcentration: state.riskConcentration,
      fetchSnapshot: state.fetchSnapshot,
      fetchRiskConcentration: state.fetchRiskConcentration,
    })),
  );

  useEffect(() => {
    const visible = () =>
      typeof document === "undefined" || document.visibilityState === "visible";

    if (visible()) {
      void fetchSnapshot();
      void fetchRiskConcentration();
    }

    const timer = setInterval(() => {
      if (!visible()) return;
      void fetchSnapshot();
      void fetchRiskConcentration();
    }, 30000);

    const onVisibility = () => {
      if (visible()) {
        void fetchSnapshot();
        void fetchRiskConcentration();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [fetchSnapshot, fetchRiskConcentration]);

  // ─── Derived values ─────────────────────────────────────────────────────

  const equity = portfolio?.equity ?? 0;
  const totalPnl = performance?.total_pnl ?? portfolio?.total_pnl ?? 0;
  const totalReturnPct = equity > 0 ? (totalPnl / equity) * 100 : 0;
  const sharpe = performance?.sharpe_ratio_30d ?? 0;
  const sortino = performance?.sortino_ratio_30d ?? 0;
  const maxDD = performance?.max_drawdown_30d ?? riskMetrics?.max_drawdown_30d ?? 0;
  const winRate = performance?.win_rate_30d ?? 0;
  const profitFactor = performance?.profit_factor ?? 0;
  const avgTradePnl = performance?.avg_trade_pnl ?? 0;
  const calmar = maxDD !== 0 ? Math.abs(totalPnl / (maxDD * equity)) : 0;

  const longExp = portfolio?.long_exposure ?? 0;
  const shortExp = portfolio?.short_exposure ?? 0;
  const netExp = portfolio?.net_exposure ?? 0;
  const grossExp = portfolio?.gross_exposure ?? 0;
  const lsRatio = shortExp !== 0 ? (longExp / Math.abs(shortExp)).toFixed(2) : "—";

  const maxExposure = Math.max(longExp, Math.abs(shortExp), 1);
  const longPct = (longExp / maxExposure) * 100;
  const shortPct = (Math.abs(shortExp) / maxExposure) * 100;

  // P&L waterfall — sorted from worst to best
  const waterfallData = useMemo(
    () =>
      [...positions]
        .sort((a, b) => a.unrealized_pnl - b.unrealized_pnl)
        .map((p) => ({
          symbol: p.symbol,
          pnl: p.unrealized_pnl,
          pnlPct: p.unrealized_pnl_pct,
        })),
    [positions],
  );

  // Sector allocation bars
  const sectorData = useMemo(() => {
    const raw = riskConcentration?.sector_weights ?? {};
    return Object.entries(raw)
      .map(([sector, pct]) => ({ sector, pct: Number(pct) }))
      .sort((a, b) => b.pct - a.pct)
      .slice(0, 10);
  }, [riskConcentration]);

  // Top 10 positions by market value
  const topHoldings = useMemo(
    () =>
      [...positions]
        .sort((a, b) => Math.abs(b.market_value) - Math.abs(a.market_value))
        .slice(0, 10),
    [positions],
  );

  // Order status distribution
  const orderStatusDist = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const o of orders) {
      counts[o.status] = (counts[o.status] ?? 0) + 1;
    }
    return Object.entries(counts)
      .map(([status, count]) => ({ status, count }))
      .sort((a, b) => b.count - a.count);
  }, [orders]);

  const maxOrderCount = useMemo(
    () => Math.max(...orderStatusDist.map((d) => d.count), 1),
    [orderStatusDist],
  );

  // Risk-return scatter data
  const scatterData = useMemo(
    () =>
      positions.map((p) => ({
        returnPct: Number((p.unrealized_pnl_pct * 100).toFixed(2)),
        size: Math.abs(p.market_value),
        symbol: p.symbol,
        pnl: p.unrealized_pnl,
      })),
    [positions],
  );

  // HHI scaled 0-1 (max = 10000)
  const hhiSymbol = (riskConcentration?.hhi_symbol ?? 0) / 10000;
  const hhiSector = (riskConcentration?.hhi_sector ?? 0) / 10000;

  const orderStatusColor = (status: string) => {
    if (status === "FILLED" || status === "filled") return "rgba(52,211,153,0.7)";
    if (status === "CANCELLED" || status === "cancelled") return "rgba(148,163,184,0.5)";
    if (status === "REJECTED" || status === "rejected") return "rgba(244,63,94,0.7)";
    if (status === "PENDING" || status === "pending" || status === "new")
      return "rgba(251,191,36,0.7)";
    return "rgba(139,92,246,0.6)";
  };

  const sign = (n: number) => (n >= 0 ? "+" : "-");
  const fmtAbs = (n: number) =>
    `$${Math.abs(n).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-6">
      {/* ════════════════════════════════════════════════════════════════════
          HEADER
      ═══════════════════════════════════════════════════════════════════════ */}
      <motion.section
        variants={fadeUp}
        className="rounded-2xl border border-violet-500/10 bg-violet-500/[0.025] p-6 backdrop-blur-sm"
      >
        <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-violet-400/70">
          Institutional Analytics
        </p>
        <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-100">
          Performance Analytics
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Deep-dive P&amp;L attribution, exposure breakdown, concentration risk, TCA, and
          risk-return scatter — Bloomberg-grade analytics view.
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          <Badge variant="default">
            <Activity size={12} className="mr-1" />
            {positions.length} Positions
          </Badge>
          <Badge variant={totalReturnPct >= 0 ? "success" : "error"}>
            {sign(totalReturnPct)}
            {Math.abs(totalReturnPct).toFixed(2)}% Total Return
          </Badge>
          <Badge variant={sharpe >= 1 ? "success" : sharpe >= 0 ? "warning" : "error"}>
            Sharpe {(sharpe ?? 0).toFixed(2)}
          </Badge>
          <Badge variant="secondary">{orders.length} Orders on Record</Badge>
        </div>
      </motion.section>

      {/* ════════════════════════════════════════════════════════════════════
          SECTION 1 — KPI Strip (8 metrics)
      ═══════════════════════════════════════════════════════════════════════ */}
      <motion.section variants={fadeUp}>
        <div className="mb-2.5 flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-full bg-violet-400 shadow-[0_0_8px_rgba(139,92,246,0.6)]" />
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-violet-400/80">
            Key Performance Indicators
          </span>
          <span className="text-[9px] text-slate-600">30-Day Rolling</span>
        </div>
        <div className="grid gap-3 grid-cols-2 sm:grid-cols-4 lg:grid-cols-8">
          <KpiCard
            label="Total Return"
            value={`${sign(totalReturnPct)}${Math.abs(totalReturnPct).toFixed(2)}%`}
            sub={`${sign(totalPnl)}${fmtAbs(totalPnl)} PnL`}
            positive={totalReturnPct >= 0}
            icon={TrendingUp}
          />
          <KpiCard
            label="Sharpe Ratio"
            value={(sharpe ?? 0).toFixed(2)}
            sub="Risk-adj return"
            positive={sharpe >= 1}
            icon={Award}
          />
          <KpiCard
            label="Sortino Ratio"
            value={(sortino ?? 0).toFixed(2)}
            sub="Downside-adj"
            positive={sortino >= 1}
            icon={Award}
          />
          <KpiCard
            label="Max Drawdown"
            value={`${((maxDD ?? 0) * 100).toFixed(2)}%`}
            sub="30-day peak-trough"
            positive={Math.abs(maxDD ?? 0) < 0.05}
            icon={TrendingDown}
          />
          <KpiCard
            label="Win Rate"
            value={`${((winRate ?? 0) * 100).toFixed(1)}%`}
            sub="30-day trades"
            positive={winRate >= 0.5}
            icon={Target}
          />
          <KpiCard
            label="Profit Factor"
            value={(profitFactor ?? 0).toFixed(2)}
            sub="Gross profit / loss"
            positive={profitFactor >= 1}
            icon={Layers}
          />
          <KpiCard
            label="Avg Trade PnL"
            value={`${sign(avgTradePnl)}${fmtAbs(avgTradePnl)}`}
            sub="Per trade"
            positive={avgTradePnl >= 0}
            icon={BarChart2}
          />
          <KpiCard
            label="Calmar Ratio"
            value={calmar.toFixed(2)}
            sub="Return / max DD"
            positive={calmar >= 1}
            icon={Zap}
          />
        </div>
      </motion.section>

      {/* ════════════════════════════════════════════════════════════════════
          SECTION 2 — P&L Waterfall Chart
      ═══════════════════════════════════════════════════════════════════════ */}
      <motion.div variants={fadeUp}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart2 size={18} className="text-violet-400" />
              P&amp;L Waterfall — Position Attribution
            </CardTitle>
            <CardDescription>
              Unrealized P&amp;L per position sorted from largest loss to largest gain. Emerald =
              profit, rose = loss.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {waterfallData.length === 0 ? (
              <div className="flex h-[280px] items-center justify-center text-sm text-slate-500">
                No open positions. P&amp;L waterfall unavailable.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart
                  data={waterfallData}
                  margin={{ top: 8, right: 8, left: 0, bottom: 28 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(255,255,255,0.04)"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="symbol"
                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                    axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                    tickLine={false}
                    angle={-35}
                    textAnchor="end"
                    interval={0}
                  />
                  <YAxis
                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                    axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                    tickLine={false}
                    tickFormatter={(v: number) =>
                      `${v >= 0 ? "" : "-"}$${Math.abs(v).toLocaleString()}`
                    }
                  />
                  <Tooltip
                    {...darkTooltipStyle}
                    formatter={(
                      value: number | undefined,
                      _name: string | undefined,
                      props: { payload?: { symbol: string; pnlPct: number } },
                    ) => [
                      `${(value ?? 0) >= 0 ? "+" : ""}$${(value ?? 0).toLocaleString()} (${((props.payload?.pnlPct ?? 0) * 100).toFixed(2)}%)`,
                      "Unrealized PnL",
                    ]}
                    labelFormatter={(label: string) => label}
                  />
                  <Bar dataKey="pnl" radius={[4, 4, 0, 0]} maxBarSize={48}>
                    {waterfallData.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={
                          entry.pnl >= 0 ? "rgba(52,211,153,0.75)" : "rgba(244,63,94,0.75)"
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* ════════════════════════════════════════════════════════════════════
          SECTION 3 — Exposure Analysis
      ═══════════════════════════════════════════════════════════════════════ */}
      <motion.div variants={fadeUp}>
        <PanelLayout
          orientation="horizontal"
          storageKey="analytics-exposure"
          panels={[
            {
              id: "long-short",
              defaultSize: 48,
              minSize: 30,
              children: (
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Layers size={18} className="text-violet-400" />
                      Long / Short Exposure
                    </CardTitle>
                    <CardDescription>Portfolio directional exposure breakdown.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <ExposureRow
                      label="Long Exposure"
                      value={`$${longExp.toLocaleString()}`}
                      pct={longPct}
                      color="bg-emerald-500/60"
                    />
                    <ExposureRow
                      label="Short Exposure"
                      value={`$${Math.abs(shortExp).toLocaleString()}`}
                      pct={shortPct}
                      color="bg-rose-500/60"
                    />
                    <div className="grid grid-cols-3 gap-2 pt-1 text-center">
                      <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-2">
                        <p className="text-[9px] uppercase tracking-wider text-slate-500">Net</p>
                        <p
                          className={`mt-0.5 font-mono text-sm font-semibold ${netExp >= 0 ? "text-emerald-400" : "text-rose-400"}`}
                        >
                          {netExp >= 0 ? "" : "-"}${Math.abs(netExp).toLocaleString()}
                        </p>
                      </div>
                      <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-2">
                        <p className="text-[9px] uppercase tracking-wider text-slate-500">Gross</p>
                        <p className="mt-0.5 font-mono text-sm font-semibold text-slate-200">
                          ${grossExp.toLocaleString()}
                        </p>
                      </div>
                      <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-2">
                        <p className="text-[9px] uppercase tracking-wider text-slate-500">L/S</p>
                        <p className="mt-0.5 font-mono text-sm font-semibold text-violet-300">
                          {lsRatio}
                        </p>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-2 text-center">
                        <p className="text-[9px] uppercase tracking-wider text-slate-500">Beta</p>
                        <p className="mt-0.5 font-mono text-sm font-semibold text-amber-300">
                          {(riskMetrics?.beta_exposure ?? 0).toFixed(2)}
                        </p>
                      </div>
                      <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-2 text-center">
                        <p className="text-[9px] uppercase tracking-wider text-slate-500">
                          VaR 95%
                        </p>
                        <p className="mt-0.5 font-mono text-sm font-semibold text-rose-300">
                          ${(riskMetrics?.portfolio_var_95 ?? 0).toLocaleString()}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ),
            },
            {
              id: "sector-alloc",
              defaultSize: 52,
              minSize: 30,
              children: (
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart2 size={18} className="text-violet-400" />
                      Sector Allocation
                    </CardTitle>
                    <CardDescription>Portfolio weight by sector (top 10).</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {sectorData.length === 0 ? (
                      <div className="flex h-[240px] items-center justify-center text-sm text-slate-500">
                        No sector data available.
                      </div>
                    ) : (
                      <ResponsiveContainer width="100%" height={260}>
                        <BarChart
                          data={sectorData}
                          layout="vertical"
                          margin={{ top: 0, right: 12, left: 0, bottom: 0 }}
                        >
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="rgba(255,255,255,0.04)"
                            horizontal={false}
                          />
                          <XAxis
                            type="number"
                            tick={{ fill: "#94a3b8", fontSize: 11 }}
                            axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                            tickLine={false}
                            tickFormatter={(v: number) => `${v.toFixed(1)}%`}
                          />
                          <YAxis
                            dataKey="sector"
                            type="category"
                            tick={{ fill: "#94a3b8", fontSize: 11 }}
                            axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                            tickLine={false}
                            width={100}
                          />
                          <Tooltip
                            {...darkTooltipStyle}
                            formatter={(v: number | undefined) => [`${(v ?? 0).toFixed(2)}%`, "Weight"]}
                          />
                          <Bar dataKey="pct" radius={[0, 4, 4, 0]} maxBarSize={20}>
                            {sectorData.map((_: unknown, i: number) => (
                              <Cell
                                key={i}
                                fill={`hsla(${260 + i * 14},60%,${58 - i * 3}%,0.72)`}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    )}
                  </CardContent>
                </Card>
              ),
            },
          ]}
        />
      </motion.div>

      {/* ════════════════════════════════════════════════════════════════════
          SECTION 4 — Position Concentration
      ═══════════════════════════════════════════════════════════════════════ */}
      <motion.div variants={fadeUp}>
        <PanelLayout
          orientation="horizontal"
          storageKey="analytics-concentration"
          panels={[
            {
              id: "top-holdings",
              defaultSize: 62,
              minSize: 35,
              children: (
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Layers size={18} className="text-violet-400" />
                      Top Holdings
                    </CardTitle>
                    <CardDescription>Top 10 positions by absolute market value.</CardDescription>
                  </CardHeader>
                  <CardContent className="p-0">
                    {topHoldings.length === 0 ? (
                      <p className="px-6 pb-6 text-sm text-slate-500">No positions held.</p>
                    ) : (
                      <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-white/[0.06]">
                              {[
                                "Symbol",
                                "Qty",
                                "Entry",
                                "Current",
                                "Mkt Value",
                                "PnL",
                                "PnL%",
                              ].map((h) => (
                                <th
                                  key={h}
                                  className="px-4 py-3 text-left font-semibold uppercase tracking-wider text-slate-500"
                                >
                                  {h}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {topHoldings.map((p, i) => {
                              const pnlPos = p.unrealized_pnl >= 0;
                              return (
                                <tr
                                  key={p.symbol}
                                  className={`border-b border-white/[0.04] transition-colors hover:bg-white/[0.03] ${i % 2 === 0 ? "" : "bg-white/[0.01]"}`}
                                >
                                  <td className="px-4 py-2.5 font-medium text-slate-200">
                                    {p.symbol}
                                  </td>
                                  <td className="px-4 py-2.5 font-mono text-slate-400">
                                    {p.quantity.toLocaleString()}
                                  </td>
                                  <td className="px-4 py-2.5 font-mono text-slate-400">
                                    ${p.avg_entry_price.toFixed(2)}
                                  </td>
                                  <td className="px-4 py-2.5 font-mono text-slate-300">
                                    ${p.current_price.toFixed(2)}
                                  </td>
                                  <td className="px-4 py-2.5 font-mono text-slate-300">
                                    ${Math.abs(p.market_value).toLocaleString()}
                                  </td>
                                  <td
                                    className={`px-4 py-2.5 font-mono font-semibold ${pnlPos ? "text-emerald-400" : "text-rose-400"}`}
                                  >
                                    {pnlPos ? "+" : ""}$
                                    {p.unrealized_pnl.toLocaleString(undefined, {
                                      maximumFractionDigits: 0,
                                    })}
                                  </td>
                                  <td
                                    className={`px-4 py-2.5 font-mono font-semibold ${pnlPos ? "text-emerald-400" : "text-rose-400"}`}
                                  >
                                    {pnlPos ? "+" : ""}
                                    {(p.unrealized_pnl_pct * 100).toFixed(2)}%
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ),
            },
            {
              id: "concentration-metrics",
              defaultSize: 38,
              minSize: 25,
              children: (
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Target size={18} className="text-violet-400" />
                      Concentration Metrics
                    </CardTitle>
                    <CardDescription>HHI and position weight analysis.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-5">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex flex-col items-center gap-2 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
                        <MiniGauge
                          value={hhiSymbol}
                          size={64}
                          label={`${(riskConcentration?.hhi_symbol ?? 0).toFixed(0)}`}
                          thresholds={[0.15, 0.35]}
                        />
                        <span className="text-center text-[10px] uppercase tracking-wider text-slate-500">
                          HHI Symbol
                        </span>
                      </div>
                      <div className="flex flex-col items-center gap-2 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
                        <MiniGauge
                          value={hhiSector}
                          size={64}
                          label={`${(riskConcentration?.hhi_sector ?? 0).toFixed(0)}`}
                          thresholds={[0.15, 0.35]}
                        />
                        <span className="text-center text-[10px] uppercase tracking-wider text-slate-500">
                          HHI Sector
                        </span>
                      </div>
                    </div>

                    <div className="space-y-2">
                      {[
                        {
                          label: "Largest Position",
                          value: `${(riskConcentration?.largest_symbol_pct ?? 0).toFixed(1)}%`,
                          warn: (riskConcentration?.largest_symbol_pct ?? 0) > 20,
                        },
                        {
                          label: "Top 3 Positions",
                          value: `${(riskConcentration?.top3_symbols_pct ?? 0).toFixed(1)}%`,
                          warn: (riskConcentration?.top3_symbols_pct ?? 0) > 50,
                        },
                        {
                          label: "Correlation Risk",
                          value: (riskMetrics?.correlation_risk ?? 0).toFixed(3),
                          warn: (riskMetrics?.correlation_risk ?? 0) > 0.7,
                        },
                      ].map(({ label, value, warn }) => (
                        <div
                          key={label}
                          className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2"
                        >
                          <span className="text-xs text-slate-400">{label}</span>
                          <span
                            className={`font-mono text-xs font-semibold ${warn ? "text-amber-400" : "text-slate-200"}`}
                          >
                            {value}
                          </span>
                        </div>
                      ))}
                    </div>

                    {Object.keys(riskConcentration?.symbol_weights ?? {}).length > 0 && (
                      <div className="space-y-1.5">
                        <p className="text-[9px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                          Symbol Weights
                        </p>
                        {Object.entries(riskConcentration?.symbol_weights ?? {})
                          .sort(([, a], [, b]) => Number(b) - Number(a))
                          .slice(0, 6)
                          .map(([sym, pct]) => (
                            <div key={sym}>
                              <div className="mb-0.5 flex justify-between text-[10px]">
                                <span className="text-slate-400">{sym}</span>
                                <span className="font-mono text-slate-300">
                                  {Number(pct).toFixed(1)}%
                                </span>
                              </div>
                              <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
                                <div
                                  className="h-full rounded-full bg-violet-500/50"
                                  style={{ width: `${Math.min(100, Number(pct))}%` }}
                                />
                              </div>
                            </div>
                          ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ),
            },
          ]}
        />
      </motion.div>

      {/* ════════════════════════════════════════════════════════════════════
          SECTION 5 — Trade Analysis
      ═══════════════════════════════════════════════════════════════════════ */}
      <motion.div variants={fadeUp}>
        <PanelLayout
          orientation="horizontal"
          storageKey="analytics-trade"
          panels={[
            {
              id: "order-dist",
              defaultSize: 50,
              minSize: 30,
              children: (
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Activity size={18} className="text-violet-400" />
                      Order Status Distribution
                    </CardTitle>
                    <CardDescription>
                      Count of orders by execution status ({orders.length} total).
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {orderStatusDist.length === 0 ? (
                      <div className="flex h-[200px] items-center justify-center text-sm text-slate-500">
                        No order history available.
                      </div>
                    ) : (
                      <div className="space-y-2.5">
                        {orderStatusDist.map(({ status, count }) => (
                          <div key={status}>
                            <div className="mb-1 flex justify-between text-xs">
                              <span className="font-medium text-slate-300">{status}</span>
                              <span className="font-mono text-slate-400">{count}</span>
                            </div>
                            <div className="h-3 overflow-hidden rounded-full bg-white/[0.06]">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${(count / maxOrderCount) * 100}%` }}
                                transition={{ duration: 0.9, ease: "easeOut" }}
                                className="h-full rounded-full"
                                style={{ background: orderStatusColor(status) }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ),
            },
            {
              id: "tca",
              defaultSize: 50,
              minSize: 30,
              children: (
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Zap size={18} className="text-violet-400" />
                      Execution Cost Analysis (TCA)
                    </CardTitle>
                    <CardDescription>
                      Slippage, market impact, fill probability, VWAP savings.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {[
                      {
                        label: "Slippage",
                        value: `${(tca?.slippage_bps ?? 0).toFixed(2)} bps`,
                        sub: "vs arrival price",
                        warn: (tca?.slippage_bps ?? 0) > 5,
                      },
                      {
                        label: "Market Impact",
                        value: `${(tca?.market_impact_bps ?? 0).toFixed(2)} bps`,
                        sub: "price movement cost",
                        warn: (tca?.market_impact_bps ?? 0) > 3,
                      },
                      {
                        label: "Cost Savings vs VWAP",
                        value: `${(tca?.cost_savings_vs_vwap ?? 0).toFixed(2)} bps`,
                        sub: "benchmark outperformance",
                        warn: (tca?.cost_savings_vs_vwap ?? 0) < 0,
                      },
                      {
                        label: "Fill Probability",
                        value: `${((tca?.fill_probability ?? 0) * 100).toFixed(1)}%`,
                        sub: "order fill rate",
                        warn: (tca?.fill_probability ?? 0) < 0.9,
                      },
                      {
                        label: "Execution Speed",
                        value: `${(tca?.execution_speed_ms ?? 0).toFixed(0)} ms`,
                        sub: "avg latency",
                        warn: (tca?.execution_speed_ms ?? 0) > 100,
                      },
                    ].map(({ label, value, sub, warn }) => (
                      <div
                        key={label}
                        className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2.5"
                      >
                        <div>
                          <p className="text-xs font-medium text-slate-300">{label}</p>
                          <p className="text-[10px] text-slate-600">{sub}</p>
                        </div>
                        <span
                          className={`font-mono text-sm font-semibold ${warn ? "text-amber-400" : "text-emerald-400"}`}
                        >
                          {value}
                        </span>
                      </div>
                    ))}

                    {tca?.venue_breakdown &&
                      Object.keys(tca.venue_breakdown).length > 0 && (
                        <div className="mt-2 space-y-1.5 border-t border-white/[0.06] pt-3">
                          <p className="text-[9px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                            Venue Breakdown
                          </p>
                          {Object.entries(tca.venue_breakdown).map(([venue, pct]) => (
                            <div
                              key={venue}
                              className="flex items-center justify-between text-xs"
                            >
                              <span className="text-slate-400">{venue}</span>
                              <span className="font-mono text-slate-300">
                                {Number(pct).toFixed(1)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                  </CardContent>
                </Card>
              ),
            },
          ]}
        />
      </motion.div>

      {/* ════════════════════════════════════════════════════════════════════
          SECTION 6 — Risk-Return Scatter
      ═══════════════════════════════════════════════════════════════════════ */}
      <motion.div variants={fadeUp}>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp size={18} className="text-violet-400" />
              Risk-Return Scatter — Position Map
            </CardTitle>
            <CardDescription>
              Each bubble represents one position. X axis = return %, Y axis = absolute market
              value. Emerald = profitable, rose = loss. Bubble size scales with exposure.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {scatterData.length === 0 ? (
              <div className="flex h-[300px] items-center justify-center text-sm text-slate-500">
                No position data available for scatter plot.
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 12, right: 20, left: 0, bottom: 12 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis
                    dataKey="returnPct"
                    name="Return %"
                    type="number"
                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                    axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                    tickLine={false}
                    tickFormatter={(v: number) => `${v >= 0 ? "+" : ""}${v}%`}
                    label={{
                      value: "Return %",
                      position: "insideBottomRight",
                      offset: -8,
                      fill: "#64748b",
                      fontSize: 11,
                    }}
                  />
                  <YAxis
                    dataKey="size"
                    name="Market Value"
                    type="number"
                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                    axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                    tickLine={false}
                    tickFormatter={(v: number) =>
                      v >= 1_000_000
                        ? `$${(v / 1_000_000).toFixed(1)}M`
                        : v >= 1000
                          ? `$${(v / 1000).toFixed(0)}k`
                          : `$${v}`
                    }
                    label={{
                      value: "Market Value",
                      angle: -90,
                      position: "insideLeft",
                      offset: 10,
                      fill: "#64748b",
                      fontSize: 11,
                    }}
                  />
                  <ZAxis dataKey="size" range={[40, 400]} name="Size" />
                  <Tooltip
                    content={({ payload }) => {
                      if (!payload || payload.length === 0) return null;
                      const d = payload[0]?.payload as {
                        symbol: string;
                        returnPct: number;
                        size: number;
                        pnl: number;
                      };
                      const pos = d.returnPct >= 0;
                      return (
                        <div
                          style={{
                            background: "rgba(15,23,42,0.95)",
                            border: "1px solid rgba(255,255,255,0.08)",
                            borderRadius: 8,
                            padding: "8px 12px",
                            fontSize: 12,
                            color: "#e2e8f0",
                          }}
                        >
                          <p style={{ fontWeight: 600, color: "#c4b5fd" }}>{d.symbol}</p>
                          <p>
                            Return:{" "}
                            <span style={{ color: pos ? "#34d399" : "#f43f5e" }}>
                              {d.returnPct >= 0 ? "+" : ""}
                              {d.returnPct}%
                            </span>
                          </p>
                          <p>
                            Mkt Value:{" "}
                            <span style={{ color: "#cbd5e1" }}>${d.size.toLocaleString()}</span>
                          </p>
                          <p>
                            PnL:{" "}
                            <span style={{ color: pos ? "#34d399" : "#f43f5e" }}>
                              {d.pnl >= 0 ? "+" : ""}$
                              {d.pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                            </span>
                          </p>
                        </div>
                      );
                    }}
                  />
                  <Scatter
                    data={scatterData}
                    shape={(props: {
                      cx?: number;
                      cy?: number;
                      r?: number;
                      payload?: { returnPct: number };
                    }) => {
                      const { cx = 0, cy = 0, r = 8, payload } = props;
                      const positive = (payload?.returnPct ?? 0) >= 0;
                      return (
                        <circle
                          cx={cx}
                          cy={cy}
                          r={r}
                          fill={positive ? "rgba(52,211,153,0.55)" : "rgba(244,63,94,0.55)"}
                          stroke={positive ? "rgba(52,211,153,0.9)" : "rgba(244,63,94,0.9)"}
                          strokeWidth={1.5}
                        />
                      );
                    }}
                  />
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}
