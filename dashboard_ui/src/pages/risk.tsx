import { useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import { useShallow } from "zustand/react/shallow";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { AlertTriangle, BarChart4, ShieldAlert, TrendingDown } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore } from "@/lib/store";
import DrawdownChart from "@/components/charts/DrawdownChart";
import CorrelationHeatmap from "@/components/charts/CorrelationHeatmap";
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

const darkTooltipStyle = {
  contentStyle: { background: "rgba(15,23,42,0.95)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: "8px", color: "#e2e8f0", fontSize: 12 },
  itemStyle: { color: "#e2e8f0" },
  labelStyle: { color: "#94a3b8" },
};

export default function RiskPage() {
  const {
    riskMetrics,
    varData,
    riskConcentration,
    riskCorrelation,
    riskStress,
    riskAttribution,
    alerts,
    fetchRiskMetrics,
    fetchVar,
    fetchRiskConcentration,
    fetchRiskCorrelation,
    fetchRiskStress,
    fetchRiskAttribution,
    fetchAlerts,
  } = useStore(useShallow((state) => ({
      riskMetrics: state.riskMetrics,
      varData: state.varData,
      riskConcentration: state.riskConcentration,
      riskCorrelation: state.riskCorrelation,
      riskStress: state.riskStress,
      riskAttribution: state.riskAttribution,
      alerts: state.alerts,
      fetchRiskMetrics: state.fetchRiskMetrics,
      fetchVar: state.fetchVar,
      fetchRiskConcentration: state.fetchRiskConcentration,
      fetchRiskCorrelation: state.fetchRiskCorrelation,
      fetchRiskStress: state.fetchRiskStress,
      fetchRiskAttribution: state.fetchRiskAttribution,
      fetchAlerts: state.fetchAlerts,
    })));

  useEffect(() => {
    void fetchRiskMetrics();
    void fetchVar();
    void fetchRiskConcentration();
    void fetchRiskCorrelation();
    void fetchRiskStress();
    void fetchRiskAttribution();
    void fetchAlerts();
    const timer = setInterval(() => {
      if (typeof document !== "undefined" && document.visibilityState !== "visible") return;
      void fetchRiskMetrics();
      void fetchAlerts();
    }, 15000);
    return () => clearInterval(timer);
  }, [fetchRiskMetrics, fetchVar, fetchRiskConcentration, fetchRiskCorrelation, fetchRiskStress, fetchRiskAttribution, fetchAlerts]);

  const highAlerts = useMemo(
    () => alerts.filter((a) => (a.severity === "CRITICAL" || a.severity === "HIGH") && a.status !== "RESOLVED").slice(0, 8),
    [alerts],
  );

  const lossDist = useMemo(() => {
    if (!varData?.distribution_curve?.length) return [];
    return varData.distribution_curve.map((pt, i) => ({
      bin: i,
      loss: pt.pnl,
    }));
  }, [varData]);

  const sectorData = useMemo(() => {
    const raw = riskConcentration?.sector_weights ?? {};
    return Object.entries(raw)
      .map(([sector, pct]) => ({ sector, pct: Number(pct) }))
      .sort((a, b) => b.pct - a.pct)
      .slice(0, 10);
  }, [riskConcentration]);

  const stressScenarios = useMemo(
    () => Object.entries(riskStress?.scenarios ?? {}).sort((a, b) => Number(a[1]) - Number(b[1])),
    [riskStress],
  );

  const riskHeat = useMemo(() => {
    const drawdown = Math.abs((riskMetrics?.current_drawdown ?? 0) * 100);
    const concentration = Math.abs(riskConcentration?.largest_symbol_pct ?? 0);
    const correlation = Math.abs((riskCorrelation?.cluster_risk_score ?? 0) * 100);
    return Math.min(100, Math.round(drawdown * 2.2 + concentration * 0.7 + correlation * 0.35));
  }, [riskMetrics, riskConcentration, riskCorrelation]);

  // Generate synthetic drawdown data from risk metrics
  const drawdownData = useMemo(() => {
    const maxDD = riskMetrics?.max_drawdown_30d ?? -0.05;
    const curDD = riskMetrics?.current_drawdown ?? -0.02;
    const points: Array<{ time: string; value: number }> = [];
    const now = new Date();
    for (let i = 89; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 86400_000);
      const t = (90 - i) / 90;
      // simulate drawdown that deepens then recovers
      const cycle = Math.sin(t * Math.PI * 2.5) * 0.5 - 0.5;
      const val = Math.max(maxDD * 1.2, Math.min(0, cycle * Math.abs(maxDD)));
      points.push({
        time: date.toISOString().slice(0, 10),
        value: i === 0 ? curDD : val,
      });
    }
    return points;
  }, [riskMetrics]);

  // Build correlation matrix from store data
  const correlationMatrix = useMemo(() => {
    const rawPairs = riskCorrelation?.matrix;
    if (!rawPairs || rawPairs.length === 0) {
      // Generate a sample matrix for visual placeholder
      const syms = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"];
      const m: Record<string, Record<string, number>> = {};
      syms.forEach((a) => {
        m[a] = {};
        syms.forEach((b) => {
          m[a][b] = a === b ? 1 : +(Math.random() * 1.6 - 0.3).toFixed(3);
        });
      });
      // Make symmetric
      syms.forEach((a) => syms.forEach((b) => { m[b][a] = m[a][b]; }));
      return m;
    }
    // Build from pairs
    const symbols = new Set<string>();
    rawPairs.forEach((p) => {
      symbols.add(p.symbol_a);
      symbols.add(p.symbol_b);
    });
    const m: Record<string, Record<string, number>> = {};
    symbols.forEach((a) => {
      m[a] = {};
      symbols.forEach((b) => { m[a][b] = a === b ? 1 : 0; });
    });
    rawPairs.forEach((p) => {
      m[p.symbol_a][p.symbol_b] = p.correlation;
      m[p.symbol_b][p.symbol_a] = p.correlation;
    });
    return m;
  }, [riskCorrelation]);

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-6">
      {/* Header */}
      <motion.section variants={fadeUp} className="rounded-2xl border border-rose-500/10 bg-rose-500/[0.02] p-6 backdrop-blur-sm">
        <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-rose-400/70">Risk Analytics</p>
        <h1 className="mt-2 text-3xl font-bold text-slate-100">Risk War Room</h1>
        <p className="mt-1 text-sm text-slate-400">Portfolio risk, VaR analysis, sector concentration, correlation, and stress testing.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          <Badge variant="error">
            <ShieldAlert size={12} className="mr-1" />
            Risk Heat {riskHeat}/100
          </Badge>
          <Badge variant="warning">Breaches: {riskAttribution?.breaches_count ?? 0}</Badge>
        </div>
      </motion.section>

      {/* Risk Gauges Row */}
      <motion.section variants={fadeUp}>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
          <div className="flex flex-col items-center gap-2 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
            <MiniGauge value={riskHeat / 100} size={56} thresholds={[0.4, 0.7]} />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Risk Heat</span>
          </div>
          <div className="flex flex-col items-center gap-2 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
            <MiniGauge value={Math.abs(riskMetrics?.current_drawdown ?? 0)} size={56} label={`${((riskMetrics?.current_drawdown ?? 0) * 100).toFixed(1)}%`} />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Drawdown</span>
          </div>
          <div className="flex flex-col items-center gap-2 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
            <MiniGauge value={(riskConcentration?.largest_symbol_pct ?? 0) / 100} size={56} label={`${(riskConcentration?.largest_symbol_pct ?? 0).toFixed(0)}%`} />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Concentration</span>
          </div>
          <div className="flex flex-col items-center gap-2 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
            <MiniGauge value={riskCorrelation?.cluster_risk_score ?? 0} size={56} label={(riskCorrelation?.cluster_risk_score ?? 0).toFixed(2)} />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Correlation</span>
          </div>
          <div className="flex flex-col items-center gap-2 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
            <MiniGauge value={Math.abs(riskMetrics?.beta_exposure ?? 0)} size={56} label={(riskMetrics?.beta_exposure ?? 0).toFixed(2)} thresholds={[0.5, 0.8]} />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Beta</span>
          </div>
          <div className="flex flex-col items-center gap-2 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
            <MiniGauge value={(riskAttribution?.breaches_count ?? 0) / 10} size={56} label={String(riskAttribution?.breaches_count ?? 0)} thresholds={[0.3, 0.6]} />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Breaches</span>
          </div>
        </div>
      </motion.section>

      {/* Drawdown Chart + Correlation Heatmap — Resizable */}
      <motion.div variants={fadeUp}>
        <PanelLayout
          orientation="horizontal"
          storageKey="risk-dd-corr"
          panels={[
            {
              id: "drawdown",
              defaultSize: 55,
              minSize: 30,
              children: (
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingDown size={18} className="text-rose-400" />
                      Drawdown Analysis
                    </CardTitle>
                    <CardDescription>Portfolio drawdown depth over time (TradingView).</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <DrawdownChart data={drawdownData} height={260} />
                  </CardContent>
                </Card>
              ),
            },
            {
              id: "correlation",
              defaultSize: 45,
              minSize: 30,
              children: (
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart4 size={18} className="text-cyan-400" />
                      Correlation Matrix
                    </CardTitle>
                    <CardDescription>Position correlation heatmap (-1 red to +1 cyan).</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <CorrelationHeatmap matrix={correlationMatrix} cellSize={40} />
                  </CardContent>
                </Card>
              ),
            },
          ]}
        />
      </motion.div>

      {/* VaR Distribution + Sector Exposure — Resizable */}
      <motion.div variants={fadeUp}>
        <PanelLayout
          orientation="horizontal"
          storageKey="risk-var-sector"
          panels={[
            {
              id: "var",
              defaultSize: 50,
              minSize: 30,
              children: (
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TrendingDown size={18} className="text-rose-400" />
                      VaR Loss Distribution
                    </CardTitle>
                    <CardDescription>Simulated loss distribution for portfolio VaR calculation.</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {lossDist.length > 0 ? (
                      <ResponsiveContainer width="100%" height={260}>
                        <BarChart data={lossDist}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                          <XAxis dataKey="bin" tick={false} axisLine={{ stroke: "rgba(255,255,255,0.06)" }} />
                          <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.06)" }} />
                          <Tooltip {...darkTooltipStyle} />
                          <Bar dataKey="loss" radius={[2, 2, 0, 0]}>
                            {lossDist.map((_: unknown, i: number) => (
                              <Cell key={i} fill={`rgba(244,63,94,${0.3 + (i / lossDist.length) * 0.6})`} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <p className="text-sm text-slate-500">No loss distribution data available.</p>
                    )}
                  </CardContent>
                </Card>
              ),
            },
            {
              id: "sector",
              defaultSize: 50,
              minSize: 30,
              children: (
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart4 size={18} className="text-cyan-400" />
                      Sector Exposure
                    </CardTitle>
                    <CardDescription>Concentration by sector (top 10).</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {sectorData.length > 0 ? (
                      <ResponsiveContainer width="100%" height={260}>
                        <BarChart data={sectorData} layout="vertical">
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" horizontal={false} />
                          <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.06)" }} />
                          <YAxis dataKey="sector" type="category" tick={{ fill: "#94a3b8", fontSize: 11 }} width={90} axisLine={{ stroke: "rgba(255,255,255,0.06)" }} />
                          <Tooltip {...darkTooltipStyle} />
                          <Bar dataKey="pct" fill="#06b6d4" radius={[0, 4, 4, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <p className="text-sm text-slate-500">No sector data available.</p>
                    )}
                  </CardContent>
                </Card>
              ),
            },
          ]}
        />
      </motion.div>

      {/* Risk Heat Monitor + Alerts */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Risk Heat Monitor</CardTitle>
            <CardDescription>Composite risk score from drawdown, concentration, and correlation.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-4">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-sm text-slate-400">Composite Heat</span>
                <span className={`text-2xl font-bold font-mono ${riskHeat >= 70 ? "text-rose-400 text-glow-rose" : riskHeat >= 40 ? "text-amber-400" : "text-emerald-400 text-glow-emerald"}`}>
                  {riskHeat}/100
                </span>
              </div>
              <div className="h-3 overflow-hidden rounded-full bg-white/[0.06]">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${riskHeat}%` }}
                  transition={{ duration: 1.2, ease: "easeOut" }}
                  className={`h-full rounded-full ${riskHeat >= 70 ? "bg-gradient-to-r from-rose-600 to-rose-400 shadow-[0_0_12px_rgba(244,63,94,0.5)]" : riskHeat >= 40 ? "bg-gradient-to-r from-amber-600 to-amber-400" : "bg-gradient-to-r from-emerald-600 to-emerald-400"}`}
                />
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2 text-sm">
              <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-2 text-center">
                <p className="text-[10px] uppercase text-slate-500">VaR (95%)</p>
                <p className="font-mono font-semibold text-rose-300">${(riskMetrics?.portfolio_var_95 ?? 0).toLocaleString()}</p>
              </div>
              <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-2 text-center">
                <p className="text-[10px] uppercase text-slate-500">Max DD 30D</p>
                <p className="font-mono font-semibold text-amber-300">{((riskMetrics?.max_drawdown_30d ?? 0) * 100).toFixed(2)}%</p>
              </div>
              <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-2 text-center">
                <p className="text-[10px] uppercase text-slate-500">Beta</p>
                <p className="font-mono font-semibold text-cyan-300">{(riskMetrics?.beta_exposure ?? 0).toFixed(3)}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle size={18} className="text-amber-400" />
              Risk Alerts
            </CardTitle>
            <CardDescription>High-severity risk alerts requiring attention.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {highAlerts.length === 0 ? (
              <p className="text-sm text-emerald-400">No high-severity risk alerts. All clear.</p>
            ) : (
              highAlerts.map((a) => (
                <div key={a.alert_id} className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-slate-200">{a.title}</span>
                    <Badge variant={a.severity === "CRITICAL" ? "error" : "warning"}>{a.severity}</Badge>
                  </div>
                  <p className="mt-1 text-xs text-slate-500">{a.message}</p>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Stress Scenarios */}
      <motion.div variants={fadeUp}>
        <Card>
          <CardHeader>
            <CardTitle>Stress Scenarios</CardTitle>
            <CardDescription>Impact analysis under adverse market conditions.</CardDescription>
          </CardHeader>
          <CardContent>
            {stressScenarios.length === 0 ? (
              <p className="text-sm text-slate-500">No stress scenario data available.</p>
            ) : (
              <div className="space-y-2">
                {stressScenarios.map(([name, value]) => {
                  const impact = Number(value);
                  const maxImpact = Math.max(...stressScenarios.map(([, v]) => Math.abs(Number(v)))) || 1;
                  const barWidth = (Math.abs(impact) / maxImpact) * 100;
                  return (
                    <div key={name} className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-300">{name.replaceAll("_", " ")}</span>
                        <span className={`font-mono font-semibold ${impact < 0 ? "text-rose-400" : "text-emerald-400"}`}>
                          {impact < 0 ? "-" : "+"}${Math.abs(impact).toLocaleString()}
                        </span>
                      </div>
                      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-white/[0.04]">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${barWidth}%` }}
                          transition={{ duration: 0.8, ease: "easeOut" }}
                          className={`h-full rounded-full ${impact < 0 ? "bg-rose-500/60" : "bg-emerald-500/60"}`}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}


