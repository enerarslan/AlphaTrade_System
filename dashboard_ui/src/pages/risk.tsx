import { useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { AlertTriangle, ShieldAlert, TrendingDown, BarChart4 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
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
  } = useStore();

  useEffect(() => {
    void fetchRiskMetrics();
    void fetchVar();
    void fetchRiskConcentration();
    void fetchRiskCorrelation();
    void fetchRiskStress();
    void fetchRiskAttribution();
    void fetchAlerts();
    const timer = setInterval(() => {
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

      {/* Top KPIs */}
      <motion.div variants={fadeUp} className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { label: "VaR (95%)", value: `$${(riskMetrics?.portfolio_var_95 ?? 0).toLocaleString()}`, accent: "rose" as const },
          { label: "Max Drawdown", value: `${((riskMetrics?.max_drawdown_30d ?? 0) * 100).toFixed(2)}%`, accent: "rose" as const },
          { label: "Current DD", value: `${((riskMetrics?.current_drawdown ?? 0) * 100).toFixed(2)}%`, accent: "amber" as const },
          { label: "Beta Exposure", value: (riskMetrics?.beta_exposure ?? 0).toFixed(3), accent: "cyan" as const },
        ].map((m) => (
          <div key={m.label} className={`rounded-xl border bg-white/[0.03] px-4 py-3 backdrop-blur-sm ${m.accent === "rose" ? "border-rose-500/20 shadow-[0_0_15px_rgba(244,63,94,0.06)]" : m.accent === "amber" ? "border-amber-500/20" : "border-cyan-500/20"}`}>
            <p className="text-[10px] uppercase tracking-[0.16em] text-slate-500">{m.label}</p>
            <p className={`mt-1 text-xl font-bold font-mono ${m.accent === "rose" ? "text-rose-300" : m.accent === "amber" ? "text-amber-300" : "text-cyan-300"}`}>
              {m.value}
            </p>
          </div>
        ))}
      </motion.div>

      {/* Charts Row */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-2">
        {/* VaR Distribution */}
        <Card>
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

        {/* Sector Exposure */}
        <Card>
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
      </motion.div>

      {/* Risk Heat Monitor + Correlation */}
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
                <p className="text-[10px] uppercase text-slate-500">Drawdown</p>
                <p className="font-mono font-semibold text-rose-300">{((riskMetrics?.current_drawdown ?? 0) * 100).toFixed(2)}%</p>
              </div>
              <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-2 text-center">
                <p className="text-[10px] uppercase text-slate-500">Concentration</p>
                <p className="font-mono font-semibold text-amber-300">{(riskConcentration?.largest_symbol_pct ?? 0).toFixed(2)}%</p>
              </div>
              <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-2 text-center">
                <p className="text-[10px] uppercase text-slate-500">Correlation</p>
                <p className="font-mono font-semibold text-cyan-300">{(riskCorrelation?.cluster_risk_score ?? 0).toFixed(3)}</p>
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
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-white/[0.06] text-xs uppercase tracking-wider text-slate-500">
                      <th className="pb-2 pr-4">Scenario</th>
                      <th className="pb-2 text-right">Impact ($)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stressScenarios.map(([name, value]) => {
                      const impact = Number(value);
                      return (
                        <tr key={name} className="border-b border-white/[0.04] hover:bg-white/[0.02]">
                          <td className="py-2 pr-4 text-slate-300">{name.replaceAll("_", " ")}</td>
                          <td className={`py-2 text-right font-mono font-semibold ${impact < 0 ? "text-rose-400" : "text-emerald-400"}`}>
                            {impact < 0 ? "-" : "+"}${Math.abs(impact).toLocaleString()}
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
      </motion.div>
    </motion.div>
  );
}
