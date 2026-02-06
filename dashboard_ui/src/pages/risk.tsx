import { useEffect, useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  BarChart,
  Bar,
} from "recharts";
import { AlertTriangle, ShieldAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore } from "@/lib/store";

export default function RiskWarRoomPage() {
  const {
    varData,
    riskMetrics,
    riskConcentration,
    riskCorrelation,
    riskStress,
    riskAttribution,
    alerts,
    fetchVar,
    fetchRiskMetrics,
    fetchRiskConcentration,
    fetchRiskCorrelation,
    fetchRiskStress,
    fetchRiskAttribution,
    fetchAlerts,
  } = useStore();

  useEffect(() => {
    void Promise.all([
      fetchVar(),
      fetchRiskMetrics(),
      fetchRiskConcentration(),
      fetchRiskCorrelation(),
      fetchRiskStress(),
      fetchRiskAttribution(),
      fetchAlerts(),
    ]);
    const timer = setInterval(
      () =>
        void Promise.all([
          fetchVar(),
          fetchRiskMetrics(),
          fetchRiskConcentration(),
          fetchRiskCorrelation(),
          fetchRiskStress(),
          fetchRiskAttribution(),
          fetchAlerts(),
        ]),
      12000,
    );
    return () => clearInterval(timer);
  }, [fetchVar, fetchRiskMetrics, fetchRiskConcentration, fetchRiskCorrelation, fetchRiskStress, fetchRiskAttribution, fetchAlerts]);

  const distribution = varData?.distribution_curve ?? [];
  const sectorExposure = useMemo(
    () =>
      Object.entries(riskMetrics?.sector_exposures ?? {}).map(([sector, value]) => ({
        sector,
        value: Number((value * 100).toFixed(2)),
      })),
    [riskMetrics],
  );

  const highSeverityAlerts = alerts.filter((x) => x.severity === "CRITICAL" || x.severity === "HIGH").slice(0, 8);

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-rose-200 bg-rose-50/80 p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-rose-700">Risk Command</p>
            <h1 className="mt-1 text-3xl font-bold text-rose-900">Risk War Room</h1>
            <p className="mt-1 text-sm text-rose-800">
              VaR, drawdown, exposure concentration, and incident stream in one place.
            </p>
          </div>
          <Badge variant="warning">
            <ShieldAlert size={12} className="mr-1" />
            Live Monitoring
          </Badge>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>VaR 95%</CardDescription>
            <CardTitle className="text-3xl">${(varData?.var_95 ?? 0).toLocaleString()}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>VaR 99%</CardDescription>
            <CardTitle className="text-3xl">${(varData?.var_99 ?? 0).toLocaleString()}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>CVaR 95%</CardDescription>
            <CardTitle className="text-3xl">${(varData?.cvar_95 ?? 0).toLocaleString()}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Current Drawdown</CardDescription>
            <CardTitle className="text-3xl">{((riskMetrics?.current_drawdown ?? 0) * 100).toFixed(2)}%</CardTitle>
          </CardHeader>
        </Card>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Largest Symbol</CardDescription>
            <CardTitle className="text-2xl">{(riskConcentration?.largest_symbol_pct ?? 0).toFixed(2)}%</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Top 3 Concentration</CardDescription>
            <CardTitle className="text-2xl">{(riskConcentration?.top3_symbols_pct ?? 0).toFixed(2)}%</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Cluster Correlation</CardDescription>
            <CardTitle className="text-2xl">{(riskCorrelation?.cluster_risk_score ?? 0).toFixed(3)}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Resilience Score</CardDescription>
            <CardTitle className="text-2xl">{(riskStress?.resilience_score ?? 0).toFixed(2)}</CardTitle>
          </CardHeader>
        </Card>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>Loss Distribution</CardTitle>
            <CardDescription>One-day probability curve</CardDescription>
          </CardHeader>
          <CardContent className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={distribution}>
                <defs>
                  <linearGradient id="distFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#dc2626" stopOpacity={0.45} />
                    <stop offset="95%" stopColor="#dc2626" stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
                <XAxis dataKey="pnl" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Area type="monotone" dataKey="probability" stroke="#b91c1c" strokeWidth={2.2} fill="url(#distFill)" />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>Sector Concentration</CardTitle>
            <CardDescription>Current exposure % by sector</CardDescription>
          </CardHeader>
          <CardContent className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sectorExposure}>
                <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
                <XAxis dataKey="sector" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#0f766e" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </section>

      <Card className="border-slate-200 bg-white/90">
        <CardHeader>
          <CardTitle>High Severity Alerts</CardTitle>
          <CardDescription>Critical/high risk incidents requiring operator action</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {highSeverityAlerts.length === 0 ? (
            <p className="text-sm text-slate-500">No high severity risk incident.</p>
          ) : (
            highSeverityAlerts.map((alert) => (
              <div key={alert.alert_id} className="flex items-start justify-between gap-3 rounded-lg border border-rose-200 bg-rose-50 p-3">
                <div>
                  <p className="text-sm font-semibold text-rose-900">{alert.title}</p>
                  <p className="text-sm text-rose-800">{alert.message}</p>
                </div>
                <Badge variant="warning">
                  <AlertTriangle size={12} className="mr-1" />
                  {alert.severity}
                </Badge>
              </div>
            ))
          )}
        </CardContent>
      </Card>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>Stress Scenario Losses</CardTitle>
            <CardDescription>Scenario PnL impact in dollars</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {Object.entries(riskStress?.scenarios ?? {}).length === 0 ? (
              <p className="text-sm text-slate-500">No stress scenario data.</p>
            ) : (
              Object.entries(riskStress?.scenarios ?? {}).map(([name, value]) => (
                <div key={name} className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <p className="text-sm font-medium text-slate-700">{name.replaceAll("_", " ")}</p>
                  <p className="font-mono text-sm text-rose-700">${Number(value).toLocaleString()}</p>
                </div>
              ))
            )}
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>Risk Attribution</CardTitle>
            <CardDescription>Pre-trade and post-trade breach signals</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <p className="text-xs uppercase tracking-wider text-slate-500">Breaches</p>
              <p className="mt-1 text-2xl font-semibold text-rose-800">{riskAttribution?.breaches_count ?? 0}</p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <p className="text-xs uppercase tracking-wider text-slate-500">Pre-Trade Checks</p>
              <p className="mt-1 text-sm text-slate-700">{riskAttribution?.pre_trade_checks?.length ?? 0} evaluated</p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <p className="text-xs uppercase tracking-wider text-slate-500">Post-Trade Findings</p>
              <p className="mt-1 text-sm text-slate-700">{riskAttribution?.post_trade_findings?.length ?? 0} flagged</p>
            </div>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
