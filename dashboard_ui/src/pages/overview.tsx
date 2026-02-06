import { useEffect, useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, DollarSign, ShieldAlert, Wifi } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useStore } from "@/lib/store";

export default function OverviewPage() {
  const {
    fetchSnapshot,
    portfolio,
    riskMetrics,
    tca,
    ws,
    tradingStatus,
    alerts,
    varData,
    explainability,
  } = useStore();

  useEffect(() => {
    void fetchSnapshot();
    const timer = setInterval(() => void fetchSnapshot(), 15000);
    return () => clearInterval(timer);
  }, [fetchSnapshot]);

  const wsConnected = ws.portfolio && ws.orders && ws.signals && ws.alerts;
  const unresolvedAlerts = alerts.filter((x) => x.status !== "RESOLVED").length;

  const stressChartData = useMemo(
    () =>
      Object.entries(varData?.stress_scenarios ?? {}).map(([name, impact]) => ({
        name,
        impact: Math.abs(impact),
      })),
    [varData],
  );

  const driftData = useMemo(
    () =>
      Object.entries(explainability?.recent_shift ?? {}).map(([feature, value]) => ({
        feature,
        value: Number((value * 100).toFixed(2)),
      })),
    [explainability],
  );

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">Live Command Center</p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">Institutional Trading Control</h1>
            <p className="mt-1 text-sm text-slate-600">
              Unified monitoring for portfolio, risk, execution quality, and operational incidents.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={wsConnected ? "success" : "warning"}>
              <Wifi size={12} className="mr-1" />
              {wsConnected ? "Streams Online" : "Streams Degraded"}
            </Badge>
            <Badge variant={tradingStatus?.running ? "success" : "outline"}>
              <Activity size={12} className="mr-1" />
              {tradingStatus?.running ? "Engine Running" : "Engine Idle"}
            </Badge>
            <Badge variant={unresolvedAlerts > 0 ? "warning" : "success"}>
              <ShieldAlert size={12} className="mr-1" />
              {unresolvedAlerts} Open Alerts
            </Badge>
          </div>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Portfolio Equity</CardDescription>
            <CardTitle className="text-3xl">${(portfolio?.equity ?? 0).toLocaleString()}</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-600">
            <span className="inline-flex items-center gap-1">
              <DollarSign size={14} />
              Buying Power ${(portfolio?.buying_power ?? 0).toLocaleString()}
            </span>
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Daily PnL</CardDescription>
            <CardTitle className="text-3xl">${(portfolio?.daily_pnl ?? 0).toLocaleString()}</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-600">
            Total ${(portfolio?.total_pnl ?? 0).toLocaleString()}
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Current Drawdown</CardDescription>
            <CardTitle className="text-3xl">{((riskMetrics?.current_drawdown ?? 0) * 100).toFixed(2)}%</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-600">
            Max 30d {((riskMetrics?.max_drawdown_30d ?? 0) * 100).toFixed(2)}%
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Execution Fill Probability</CardDescription>
            <CardTitle className="text-3xl">{((tca?.fill_probability ?? 0) * 100).toFixed(1)}%</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-600">
            Slippage {tca?.slippage_bps?.toFixed(2) ?? "--"} bps
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle className="text-xl">Stress Scenario Impact</CardTitle>
            <CardDescription>Absolute downside (%) across predefined shocks</CardDescription>
          </CardHeader>
          <CardContent className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={stressChartData}>
                <defs>
                  <linearGradient id="stressFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#fb923c" stopOpacity={0.55} />
                    <stop offset="100%" stopColor="#fb923c" stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#dbe2f3" strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Area type="monotone" dataKey="impact" stroke="#ea580c" fill="url(#stressFill)" strokeWidth={2.2} />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle className="text-xl">Model Factor Drift</CardTitle>
            <CardDescription>Recent change (%) in explainability weights</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {driftData.length === 0 ? (
              <p className="text-sm text-slate-500">No explainability artifact detected yet.</p>
            ) : (
              driftData.map((row) => (
                <div key={row.feature} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium text-slate-700">{row.feature.replaceAll("_", " ")}</span>
                    <span className={row.value >= 0 ? "text-emerald-700" : "text-rose-700"}>
                      {row.value >= 0 ? "+" : ""}
                      {row.value.toFixed(2)}%
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded bg-slate-100">
                    <div
                      className={row.value >= 0 ? "h-full bg-emerald-500" : "h-full bg-rose-500"}
                      style={{ width: `${Math.min(Math.abs(row.value) * 5, 100)}%` }}
                    />
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </section>
    </div>
  );
}

