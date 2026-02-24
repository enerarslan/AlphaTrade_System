import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { useShallow } from "zustand/react/shallow";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Bot, FlaskConical, GitCompare, Layers, ShieldCheck, Sparkles } from "lucide-react";
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

const darkTooltipStyle = {
  contentStyle: { background: "rgba(15,23,42,0.95)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: "8px", color: "#e2e8f0", fontSize: 12 },
  itemStyle: { color: "#e2e8f0" },
  labelStyle: { color: "#94a3b8" },
};

export default function ModelsPage() {
  const {
    hasPermission,
    modelStatuses,
    explainability,
    modelRegistry,
    modelDrift,
    modelValidation,
    mfaStatus,
    fetchModelStatuses,
    fetchExplainability,
    fetchModelRegistry,
    fetchModelDrift,
    fetchModelValidation,
    fetchChampionChallenger,
    promoteChampion,
  } = useStore(useShallow((state) => ({
      hasPermission: state.hasPermission,
      modelStatuses: state.modelStatuses,
      explainability: state.explainability,
      modelRegistry: state.modelRegistry,
      modelDrift: state.modelDrift,
      modelValidation: state.modelValidation,
      mfaStatus: state.mfaStatus,
      fetchModelStatuses: state.fetchModelStatuses,
      fetchExplainability: state.fetchExplainability,
      fetchModelRegistry: state.fetchModelRegistry,
      fetchModelDrift: state.fetchModelDrift,
      fetchModelValidation: state.fetchModelValidation,
      fetchChampionChallenger: state.fetchChampionChallenger,
      promoteChampion: state.promoteChampion,
    })));

  const [promoteName, setPromoteName] = useState("");
  const [promoteVersion, setPromoteVersion] = useState("");
  const [promoteReason, setPromoteReason] = useState("");
  const [mfaCode, setMfaCode] = useState("");
  const requiresMfa = Boolean(mfaStatus?.mfa_enabled);
  const canPromote = hasPermission("models.governance.promote");

  useEffect(() => {
    void fetchModelStatuses();
    void fetchExplainability();
    void fetchModelRegistry();
    void fetchModelDrift();
    void fetchModelValidation();
    void fetchChampionChallenger();
    const timer = setInterval(() => {
      if (typeof document !== "undefined" && document.visibilityState !== "visible") return;
      void fetchModelStatuses();
    }, 30000);
    return () => clearInterval(timer);
  }, [fetchModelStatuses, fetchExplainability, fetchModelRegistry, fetchModelDrift, fetchModelValidation, fetchChampionChallenger]);

  const featureData = useMemo(() => {
    const raw = explainability?.global_importance ?? {};
    return Object.entries(raw)
      .map(([name, value]) => ({ name, importance: Number(value) }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 12);
  }, [explainability]);

  const driftStatus = useMemo(() => {
    const score = modelDrift?.drift_score ?? 0;
    if (score >= 0.5) return { label: "DRIFT DETECTED", variant: "error" as const, color: "text-rose-400" };
    if (score >= 0.2) return { label: "DRIFT ELEVATED", variant: "warning" as const, color: "text-amber-400" };
    return { label: "STABLE", variant: "success" as const, color: "text-emerald-400" };
  }, [modelDrift]);

  const validationGates = useMemo(
    () => (modelValidation?.gates ?? []).map((g) => {
      const entries = Object.entries(g);
      const name = entries[0]?.[0] ?? "unknown";
      const passed = Boolean(entries[0]?.[1]);
      return [name, passed] as [string, boolean];
    }),
    [modelValidation],
  );

  const recentEntries = useMemo(
    () => (modelRegistry?.entries ?? []).slice(0, 6),
    [modelRegistry],
  );

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-6">
      {/* Header */}
      <motion.section variants={fadeUp} className="rounded-2xl border border-indigo-500/10 bg-indigo-500/[0.02] p-6 backdrop-blur-sm">
        <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-indigo-400/70">Model Governance</p>
        <h1 className="mt-2 text-3xl font-bold text-slate-100">Model Intelligence</h1>
        <p className="mt-1 text-sm text-slate-400">Feature explainability, drift monitoring, validation gates, and model promotion.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          <Badge variant="success">{modelStatuses.filter((m) => m.status.toLowerCase() === "healthy").length} Healthy</Badge>
          <Badge variant={driftStatus.variant}>{driftStatus.label}</Badge>
          <Badge variant="outline">Registry: {modelRegistry?.versions_count ?? 0} versions</Badge>
        </div>
      </motion.section>

      {/* Feature Importance + Model Status */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles size={18} className="text-cyan-400" />
              Feature Importance
            </CardTitle>
            <CardDescription>Top features by SHAP importance score.</CardDescription>
          </CardHeader>
          <CardContent>
            {featureData.length > 0 ? (
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={featureData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" horizontal={false} />
                  <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={{ stroke: "rgba(255,255,255,0.06)" }} />
                  <YAxis dataKey="name" type="category" tick={{ fill: "#94a3b8", fontSize: 11 }} width={100} axisLine={{ stroke: "rgba(255,255,255,0.06)" }} />
                  <Tooltip {...darkTooltipStyle} />
                  <Bar dataKey="importance" fill="#06b6d4" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-sm text-slate-500">No explainability data available.</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bot size={18} className="text-emerald-400" />
              Model Status
            </CardTitle>
            <CardDescription>Runtime model health indicators.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {modelStatuses.length === 0 ? (
              <p className="text-sm text-slate-500">No models registered.</p>
            ) : (
              modelStatuses.map((m) => (
                <div key={m.model_name} className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-4 py-3">
                  <div>
                    <p className="font-medium text-slate-200">{m.model_name}</p>
                    <p className="text-xs text-slate-500">Predictions: {m.prediction_count} | Errors: {m.error_count}</p>
                  </div>
                  <Badge variant={m.status.toLowerCase() === "healthy" ? "success" : "warning"}>
                    {m.status}
                  </Badge>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Drift + Validation */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FlaskConical size={18} className="text-amber-400" />
              Drift Monitor
            </CardTitle>
            <CardDescription>Tracks feature and prediction distribution drift.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between rounded-xl border border-white/[0.08] bg-white/[0.02] p-4">
              <div>
                <p className="text-sm text-slate-400">Drift Score</p>
                <p className={`text-3xl font-bold font-mono ${driftStatus.color}`}>
                  {(modelDrift?.drift_score ?? 0).toFixed(4)}
                </p>
              </div>
              <Badge variant={driftStatus.variant}>{driftStatus.label}</Badge>
            </div>
            <div className="space-y-2">
              {Object.entries(modelDrift?.feature_shift ?? {}).slice(0, 6).map(([feature, score]) => (
                <div key={feature} className="flex items-center gap-3">
                  <span className="min-w-[100px] truncate text-xs text-slate-400">{feature}</span>
                  <div className="flex-1">
                    <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.min(Number(score) * 100, 100)}%` }}
                        transition={{ duration: 0.8, ease: "easeOut" }}
                        className={`h-full rounded-full ${Number(score) >= 0.5 ? "bg-rose-500" : Number(score) >= 0.2 ? "bg-amber-500" : "bg-emerald-500"}`}
                      />
                    </div>
                  </div>
                  <span className="text-xs font-mono text-slate-300">{Number(score).toFixed(3)}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ShieldCheck size={18} className="text-emerald-400" />
              Validation Gates
            </CardTitle>
            <CardDescription>Model quality gates before production promotion.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {validationGates.length === 0 ? (
              <p className="text-sm text-slate-500">No validation gate data available.</p>
            ) : (
              validationGates.map(([gate, passed]) => (
                <div key={gate} className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-4 py-3">
                  <span className="text-sm text-slate-300">{gate.replaceAll("_", " ")}</span>
                  <Badge variant={passed ? "success" : "error"}>
                    {passed ? "PASS" : "FAIL"}
                  </Badge>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Champion / Challenger Promotion + Training Jobs */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <GitCompare size={18} className="text-cyan-400" />
              Champion Promotion
            </CardTitle>
            <CardDescription>Promote a model version to champion status.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <label className="block text-sm">
              <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">Model Name</span>
              <input className="glass-input h-10 w-full" value={promoteName} onChange={(e) => setPromoteName(e.target.value)} placeholder="model_name" />
            </label>
            <label className="block text-sm">
              <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">Version ID</span>
              <input className="glass-input h-10 w-full" value={promoteVersion} onChange={(e) => setPromoteVersion(e.target.value)} placeholder="v1.0.0" />
            </label>
            <label className="block text-sm">
              <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">Reason</span>
              <input className="glass-input h-10 w-full" value={promoteReason} onChange={(e) => setPromoteReason(e.target.value)} placeholder="e.g. better validation accuracy" />
            </label>
            {requiresMfa && (
              <label className="block text-sm">
                <span className="mb-1 block text-xs uppercase tracking-wider text-slate-500">MFA Code</span>
                <input className="glass-input h-10 w-full font-mono" value={mfaCode} onChange={(e) => setMfaCode(e.target.value.replace(/\D/g, "").slice(0, 6))} placeholder="6-digit code" />
              </label>
            )}
            <Button disabled={!canPromote || !promoteName || !promoteVersion || (requiresMfa && mfaCode.length !== 6)} onClick={() => void promoteChampion(promoteName, promoteVersion, promoteReason || undefined, mfaCode || undefined)}>
              Promote Champion
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Layers size={18} className="text-amber-400" />
              Recent Model Versions
            </CardTitle>
            <CardDescription>Latest registry entries.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {recentEntries.length === 0 ? (
              <p className="text-sm text-slate-500">No registry entries found.</p>
            ) : (
              recentEntries.map((entry, i: number) => (
                <div key={i} className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-4 py-3">
                  <div>
                    <p className="font-medium text-slate-200">{entry.model_name}</p>
                    <p className="text-xs text-slate-500">{entry.model_version} — {entry.registered_at}</p>
                  </div>
                  <Badge variant={entry.is_active ? "success" : "outline"}>
                    {entry.is_active ? "active" : "inactive"}
                  </Badge>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}


