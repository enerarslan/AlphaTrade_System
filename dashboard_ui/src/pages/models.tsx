import { useEffect, useMemo, useState } from "react";
import { BrainCircuit, Cpu, FlaskConical } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore } from "@/lib/store";

export default function ModelsPage() {
  const {
    explainability,
    modelRegistry,
    modelDrift,
    modelValidation,
    championChallenger,
    mfaStatus,
    jobs,
    hasPermission,
    fetchExplainability,
    fetchModelRegistry,
    fetchModelDrift,
    fetchModelValidation,
    fetchChampionChallenger,
    fetchJobs,
    createJob,
    promoteChampion,
  } = useStore();

  const canReadGovernance = hasPermission("models.governance.read");
  const canPromote = hasPermission("models.governance.write");
  const canCreateJobs = hasPermission("control.jobs.create");
  const requiresMfa = Boolean(mfaStatus?.mfa_enabled);

  const [mfaCode, setMfaCode] = useState("");
  const [selectedVersion, setSelectedVersion] = useState("");

  useEffect(() => {
    if (!canReadGovernance) {
      return;
    }
    void Promise.all([
      fetchExplainability(),
      fetchModelRegistry(),
      fetchModelDrift(),
      fetchModelValidation(),
      fetchChampionChallenger(),
      fetchJobs(),
    ]);
  }, [canReadGovernance, fetchExplainability, fetchModelRegistry, fetchModelDrift, fetchModelValidation, fetchChampionChallenger, fetchJobs]);

  const topImportance = useMemo(
    () =>
      Object.entries(explainability?.global_importance ?? {})
        .map(([feature, value]) => ({ feature, value }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 12),
    [explainability],
  );

  const entries = modelRegistry?.entries ?? [];
  const trainingJobs = jobs.filter((x) => x.command === "train").slice(0, 6);

  useEffect(() => {
    if (!selectedVersion && entries.length > 0) {
      const first = entries[0];
      setSelectedVersion(`${first.model_name}::${first.version_id}`);
    }
  }, [entries, selectedVersion]);

  const selectedEntry = useMemo(() => {
    if (!selectedVersion) {
      return null;
    }
    const [modelName, versionId] = selectedVersion.split("::");
    return entries.find((x) => x.model_name === modelName && x.version_id === versionId) ?? null;
  }, [entries, selectedVersion]);

  if (!canReadGovernance) {
    return (
      <div className="space-y-6">
        <section className="rounded-2xl border border-slate-200 bg-white/90 p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">ModelOps</p>
          <h1 className="mt-2 text-3xl font-bold text-slate-900">Model Lifecycle Console</h1>
          <p className="mt-1 text-sm text-rose-700">Your role does not have model governance access.</p>
        </section>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-slate-200 bg-white/90 p-6">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">ModelOps</p>
        <h1 className="mt-2 text-3xl font-bold text-slate-900">Model Governance Console</h1>
        <p className="mt-1 text-sm text-slate-600">
          Registry, drift, validation gates, and champion/challenger promotion.
        </p>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Registry Models</CardDescription>
            <CardTitle className="text-3xl">{modelRegistry?.model_count ?? 0}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Drift Score</CardDescription>
            <CardTitle className="text-3xl">{(modelDrift?.drift_score ?? 0).toFixed(3)}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Validation</CardDescription>
            <CardTitle className="text-2xl">{modelValidation?.passed ? "PASS" : "BLOCKED"}</CardTitle>
          </CardHeader>
        </Card>
        <Card className="border-slate-200 bg-white/90">
          <CardHeader className="pb-2">
            <CardDescription>Champion</CardDescription>
            <CardTitle className="truncate text-lg">{championChallenger?.champion ?? "--"}</CardTitle>
          </CardHeader>
        </Card>
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <Card className="border-slate-200 bg-white/90 lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BrainCircuit size={18} />
              Feature Importance
            </CardTitle>
            <CardDescription>Loaded from latest model artifact in `/models/*_artifacts.json`</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {topImportance.length === 0 ? (
              <p className="text-sm text-slate-500">No explainability artifact found yet.</p>
            ) : (
              topImportance.map((row) => (
                <div key={row.feature} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium text-slate-700">{row.feature.replaceAll("_", " ")}</span>
                    <span className="font-mono">{(row.value * 100).toFixed(2)}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded bg-slate-100">
                    <div className="h-full bg-sky-600" style={{ width: `${Math.min(Math.abs(row.value) * 100, 100)}%` }} />
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FlaskConical size={18} />
              Promotion Control
            </CardTitle>
            <CardDescription>Champion/challenger switching with audit trail</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <label className="block text-sm">
              Candidate Version
              <select
                className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3"
                value={selectedVersion}
                onChange={(e) => setSelectedVersion(e.target.value)}
              >
                {entries.length === 0 ? <option value="">No model versions</option> : null}
                {entries.map((entry) => (
                  <option key={`${entry.model_name}::${entry.version_id}`} value={`${entry.model_name}::${entry.version_id}`}>
                    {entry.model_name}:{entry.version_id}
                  </option>
                ))}
              </select>
            </label>
            {requiresMfa ? (
              <label className="block text-sm">
                MFA Code
                <input
                  className="mt-1 h-10 w-full rounded-lg border border-slate-300 bg-white px-3 font-mono"
                  value={mfaCode}
                  onChange={(e) => setMfaCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                  placeholder="6-digit TOTP"
                />
              </label>
            ) : null}
            <Button
              className="w-full justify-start gap-2"
              variant="outline"
              disabled={!canPromote || !selectedEntry || (requiresMfa && mfaCode.length !== 6)}
              onClick={() => {
                if (!selectedEntry) {
                  return;
                }
                void promoteChampion(selectedEntry.model_name, selectedEntry.version_id, "dashboard_manual_promotion", mfaCode || undefined);
              }}
            >
              <Cpu size={16} />
              Promote To Champion
            </Button>
            {!canPromote ? <p className="text-xs text-rose-700">Role is read-only for model governance.</p> : null}
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <p className="text-xs uppercase tracking-wider text-slate-500">Recommendation</p>
              <p className="mt-1 text-sm text-slate-700">{championChallenger?.recommendation ?? "--"}</p>
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>Validation Gates</CardTitle>
            <CardDescription>{modelValidation?.decision ?? "No decision"}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {(modelValidation?.gates ?? []).map((gate, idx) => (
              <div key={idx} className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 p-2 text-sm">
                <span>{String(gate.gate ?? "gate")}</span>
                <Badge variant={gate.passed ? "success" : "warning"}>{gate.passed ? "pass" : "fail"}</Badge>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white/90">
          <CardHeader>
            <CardTitle>Drift Monitor</CardTitle>
            <CardDescription>{modelDrift?.recommendation ?? "No recommendation"}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <p className="text-xs uppercase tracking-wider text-slate-500">Drift Status</p>
              <p className="mt-1 text-sm font-medium text-slate-700">{modelDrift?.drift_status ?? "--"}</p>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <p className="text-xs uppercase tracking-wider text-slate-500">Staleness Reason</p>
              <p className="mt-1 text-sm text-slate-700">{modelDrift?.staleness_reason ?? "none"}</p>
            </div>
          </CardContent>
        </Card>
      </section>

      <Card className="border-slate-200 bg-white/90">
        <CardHeader>
          <CardTitle>Recent Training Jobs</CardTitle>
          <CardDescription>Operational view of model-related command jobs</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex flex-wrap gap-2 pb-2">
            <Button className="gap-2" variant="outline" onClick={() => void createJob("train", ["--model", "xgboost", "--symbols", "AAPL", "MSFT"])} disabled={!canCreateJobs}>
              <Cpu size={16} />
              Train XGBoost
            </Button>
            <Button className="gap-2" variant="outline" onClick={() => void createJob("backtest", ["--start", "2025-01-01", "--end", "2025-12-31", "--symbols", "AAPL", "MSFT"])} disabled={!canCreateJobs}>
              <Cpu size={16} />
              Backtest Candidate
            </Button>
          </div>
          {trainingJobs.length === 0 ? (
            <p className="text-sm text-slate-500">No training jobs yet.</p>
          ) : (
            trainingJobs.map((job) => (
              <div key={job.job_id} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div className="flex items-center justify-between">
                  <p className="font-medium text-slate-800">{job.command}</p>
                  <span className="text-sm text-slate-600">{job.status}</span>
                </div>
                <p className="mt-1 truncate font-mono text-xs text-slate-600">{job.args.join(" ")}</p>
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}
