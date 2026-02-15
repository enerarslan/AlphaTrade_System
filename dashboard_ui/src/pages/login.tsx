import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Zap, Lock, ArrowRight, Globe } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useStore } from "@/lib/store";

export default function LoginPage() {
  const { login, authError, isLoading, fetchSsoStatus, ssoStatus, startSsoLogin, completeSsoLoginFromHash } = useStore();
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  useEffect(() => {
    void fetchSsoStatus();
  }, [fetchSsoStatus]);

  useEffect(() => {
    if (typeof window !== "undefined" && window.location.hash) {
      void completeSsoLoginFromHash(window.location.hash).then((ok) => {
        if (ok) navigate("/", { replace: true });
      });
    }
  }, [completeSsoLoginFromHash, navigate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const ok = await login(username, password);
    if (ok) navigate("/", { replace: true });
  };

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden">
      {/* Background effects */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_30%_20%,rgba(6,182,212,0.08),transparent_60%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_70%_80%,rgba(16,185,129,0.06),transparent_60%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_50%_50%,rgba(99,102,241,0.04),transparent_60%)]" />

      {/* Grid overlay */}
      <div className="pointer-events-none absolute inset-0 mission-grid opacity-40" />

      <motion.div
        initial={{ opacity: 0, y: 30, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className="relative z-10 w-full max-w-md px-4"
      >
        {/* Logo */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15, duration: 0.5 }}
          className="mb-8 text-center"
        >
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-cyan-500/20 shadow-[0_0_30px_rgba(6,182,212,0.2)]">
            <Zap size={28} className="text-cyan-400" />
          </div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.3em] text-cyan-500/70">AlphaTrade System</p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight text-slate-100">Control Plane</h1>
          <p className="mt-2 text-sm text-slate-500">Institutional-grade quantitative trading platform</p>
        </motion.div>

        {/* Login Card */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
          className="rounded-2xl border border-white/[0.08] bg-white/[0.04] p-8 shadow-2xl backdrop-blur-xl"
        >
          <form onSubmit={(e) => void handleSubmit(e)} className="space-y-5">
            <div>
              <label className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-slate-400">
                Username
              </label>
              <input
                className="glass-input h-11 w-full"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
                autoFocus
              />
            </div>

            <div>
              <label className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-slate-400">
                Password
              </label>
              <input
                className="glass-input h-11 w-full"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password"
              />
            </div>

            {authError && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                className="rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-sm text-rose-400"
              >
                {authError}
              </motion.div>
            )}

            <Button
              type="submit"
              className="w-full gap-2"
              disabled={isLoading || !username || !password}
            >
              <Lock size={16} />
              {isLoading ? "Authenticating..." : "Sign In"}
              {!isLoading && <ArrowRight size={16} />}
            </Button>
          </form>

          {ssoStatus?.enabled && ssoStatus?.configured && (
            <>
              <div className="my-5 flex items-center gap-3">
                <div className="h-px flex-1 bg-white/[0.08]" />
                <span className="text-xs text-slate-500">OR</span>
                <div className="h-px flex-1 bg-white/[0.08]" />
              </div>
              <Button
                variant="outline"
                className="w-full gap-2"
                onClick={startSsoLogin}
              >
                <Globe size={16} />
                Enterprise SSO Login
              </Button>
            </>
          )}
        </motion.div>

        {/* Footer */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          className="mt-6 text-center text-xs text-slate-600"
        >
          AlphaTrade v1.3.0 — Institutional Quantitative Trading Platform
        </motion.p>
      </motion.div>
    </div>
  );
}
