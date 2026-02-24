import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, Lock, ArrowRight, Globe, Eye, EyeOff, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useStore } from "@/lib/store";
import { useShallow } from "zustand/react/shallow";

export default function LoginPage() {
  const { login, authError, isLoading, fetchSsoStatus, ssoStatus, startSsoLogin, completeSsoLoginFromHash } = useStore(useShallow((state) => ({
      login: state.login,
      authError: state.authError,
      isLoading: state.isLoading,
      fetchSsoStatus: state.fetchSsoStatus,
      ssoStatus: state.ssoStatus,
      startSsoLogin: state.startSsoLogin,
      completeSsoLoginFromHash: state.completeSsoLoginFromHash,
    })));
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [focused, setFocused] = useState<string | null>(null);

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

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    const ok = await login(username, password);
    if (ok) navigate("/", { replace: true });
  }, [login, username, password, navigate]);

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[#04080f]">
      {/* Static radial glows (no canvas, no blur layers — pure CSS) */}
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute left-[20%] top-[20%] h-[40vh] w-[40vh] rounded-full bg-cyan-500/[0.04] blur-[60px]" />
        <div className="absolute bottom-[20%] right-[20%] h-[30vh] w-[30vh] rounded-full bg-indigo-500/[0.03] blur-[50px]" />
      </div>

      {/* Grid overlay */}
      <div className="pointer-events-none absolute inset-0 mission-grid opacity-15" />

      <motion.div
        initial={{ opacity: 0, y: 24, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className="relative z-10 w-full max-w-md px-4"
      >
        {/* Logo */}
        <div className="mb-10 text-center">
          <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-2xl border border-cyan-500/20 bg-cyan-500/10">
            <Zap size={30} className="text-cyan-400" />
          </div>
          <p className="text-[10px] font-bold uppercase tracking-[0.35em] text-cyan-500/60">AlphaTrade System</p>
          <h1 className="mt-2 text-4xl font-bold tracking-tight text-slate-100">
            Control <span className="gradient-text">Plane</span>
          </h1>
          <p className="mt-3 text-sm text-slate-500">Institutional-grade quantitative trading platform</p>
        </div>

        {/* Login Card */}
        <div className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-8 shadow-2xl">
          <form onSubmit={(e) => void handleSubmit(e)} className="space-y-5">
            {/* Username */}
            <div>
              <label className="mb-1.5 flex items-center gap-1 text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
                <Shield size={10} />
                Operator ID
              </label>
              <div className={`relative rounded-xl border transition-colors ${focused === "user" ? "border-cyan-500/40" : "border-white/[0.08]"}`}>
                <input
                  className="h-12 w-full rounded-xl bg-white/[0.03] px-4 text-sm text-slate-100 placeholder-slate-600 outline-none"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  onFocus={() => setFocused("user")}
                  onBlur={() => setFocused(null)}
                  placeholder="Enter operator ID"
                  autoFocus
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="mb-1.5 flex items-center gap-1 text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
                <Lock size={10} />
                Passphrase
              </label>
              <div className={`relative rounded-xl border transition-colors ${focused === "pass" ? "border-cyan-500/40" : "border-white/[0.08]"}`}>
                <input
                  className="h-12 w-full rounded-xl bg-white/[0.03] px-4 pr-12 text-sm text-slate-100 placeholder-slate-600 outline-none"
                  type={showPw ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onFocus={() => setFocused("pass")}
                  onBlur={() => setFocused(null)}
                  placeholder="Enter passphrase"
                />
                <button
                  type="button"
                  onClick={() => setShowPw(!showPw)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 transition-colors hover:text-slate-300"
                >
                  {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            {/* Auth Error */}
            <AnimatePresence>
              {authError && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-sm text-rose-400">
                    {authError}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Submit */}
            <Button
              type="submit"
              className="w-full gap-2 h-12 text-sm font-semibold"
              disabled={isLoading || !username || !password}
            >
              {isLoading ? (
                <>
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                  Authenticating...
                </>
              ) : (
                <>
                  <Lock size={14} />
                  Authenticate
                  <ArrowRight size={14} />
                </>
              )}
            </Button>
          </form>

          {/* SSO */}
          {ssoStatus?.enabled && ssoStatus?.configured && (
            <>
              <div className="my-5 flex items-center gap-3">
                <div className="h-px flex-1 bg-gradient-to-r from-transparent via-white/[0.08] to-transparent" />
                <span className="text-[9px] font-bold uppercase tracking-[0.2em] text-slate-600">or</span>
                <div className="h-px flex-1 bg-gradient-to-l from-transparent via-white/[0.08] to-transparent" />
              </div>
              <Button
                variant="outline"
                className="w-full gap-2 h-11"
                onClick={startSsoLogin}
              >
                <Globe size={16} />
                Enterprise SSO Login
              </Button>
            </>
          )}
        </div>

        {/* Footer */}
        <p className="mt-8 text-center text-[10px] text-slate-700">
          AlphaTrade v1.3.0 · Encrypted connection · All sessions monitored
        </p>
      </motion.div>
    </div>
  );
}


