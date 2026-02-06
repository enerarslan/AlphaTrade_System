import { useEffect, useState, type FormEvent } from "react";
import { Navigate } from "react-router-dom";
import { Building2, Lock, UserRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useStore } from "@/lib/store";

export default function LoginPage() {
  const {
    token,
    login,
    authError,
    isLoading,
    ssoStatus,
    fetchSsoStatus,
    startSsoLogin,
    completeSsoLoginFromHash,
  } = useStore();
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("admin");

  useEffect(() => {
    void fetchSsoStatus();
    if (typeof window !== "undefined" && window.location.hash) {
      void completeSsoLoginFromHash(window.location.hash);
    }
  }, [fetchSsoStatus, completeSsoLoginFromHash]);

  if (token) {
    return <Navigate to="/" replace />;
  }

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    await login(username, password);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-[radial-gradient(circle_at_15%_20%,#d7e7ff_0,#f9f8f3_55%,#efece3_100%)] px-4">
      <Card className="w-full max-w-md border-slate-200 bg-white/90 shadow-xl backdrop-blur">
        <CardHeader>
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">AlphaTrade</p>
          <CardTitle className="text-3xl text-slate-900">Operator Login</CardTitle>
          <CardDescription>Control-plane access for live trading operations.</CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={onSubmit}>
            <label className="block text-sm font-medium text-slate-700">
              Username
              <div className="mt-1 flex items-center rounded-lg border border-slate-300 bg-white px-3">
                <UserRound size={16} className="text-slate-500" />
                <input
                  className="h-11 w-full border-0 bg-transparent px-2 outline-none"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username"
                />
              </div>
            </label>

            <label className="block text-sm font-medium text-slate-700">
              Password
              <div className="mt-1 flex items-center rounded-lg border border-slate-300 bg-white px-3">
                <Lock size={16} className="text-slate-500" />
                <input
                  className="h-11 w-full border-0 bg-transparent px-2 outline-none"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                />
              </div>
            </label>

            {authError ? <p className="text-sm text-red-600">{authError}</p> : null}

            <Button className="h-11 w-full bg-slate-900 text-white hover:bg-slate-800" type="submit" disabled={isLoading}>
              {isLoading ? "Signing in..." : "Sign In"}
            </Button>
            {ssoStatus?.enabled ? (
              <Button
                className="h-11 w-full gap-2"
                variant="outline"
                type="button"
                onClick={() => startSsoLogin()}
                disabled={!ssoStatus.configured || isLoading}
              >
                <Building2 size={16} />
                {ssoStatus.configured ? "Sign In with Enterprise SSO" : "SSO Unavailable"}
              </Button>
            ) : null}
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
