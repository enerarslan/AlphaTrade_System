import { useMemo, useState, type ElementType } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bot,
  BriefcaseBusiness,
  Gauge,
  LogOut,
  Menu,
  Settings,
  ShieldAlert,
  Workflow,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { hasPermissionForRole, useStore } from "@/lib/store";

type NavItem = {
  icon: ElementType;
  label: string;
  href: string;
  permission?: string;
};

const navItems: NavItem[] = [
  { icon: Gauge, label: "Command Center", href: "/" },
  { icon: BriefcaseBusiness, label: "Execution", href: "/trading", permission: "control.trading.status" },
  { icon: ShieldAlert, label: "Risk", href: "/risk", permission: "risk.advanced.read" },
  { icon: Bot, label: "Models", href: "/models", permission: "models.governance.read" },
  { icon: Workflow, label: "Operations", href: "/operations", permission: "operations.sre.read" },
  { icon: AlertTriangle, label: "Alerts", href: "/alerts" },
  { icon: Settings, label: "Settings", href: "/settings", permission: "control.risk.kill_switch.reset" },
];

export function DashboardLayout({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();
  const [mobileOpen, setMobileOpen] = useState(false);
  const { user, role, mfaStatus, ws, tradingStatus, logout, alerts, lastRefreshAt } = useStore();

  const activeAlerts = useMemo(
    () => alerts.filter((x) => x.status !== "RESOLVED").length,
    [alerts],
  );
  const visibleNavItems = useMemo(
    () => navItems.filter((item) => !item.permission || hasPermissionForRole(role, item.permission)),
    [role],
  );

  const wsAllConnected = ws.portfolio && ws.orders && ws.signals && ws.alerts;

  const onLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div className="min-h-screen w-full bg-[radial-gradient(circle_at_20%_10%,#dce9ff_0,#f5f8ff_40%,#f8f7f3_100%)] text-slate-900">
      <div className="mx-auto flex max-w-[1800px]">
        <aside
          className={cn(
            "fixed inset-y-0 left-0 z-40 w-72 border-r border-slate-200 bg-white/90 p-5 backdrop-blur-xl transition-transform lg:relative lg:translate-x-0",
            mobileOpen ? "translate-x-0" : "-translate-x-full",
          )}
        >
          <div className="mb-8 flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">AlphaTrade</p>
              <h1 className="text-2xl font-bold tracking-tight text-slate-900">Control Plane</h1>
            </div>
            <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setMobileOpen(false)}>
              <X size={18} />
            </Button>
          </div>

          <nav className="space-y-1.5">
            {visibleNavItems.map((item) => {
              const active = location.pathname === item.href;
              return (
                <Link key={item.href} to={item.href} onClick={() => setMobileOpen(false)}>
                  <div
                    className={cn(
                      "group flex items-center justify-between rounded-xl border px-4 py-3 text-sm transition",
                      active
                        ? "border-sky-300 bg-sky-50 text-sky-900 shadow-sm"
                        : "border-transparent text-slate-600 hover:border-slate-200 hover:bg-white",
                    )}
                  >
                    <div className="flex items-center gap-3">
                      <item.icon size={18} />
                      <span className="font-medium">{item.label}</span>
                    </div>
                    {item.href === "/alerts" && activeAlerts > 0 ? (
                      <Badge variant="warning">{activeAlerts}</Badge>
                    ) : null}
                  </div>
                </Link>
              );
            })}
          </nav>

          <div className="mt-8 rounded-xl border border-slate-200 bg-slate-50/70 p-4">
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Session</p>
            <p className="mt-1 text-sm text-slate-700">{user?.username ?? "operator"}</p>
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <Badge variant="outline">{role.toUpperCase()}</Badge>
              <Badge variant={mfaStatus?.mfa_enabled ? "success" : "warning"}>
                MFA {mfaStatus?.mfa_enabled ? "ON" : "OFF"}
              </Badge>
            </div>
            <Button className="mt-4 w-full justify-start gap-2 bg-slate-900 text-white hover:bg-slate-800" onClick={onLogout}>
              <LogOut size={16} />
              Sign Out
            </Button>
          </div>
        </aside>

        {mobileOpen ? (
          <button
            className="fixed inset-0 z-30 bg-slate-900/40 lg:hidden"
            onClick={() => setMobileOpen(false)}
            aria-label="Close menu"
          />
        ) : null}

        <div className="min-h-screen flex-1">
          <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/80 px-4 py-3 backdrop-blur-xl lg:px-8">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <Button variant="outline" size="icon" className="lg:hidden" onClick={() => setMobileOpen(true)}>
                  <Menu size={18} />
                </Button>
                <h2 className="text-lg font-semibold tracking-tight text-slate-900">
                  {visibleNavItems.find((x) => x.href === location.pathname)?.label ?? "Dashboard"}
                </h2>
              </div>

              <div className="flex items-center gap-2">
                <Badge variant={wsAllConnected ? "success" : "warning"}>
                  <Activity size={12} className="mr-1" />
                  {wsAllConnected ? "Live Feeds" : "Reconnect"}
                </Badge>
                <Badge variant={tradingStatus?.running ? "success" : "outline"}>
                  <BarChart3 size={12} className="mr-1" />
                  {tradingStatus?.running ? `Trading PID ${tradingStatus.pid}` : "Engine Idle"}
                </Badge>
                <span className="hidden text-xs text-slate-500 md:block">
                  Last refresh: {lastRefreshAt ? new Date(lastRefreshAt).toLocaleTimeString() : "--"}
                </span>
              </div>
            </div>
          </header>

          <main className="px-4 py-6 lg:px-8">{children}</main>
        </div>
      </div>
    </div>
  );
}
