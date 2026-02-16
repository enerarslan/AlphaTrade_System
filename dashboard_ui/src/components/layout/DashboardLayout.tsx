import { useEffect, useMemo, useState, type ElementType } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertTriangle,
  BarChart3,
  Boxes,
  Bot,
  BriefcaseBusiness,
  Gauge,
  LogOut,
  Menu,
  Settings,
  ShieldAlert,
  Workflow,
  X,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { hasPermissionForRole, useStore } from "@/lib/store";
import ConnectionStatus from "@/components/live/ConnectionStatus";
import CommandPalette from "@/components/ui/command-palette";
import { NotificationToaster } from "@/lib/notifications";
import HotkeyHelp from "@/components/ui/hotkey-help";

type NavItem = {
  icon: ElementType;
  label: string;
  href: string;
  permission?: string;
};

const navItems: NavItem[] = [
  { icon: Gauge, label: "Command Center", href: "/" },
  { icon: BriefcaseBusiness, label: "Execution", href: "/trading", permission: "control.trading.status" },
  { icon: Boxes, label: "Platform", href: "/platform", permission: "read.basic" },
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
  const { user, role, mfaStatus, tradingStatus, logout, alerts, lastRefreshAt } = useStore();

  const activeAlerts = useMemo(
    () => alerts.filter((x) => x.status !== "RESOLVED").length,
    [alerts],
  );
  const visibleNavItems = useMemo(
    () => navItems.filter((item) => !item.permission || hasPermissionForRole(role, item.permission)),
    [role],
  );

  const { connectLiveChannels, disconnectLiveChannels } = useStore();

  // Auto-connect WebSocket channels
  useEffect(() => {
    connectLiveChannels();
    return () => disconnectLiveChannels();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const onLogout = () => {
    disconnectLiveChannels();
    logout();
    navigate("/login");
  };

  return (
    <div className="min-h-screen w-full text-slate-200">
      <CommandPalette />
      <NotificationToaster />
      <HotkeyHelp />
      {/* Ambient glow orbs */}
      <div className="pointer-events-none fixed left-[-180px] top-[-200px] h-[400px] w-[400px] rounded-full bg-cyan-500/[0.06] blur-[100px]" />
      <div className="pointer-events-none fixed bottom-[-150px] right-[-120px] h-[350px] w-[350px] rounded-full bg-emerald-500/[0.05] blur-[100px]" />
      <div className="pointer-events-none fixed right-[30%] top-[40%] h-[300px] w-[300px] rounded-full bg-indigo-500/[0.03] blur-[100px]" />

      <div className="mx-auto flex max-w-[1800px]">
        {/* Sidebar */}
        <aside
          className={cn(
            "fixed inset-y-0 left-0 z-40 w-72 border-r border-white/[0.06] bg-slate-950/80 p-5 backdrop-blur-2xl transition-transform lg:relative lg:translate-x-0",
            mobileOpen ? "translate-x-0" : "-translate-x-full",
          )}
        >
          <div className="mb-8 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-cyan-500/20 text-cyan-400">
                <Zap size={18} />
              </div>
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-[0.24em] text-cyan-500/70">AlphaTrade</p>
                <h1 className="text-lg font-bold tracking-tight text-slate-100">Control Plane</h1>
              </div>
            </div>
            <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setMobileOpen(false)}>
              <X size={18} />
            </Button>
          </div>

          <nav className="space-y-1">
            {visibleNavItems.map((item) => {
              const active = location.pathname === item.href;
              return (
                <Link key={item.href} to={item.href} onClick={() => setMobileOpen(false)}>
                  <motion.div
                    whileHover={{ x: 2 }}
                    className={cn(
                      "group flex items-center justify-between rounded-xl border px-4 py-3 text-sm transition-all duration-200",
                      active
                        ? "border-cyan-500/30 bg-cyan-500/10 text-cyan-300 shadow-[0_0_15px_rgba(6,182,212,0.1)]"
                        : "border-transparent text-slate-400 hover:border-white/[0.08] hover:bg-white/[0.04] hover:text-slate-200",
                    )}
                  >
                    <div className="flex items-center gap-3">
                      <item.icon size={18} className={active ? "text-cyan-400" : ""} />
                      <span className="font-medium">{item.label}</span>
                    </div>
                    {item.href === "/alerts" && activeAlerts > 0 ? (
                      <Badge variant="warning">{activeAlerts}</Badge>
                    ) : null}
                    {active && (
                      <div className="h-1.5 w-1.5 rounded-full bg-cyan-400 shadow-[0_0_6px_rgba(6,182,212,0.6)]" />
                    )}
                  </motion.div>
                </Link>
              );
            })}
          </nav>

          {/* Session Card */}
          <div className="mt-8 rounded-xl border border-white/[0.08] bg-white/[0.03] p-4 backdrop-blur-sm">
            <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">Session</p>
            <p className="mt-1 text-sm font-medium text-slate-200">{user?.username ?? "operator"}</p>
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <Badge variant="outline">{role.toUpperCase()}</Badge>
              <Badge variant={mfaStatus?.mfa_enabled ? "success" : "warning"}>
                MFA {mfaStatus?.mfa_enabled ? "ON" : "OFF"}
              </Badge>
            </div>
            <Button className="mt-4 w-full justify-start gap-2" variant="outline" onClick={onLogout}>
              <LogOut size={16} />
              Sign Out
            </Button>
          </div>
        </aside>

        {/* Mobile overlay */}
        <AnimatePresence>
          {mobileOpen && (
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm lg:hidden"
              onClick={() => setMobileOpen(false)}
              aria-label="Close menu"
            />
          )}
        </AnimatePresence>

        {/* Main content */}
        <div className="min-h-screen flex-1">
          <header className="sticky top-0 z-20 border-b border-white/[0.06] bg-slate-950/70 px-4 py-3 backdrop-blur-2xl lg:px-8">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setMobileOpen(true)}>
                  <Menu size={18} />
                </Button>
                <h2 className="text-lg font-semibold tracking-tight text-slate-100">
                  {visibleNavItems.find((x) => x.href === location.pathname)?.label ?? "Dashboard"}
                </h2>
              </div>

              <div className="flex items-center gap-2">
                <ConnectionStatus />
                <Badge variant={tradingStatus?.running ? "success" : "outline"}>
                  <BarChart3 size={12} className="mr-1" />
                  {tradingStatus?.running ? `PID ${tradingStatus.pid}` : "Engine Idle"}
                </Badge>
                <span className="hidden text-xs text-slate-500 md:block">
                  {lastRefreshAt ? new Date(lastRefreshAt).toLocaleTimeString() : "--"}
                </span>
              </div>
            </div>
          </header>

          <main className="px-4 py-6 lg:px-8">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
            >
              {children}
            </motion.div>
          </main>
        </div>
      </div>
    </div>
  );
}
