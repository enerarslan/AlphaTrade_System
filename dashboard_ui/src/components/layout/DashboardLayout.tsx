import { useEffect, useMemo, useState, type ElementType } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { useShallow } from "zustand/react/shallow";
import {
  AlertTriangle,
  BarChart2,
  BarChart3,
  Bot,
  Boxes,
  BriefcaseBusiness,
  Database,
  FlaskConical,
  Gauge,
  LogOut,
  Menu,
  Settings,
  ShieldAlert,
  Workflow,
  X,
  Zap,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import ConnectionStatus from "@/components/live/ConnectionStatus";
import MarketTicker from "@/components/live/MarketTicker";
import { cn } from "@/lib/utils";
import { hasPermissionForRole, useStore } from "@/lib/store";
import CommandPalette from "@/components/ui/command-palette";
import { NotificationToaster } from "@/lib/notification-toaster";
import HotkeyHelp from "@/components/ui/hotkey-help";
import ParticleBackground from "@/components/ui/ParticleBackground";
import LiveClock from "@/components/ui/LiveClock";
import FearGreedGauge from "@/components/ui/FearGreedGauge";
import NotificationPanel from "@/components/ui/NotificationPanel";
import AICopilotWidget from "@/components/ui/AICopilotWidget";

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
  { icon: BarChart2, label: "Analytics", href: "/analytics", permission: "read.basic" },
  { icon: Database, label: "Alt Data & News", href: "/alt-data", permission: "read.basic" },
  { icon: ShieldAlert, label: "Risk", href: "/risk", permission: "risk.advanced.read" },
  { icon: Bot, label: "Models", href: "/models", permission: "models.governance.read" },
  { icon: FlaskConical, label: "Backtest Lab", href: "/backtest", permission: "control.jobs.create" },
  { icon: Workflow, label: "Operations", href: "/operations", permission: "operations.sre.read" },
  { icon: Database, label: "Database", href: "/database", permission: "read.basic" },
  { icon: AlertTriangle, label: "Alerts", href: "/alerts" },
  { icon: Settings, label: "Settings", href: "/settings", permission: "control.risk.kill_switch.reset" },
];

function shouldEnableVisualEffects() {
  if (typeof window === "undefined") return true;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const lowPower = (navigator.hardwareConcurrency ?? 8) <= 4 || window.innerWidth < 960;
  return !reducedMotion && !lowPower;
}

export function DashboardLayout({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();
  const [mobileOpen, setMobileOpen] = useState(false);
  // AI Copilot State
  const [isCopilotOpen, setIsCopilotOpen] = useState(false);
  const [visualEffectsEnabled, setVisualEffectsEnabled] = useState(shouldEnableVisualEffects);

  const { user, role, mfaStatus, tradingStatus, logout, activeAlerts, lastRefreshAt } = useStore(
    useShallow((state) => ({
      user: state.user,
      role: state.role,
      mfaStatus: state.mfaStatus,
      tradingStatus: state.tradingStatus,
      logout: state.logout,
      activeAlerts: state.alerts.reduce(
        (count, alert) => (alert.status !== "RESOLVED" ? count + 1 : count),
        0,
      ),
      lastRefreshAt: state.lastRefreshAt,
    })),
  );

  const connectLiveChannels = useStore((state) => state.connectLiveChannels);
  const disconnectLiveChannels = useStore((state) => state.disconnectLiveChannels);

  const visibleNavItems = useMemo(
    () => navItems.filter((item) => !item.permission || hasPermissionForRole(role, item.permission)),
    [role],
  );

  useEffect(() => {
    connectLiveChannels();
    return () => disconnectLiveChannels();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");

    const recompute = () => {
      setVisualEffectsEnabled(shouldEnableVisualEffects());
    };

    recompute();
    window.addEventListener("resize", recompute, { passive: true });
    media.addEventListener("change", recompute);

    return () => {
      window.removeEventListener("resize", recompute);
      media.removeEventListener("change", recompute);
    };
  }, []);

  const onLogout = () => {
    disconnectLiveChannels();
    logout();
    navigate("/login");
  };

  return (
    <div className="h-screen w-screen overflow-hidden text-slate-200">
      <CommandPalette />
      <NotificationToaster />
      <HotkeyHelp />
      {visualEffectsEnabled && <ParticleBackground />}

      {visualEffectsEnabled && (
        <>
          <div
            className="pointer-events-none fixed left-[-10%] top-[-10%] h-[40vh] w-[40vh] rounded-full bg-cyan-500/[0.04] blur-[80px]"
            style={{ transform: "translateZ(0)" }}
          />
          <div
            className="pointer-events-none fixed bottom-[-10%] right-[-5%] h-[35vh] w-[35vh] rounded-full bg-emerald-500/[0.03] blur-[80px]"
            style={{ transform: "translateZ(0)" }}
          />
        </>
      )}

      <div className="flex h-full w-full">
        <aside
          className={cn(
            "fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-white/[0.06] bg-slate-950/90 p-4 transition-transform lg:relative lg:translate-x-0",
            mobileOpen ? "translate-x-0" : "-translate-x-full",
          )}
        >
          <div className="mb-6 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-500/20 text-cyan-400">
                <Zap size={16} />
              </div>
              <div>
                <p className="text-[9px] font-semibold uppercase tracking-[0.24em] text-cyan-500/60">AlphaTrade</p>
                <p className="text-sm font-bold tracking-tight text-slate-100">Control Plane</p>
              </div>
            </div>
            <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setMobileOpen(false)}>
              <X size={16} />
            </Button>
          </div>

          <nav className="flex-1 space-y-0.5 overflow-y-auto">
            {visibleNavItems.map((item) => {
              const active = location.pathname === item.href;
              return (
                <Link key={item.href} to={item.href} onClick={() => setMobileOpen(false)}>
                  <div
                    className={cn(
                      "flex items-center justify-between rounded-lg border px-3 py-2.5 text-sm transition-colors",
                      active
                        ? "border-cyan-500/20 bg-cyan-500/10 text-cyan-300"
                        : "border-transparent text-slate-400 hover:bg-white/[0.04] hover:text-slate-200",
                    )}
                  >
                    <div className="flex items-center gap-2.5">
                      <item.icon size={16} className={active ? "text-cyan-400" : ""} />
                      <span className="font-medium">{item.label}</span>
                    </div>
                    {item.href === "/alerts" && activeAlerts > 0 ? <Badge variant="warning">{activeAlerts}</Badge> : null}
                    {active && <div className="h-1 w-1 rounded-full bg-cyan-400" />}
                  </div>
                </Link>
              );
            })}
          </nav>

          <div className="mt-auto space-y-3 pt-3">
            <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
              <p className="text-[9px] font-semibold uppercase tracking-[0.16em] text-slate-500">Session</p>
              <p className="mt-0.5 text-sm font-medium text-slate-200">{user?.username ?? "operator"}</p>
              <div className="mt-1.5 flex flex-wrap items-center gap-1">
                <Badge variant="outline">{role.toUpperCase()}</Badge>
                <Badge variant={mfaStatus?.mfa_enabled ? "success" : "warning"}>
                  MFA {mfaStatus?.mfa_enabled ? "ON" : "OFF"}
                </Badge>
              </div>
              <Button className="mt-3 w-full justify-start gap-2" variant="outline" size="sm" onClick={onLogout}>
                <LogOut size={14} />
                Sign Out
              </Button>
            </div>

            <LiveClock />

            <div>
              <p className="mb-1.5 text-[9px] font-semibold uppercase tracking-[0.2em] text-slate-500">Sentiment</p>
              <FearGreedGauge />
            </div>

            <p className="text-center text-[8px] font-mono text-slate-600">AlphaTrade v2.0.0</p>
          </div>
        </aside>

        <AnimatePresence>
          {mobileOpen && (
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-30 bg-black/60 lg:hidden"
              onClick={() => setMobileOpen(false)}
              aria-label="Close menu"
            />
          )}
        </AnimatePresence>

        <div className="flex flex-1 flex-col overflow-hidden">
          <MarketTicker />

          <header className="z-20 flex items-center justify-between border-b border-white/[0.06] bg-slate-950/70 px-4 py-2 lg:px-6">
            <div className="flex items-center gap-3">
              <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setMobileOpen(true)}>
                <Menu size={16} />
              </Button>
              <h2 className="text-base font-semibold tracking-tight text-slate-100">
                {visibleNavItems.find((item) => item.href === location.pathname)?.label ?? "Dashboard"}
              </h2>
            </div>
            <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="sm"
              className={`gap-2 h-9 border-white/[0.08] ${isCopilotOpen ? "bg-fuchsia-500/20 text-fuchsia-300 border-fuchsia-500/30" : "bg-white/[0.02] text-slate-300 hover:text-white"}`}
              onClick={() => setIsCopilotOpen(!isCopilotOpen)}
            >
              <Bot size={16} className={isCopilotOpen ? "text-fuchsia-400" : ""} />
              Copilot
            </Button>
              <ConnectionStatus />
              <NotificationPanel />
              <Badge variant={tradingStatus?.running ? "success" : "outline"}>
                <BarChart3 size={12} className="mr-1" />
                {tradingStatus?.running ? `PID ${tradingStatus.pid}` : "Engine Idle"}
              </Badge>
              <span className="hidden text-[10px] text-slate-500 md:block">
                {lastRefreshAt ? new Date(lastRefreshAt).toLocaleTimeString() : "--"}
              </span>
            </div>
          </header>

          <main className="flex-1 overflow-y-auto overflow-x-hidden px-4 py-5 lg:px-6">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: visualEffectsEnabled ? 0.2 : 0.01, ease: "easeOut" }}
            >
              {children}
            </motion.div>
          </main>
        </div>
        
        {/* Persistent AI Copilot Sidebar / Drawer */}
        <AICopilotWidget isOpen={isCopilotOpen} onClose={() => setIsCopilotOpen(false)} />
      </div>
    </div>
  );
}
