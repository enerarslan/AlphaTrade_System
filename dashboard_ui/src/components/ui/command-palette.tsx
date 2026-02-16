import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Command } from "cmdk";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertTriangle,
  BarChart3,
  Bot,
  Boxes,
  Gauge,
  Search,
  Settings,
  ShieldAlert,
  Workflow,
  Zap,
  type LucideIcon,
} from "lucide-react";
import { useStore } from "@/lib/store";

type CmdItem = { label: string; icon: LucideIcon; action: () => void; group: string; keywords?: string };

export default function CommandPalette() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const { activateKillSwitch, stopTrading, startTrading, tradingStatus } = useStore();

  // Ctrl+K to toggle
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      }
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, []);

  const go = (path: string) => {
    navigate(path);
    setOpen(false);
  };

  const items: CmdItem[] = [
    // Navigation
    { label: "Command Center", icon: Gauge, action: () => go("/"), group: "Navigate", keywords: "overview home dashboard" },
    { label: "Execution Desk", icon: BarChart3, action: () => go("/trading"), group: "Navigate", keywords: "trading blotter orders" },
    { label: "Platform Intelligence", icon: Boxes, action: () => go("/platform"), group: "Navigate", keywords: "scripts coverage" },
    { label: "Risk War Room", icon: ShieldAlert, action: () => go("/risk"), group: "Navigate", keywords: "var drawdown exposure" },
    { label: "Model Intelligence", icon: Bot, action: () => go("/models"), group: "Navigate", keywords: "ml drift champion" },
    { label: "Operations Console", icon: Workflow, action: () => go("/operations"), group: "Navigate", keywords: "sre health slo" },
    { label: "Alert Center", icon: AlertTriangle, action: () => go("/alerts"), group: "Navigate", keywords: "alerts notifications" },
    { label: "Settings & Security", icon: Settings, action: () => go("/settings"), group: "Navigate", keywords: "mfa jwt sso roles" },
    // Actions
    {
      label: tradingStatus?.running ? "Stop Trading Engine" : "Start Trading Engine",
      icon: Zap,
      action: () => {
        if (tradingStatus?.running) {
          void stopTrading();
        } else {
          void startTrading({ mode: "paper", strategy: "default", capital: 100000, symbols: [] });
        }
        setOpen(false);
      },
      group: "Actions",
      keywords: "engine start stop",
    },
    {
      label: "Activate Kill Switch",
      icon: ShieldAlert,
      action: () => {
        if (confirm("Are you sure you want to activate the kill switch?")) {
          void activateKillSwitch("Manual kill switch via command palette");
        }
        setOpen(false);
      },
      group: "Actions",
      keywords: "kill switch emergency",
    },
  ];

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
            onClick={() => setOpen(false)}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: -20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: -20 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
            className="fixed inset-x-0 top-[15%] z-50 mx-auto w-full max-w-xl"
          >
            <Command
              className="overflow-hidden rounded-2xl border border-white/10 bg-slate-900/95 shadow-2xl backdrop-blur-2xl"
              label="Command Palette"
            >
              <div className="flex items-center border-b border-white/[0.06] px-4">
                <Search className="mr-3 h-4 w-4 text-slate-500" />
                <Command.Input
                  placeholder="Type a command or search..."
                  className="h-12 w-full bg-transparent text-sm text-slate-200 outline-none placeholder:text-slate-500"
                />
                <kbd className="ml-2 rounded border border-white/10 bg-white/5 px-1.5 py-0.5 text-[10px] font-medium text-slate-500">
                  ESC
                </kbd>
              </div>
              <Command.List className="max-h-80 overflow-y-auto px-2 py-2">
                <Command.Empty className="py-6 text-center text-sm text-slate-500">
                  No results found.
                </Command.Empty>
                {["Navigate", "Actions"].map((group) => (
                  <Command.Group
                    key={group}
                    heading={group}
                    className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-widest [&_[cmdk-group-heading]]:text-slate-500"
                  >
                    {items
                      .filter((i) => i.group === group)
                      .map((item) => (
                        <Command.Item
                          key={item.label}
                          value={`${item.label} ${item.keywords ?? ""}`}
                          onSelect={item.action}
                          className="flex cursor-pointer items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-slate-300 transition-colors data-[selected=true]:bg-cyan-500/10 data-[selected=true]:text-cyan-300"
                        >
                          <item.icon className="h-4 w-4 text-slate-500" />
                          {item.label}
                        </Command.Item>
                      ))}
                  </Command.Group>
                ))}
              </Command.List>
            </Command>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
