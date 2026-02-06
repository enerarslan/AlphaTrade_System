import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { 
  LayoutGrid, 
  Activity, 
  LineChart, 
  ShieldAlert, 
  FileText, 
  Settings,
  Menu,
  X,
  Zap,
  Cpu,
  Globe
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface SidebarItem {
  icon: React.ElementType;
  label: string;
  href: string;
}

const sidebarItems: SidebarItem[] = [
  { icon: LayoutGrid, label: "Command Center", href: "/" },
  { icon: Activity, label: "System Status", href: "/system-status" },
  { icon: LineChart, label: "Trading Terminal", href: "/trading" },
  { icon: ShieldAlert, label: "Risk War Room", href: "/risk" },
  { icon: FileText, label: "Logs", href: "/logs" },
  { icon: Settings, label: "Settings", href: "/settings" },
];

export function DashboardLayout({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  return (
    <div className="flex h-screen w-full bg-[#050505] text-foreground overflow-hidden font-sans selection:bg-primary/20">
      {/* Background Animated Grid */}
      <div className="absolute inset-0 z-0 opacity-10 pointer-events-none animate-grid" />
      
      {/* Framer Motion Sidebar */}
      <motion.aside
        initial={false}
        animate={{ width: isSidebarOpen ? 280 : 80 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
        className="relative z-50 flex flex-col border-r border-white/5 bg-black/60 backdrop-blur-2xl shadow-2xl"
      >
        {/* Logo Area */}
        <div className="relative flex h-20 items-center justify-between px-6 border-b border-white/5">
          <AnimatePresence mode="wait">
            {isSidebarOpen ? (
              <motion.div 
                key="logo-full"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-2"
              >
                <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-primary to-violet-600 flex items-center justify-center shadow-lg shadow-primary/20">
                  <Zap className="h-5 w-5 text-white fill-white" />
                </div>
                <div className="flex flex-col">
                  <span className="text-lg font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-white/70">
                    AlphaTrade
                  </span>
                  <span className="text-[10px] text-primary font-medium tracking-widest uppercase">
                    Pro Terminal
                  </span>
                </div>
              </motion.div>
            ) : (
               <motion.div 
                key="logo-icon"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="w-full flex justify-center"
               >
                 <Zap className="h-6 w-6 text-primary fill-primary/20" />
               </motion.div>
            )}
          </AnimatePresence>

          {isSidebarOpen && (
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsSidebarOpen(false)}
                className="text-muted-foreground hover:text-white"
              >
                <X size={18} />
              </Button>
          )}
        </div>
        
        {!isSidebarOpen && (
            <div className="flex justify-center py-4 border-b border-white/5">
                 <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setIsSidebarOpen(true)}
                    className="text-muted-foreground hover:text-white"
                  >
                    <Menu size={20} />
                  </Button>
            </div>
        )}

        {/* Navigation */}
        <nav className="flex-1 space-y-1 p-4 overflow-y-auto">
          {sidebarItems.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.href}
                to={item.href}
                className="block outline-none"
              >
                <div
                    className={cn(
                    "group flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all duration-200 relative overflow-hidden",
                    isActive 
                        ? "bg-primary/10 text-primary shadow-inner shadow-primary/5" 
                        : "text-muted-foreground hover:text-white hover:bg-white/5",
                    !isSidebarOpen && "justify-center px-2"
                    )}
                >
                    {isActive && (
                    <motion.div 
                        layoutId="active-pill"
                        className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-primary rounded-r-full shadow-[0_0_12px_rgba(var(--primary),0.5)]" 
                    />
                    )}
                    <item.icon 
                        size={22} 
                        className={cn(
                            "transition-colors duration-200 z-10",
                            isActive ? "text-primary fill-primary/20" : "text-muted-foreground/70 group-hover:text-white"
                        )} 
                    />
                    {isSidebarOpen && (
                    <motion.span 
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="z-10"
                    >
                        {item.label}
                    </motion.span>
                    )}
                </div>
              </Link>
            );
          })}
        </nav>

        {/* Status Module */}
        <div className="p-4 border-t border-white/5 bg-gradient-to-t from-black/80 to-transparent">
             {isSidebarOpen ? (
                 <div className="space-y-4">
                     <div className="flex items-center justify-between text-xs text-muted-foreground">
                         <span className="flex items-center gap-2"><Globe size={12}/> NETWORK</span>
                         <span className="text-green-500 font-mono">12ms</span>
                     </div>
                     <div className="flex items-center justify-between text-xs text-muted-foreground">
                         <span className="flex items-center gap-2"><Cpu size={12}/> AI ENGINE</span>
                         <span className="text-purple-500 font-mono">IDLE</span>
                     </div>
                 </div>
             ) : (
                 <div className="flex flex-col items-center gap-3">
                     <div className="h-2 w-2 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)] animate-pulse" />
                 </div>
             )}
        </div>
      </motion.aside>

      {/* Main Content Area */}
      <main className="flex-1 relative overflow-y-auto overflow-x-hidden bg-[#050505]">
          <div className="p-6 md:p-8 max-w-[1920px] mx-auto min-h-screen">
             <AnimatePresence mode="wait">
                 <motion.div
                    key={location.pathname}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.2 }}
                 >
                     {children}
                 </motion.div>
             </AnimatePresence>
          </div>
      </main>
    </div>
  );
}
