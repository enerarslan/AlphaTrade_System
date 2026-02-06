import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Activity, Database, Server, Cpu, ShieldCheck, Wifi, RefreshCw, ShieldAlert } from "lucide-react";

interface HealthCheckResult {
  component: string;
  status: "HEALTHY" | "DEGRADED" | "UNHEALTHY" | "UNKNOWN";
  message: string;
  latency_ms: number;
  details: Record<string, any>;
  timestamp: string;
}

interface SystemHealth {
  status: string;
  timestamp: string;
  checks: HealthCheckResult[];
}

export default function SystemStatusPage() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHealth = async () => {
    try {
      const response = await api.get("/health/detailed");
      setHealth(response.data);
      setError(null);
    } catch (err) {
      console.error("Failed to fetch health:", err);
      setError("Failed to connect to backend");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 3000); // Fast polling for "live" feel
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "HEALTHY": return "text-green-500 border-green-500/50 bg-green-500/10";
      case "DEGRADED": return "text-yellow-500 border-yellow-500/50 bg-yellow-500/10";
      case "UNHEALTHY": return "text-red-500 border-red-500/50 bg-red-500/10";
      default: return "text-gray-500 border-gray-500/50 bg-gray-500/10";
    }
  };

  const getIcon = (component: string) => {
    if (component.includes("database")) return Database;
    if (component.includes("redis")) return Server;
    if (component.includes("gpu")) return Cpu;
    if (component.includes("risk")) return ShieldCheck;
    if (component.includes("broker")) return Wifi;
    return Activity;
  };

  if (loading && !health) {
    return (
        <div className="flex items-center justify-center h-[50vh]">
            <RefreshCw className="animate-spin text-primary" size={32} />
        </div>
    );
  }

  if (error && !health) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <Card className="border-red-500/50 bg-red-500/10 backdrop-blur-md">
          <CardHeader>
            <CardTitle className="text-red-500 flex items-center gap-2">
                <ShieldAlert size={20}/> Connection Lost
            </CardTitle>
            <CardDescription className="text-red-200/70">{error}</CardDescription>
          </CardHeader>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="flex items-center justify-between border-b border-white/10 pb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <Activity className="text-primary" />
            System Status
          </h1>
          <p className="text-muted-foreground mt-1">Real-time infrastructure diagnostic matrix</p>
        </div>
        {health && (
           <div className={`px-4 py-1.5 rounded-full border flex items-center gap-2 ${getStatusColor(health.status)}`}>
             <div className="relative flex h-2.5 w-2.5">
               <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 bg-current`}></span>
               <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-current"></span>
             </div>
             <span className="font-mono font-bold tracking-wider">{health.status}</span>
           </div>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {health?.checks.map((check) => {
          const Icon = getIcon(check.component);
          const statusStyle = getStatusColor(check.status);
          
          return (
            <Card key={check.component} className="group bg-white/5 border-white/10 backdrop-blur-sm overflow-hidden hover:bg-white/10 transition-colors">
               <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                 <CardTitle className="text-sm font-medium capitalize text-white flex items-center gap-2">
                    <Icon className="h-4 w-4 text-primary" />
                   {check.component.replace("_", " ")}
                 </CardTitle>
                 <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${statusStyle}`}>
                    {check.status}
                 </span>
               </CardHeader>
               <CardContent>
                 <div className="mt-2 space-y-2">
                    <p className="text-sm text-gray-300 truncate font-medium">
                        {check.message}
                    </p>
                    <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t border-white/5">
                        <span className="font-mono">LATENCY</span>
                        <span className={`font-mono ${check.latency_ms > 100 ? "text-yellow-500" : "text-green-500"}`}>
                            {check.latency_ms.toFixed(2)}ms
                        </span>
                    </div>
                 </div>
                 {/* Detail Overlay on Hover */}
                 {check.details && Object.keys(check.details).length > 0 && (
                   <div className="mt-3 text-[10px] bg-black/40 p-2 rounded border border-white/5 font-mono space-y-1 opacity-60 group-hover:opacity-100 transition-opacity">
                     {Object.entries(check.details).slice(0, 3).map(([k, v]) => (
                       <div key={k} className="flex justify-between">
                         <span className="text-gray-500 uppercase">{k}:</span>
                         <span className="text-gray-300 truncate ml-2 max-w-[150px]">{String(v)}</span>
                       </div>
                     ))}
                   </div>
                 )}
               </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
