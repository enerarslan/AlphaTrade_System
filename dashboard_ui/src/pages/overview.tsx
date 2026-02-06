import { useEffect } from "react";
import { useStore } from "@/lib/store";
import { 
  Tooltip, 
  ResponsiveContainer,
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  Radar
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Activity, TrendingUp, Zap, Target, Brain, Cpu } from "lucide-react";

export default function OverviewPage() {
  const { fetchAll, explainability, health } = useStore();

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 5000); // 5s refresh
    return () => clearInterval(interval);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Prepare Radar Data
  const radarData = explainability 
    ? Object.entries(explainability.global_importance).map(([feature, value]) => ({
        feature: feature.replace(/_/g, ' '),
        value: value * 100,
        fullMark: 100,
      })) 
    : [];

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      
      {/* Header / Connectivity */}
      <div className="flex items-center justify-between">
          <div>
              <h2 className="text-3xl font-bold tracking-tight text-white glow-text">AI Command Center</h2>
              <p className="text-muted-foreground">Operational Overview & Alpha Signals</p>
          </div>
          <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-3 py-1 bg-white/5 rounded-full border border-white/10">
                  <div className={`h-2 w-2 rounded-full ${health?.status === 'healthy' ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                  <span className="text-xs font-mono text-muted-foreground">{health?.status || 'CONNECTING...'}</span>
              </div>
          </div>
      </div>
      
      {/* KPI Row */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {[
          { title: "Active Model", value: explainability?.model_name || "Loading...", icon: Brain, color: "text-purple-500" },
          { title: "Alpha Confidence", value: "87.4%", change: "+2.1%", icon: Zap, color: "text-yellow-500" },
          { title: "System Latency", value: "12ms", change: "Optimal", icon: Activity, color: "text-green-500" },
          { title: "Risk Regime", value: "Low Vol", change: "Leverage OK", icon: Target, color: "text-blue-500" },
        ].map((stat, i) => (
          <Card key={i} className="bg-black/40 border-white/10 backdrop-blur-sm neon-border">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">{stat.title}</CardTitle>
              <stat.icon className={`h-4 w-4 ${stat.color}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">{stat.value}</div>
              <p className="text-xs text-muted-foreground mt-1">{stat.change}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Grid */}
      <div className="grid gap-6 md:grid-cols-7 lg:grid-cols-7 h-[500px]">
        
        {/* Alpha Radar Chart */}
        <Card className="col-span-4 bg-black/40 border-white/10 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              <Cpu size={18} className="text-purple-500" />
              Feature Importance Radar
            </CardTitle>
            <CardDescription>Real-time influence of market factors on AI decision making</CardDescription>
          </CardHeader>
          <CardContent className="h-[400px]">
            {radarData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                    <PolarGrid stroke="rgba(255,255,255,0.1)" />
                    <PolarAngleAxis dataKey="feature" tick={{ fill: '#888', fontSize: 10 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 50]} tick={false} axisLine={false} />
                    <Radar
                        name="Importance"
                        dataKey="value"
                        stroke="#8b5cf6"
                        strokeWidth={2}
                        fill="#8b5cf6"
                        fillOpacity={0.3}
                    />
                    <Tooltip 
                         contentStyle={{ backgroundColor: 'rgba(0,0,0,0.9)', border: '1px solid rgba(255,255,255,0.2)' }}
                         itemStyle={{ color: '#fff' }}
                    />
                </RadarChart>
                </ResponsiveContainer>
            ) : (
                <div className="h-full flex items-center justify-center text-muted-foreground">
                    Waiting for Model Telemetry...
                </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Shifts (Trend) */}
        <Card className="col-span-3 bg-black/40 border-white/10 backdrop-blur-sm flex flex-col">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-white">
              <TrendingUp size={18} className="text-blue-500" />
              Factor Drift
            </CardTitle>
            <CardDescription>Delta change in feature importance (1h)</CardDescription>
          </CardHeader>
          <CardContent className="flex-1 overflow-auto">
             <div className="space-y-4">
                 {explainability?.recent_shift && Object.entries(explainability.recent_shift).map(([k, v]) => (
                     <div key={k} className="flex items-center justify-between">
                         <span className="text-sm text-gray-400 capitalize">{k.replace(/_/g, ' ')}</span>
                         <div className="flex items-center gap-2">
                             <div className={`h-1.5 w-16 rounded-full bg-white/10 overflow-hidden`}>
                                 <div 
                                    className={`h-full ${v > 0 ? 'bg-green-500' : 'bg-red-500'}`} 
                                    style={{ width: `${Math.min(Math.abs(v) * 500, 100)}%` }} 
                                 />
                             </div>
                             <span className={`text-xs font-mono w-12 text-right ${v > 0 ? 'text-green-500' : 'text-red-500'}`}>
                                 {v > 0 ? '+' : ''}{(v * 100).toFixed(1)}%
                             </span>
                         </div>
                     </div>
                 ))}
                 {!explainability && [1,2,3].map(i => (
                     <div key={i} className="h-8 w-full bg-white/5 animate-pulse rounded" />
                 ))}
             </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
