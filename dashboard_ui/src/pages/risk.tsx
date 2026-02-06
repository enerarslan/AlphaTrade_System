import { useEffect } from "react";
import { useStore } from "@/lib/store";
import { 
  AreaChart, // Using Area for Distribution curve
  Area,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  ReferenceLine
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ShieldAlert, TrendingDown, Activity, AlertTriangle } from "lucide-react";

export default function RiskWarRoomPage() {
  const { fetchRisk, varData } = useStore();

  useEffect(() => {
    fetchRisk();
    const interval = setInterval(fetchRisk, 10000); // 10s refresh for risk
    return () => clearInterval(interval);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      
      {/* Header */}
      <div className="flex items-center justify-between">
         <div>
             <h2 className="text-3xl font-bold tracking-tight text-white glow-text">Risk War Room</h2>
             <p className="text-muted-foreground">Quantitative Risk Management & Stress Testing</p>
         </div>
         <Badge variant="outline" className="border-red-500 text-red-500 px-4 py-1 animate-pulse">
            <ShieldAlert size={14} className="mr-2"/> LIVE MONITORING
         </Badge>
      </div>

      {/* Risk Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {[
          { title: "VaR (95%)", value: varData ? `$${varData.var_95.toLocaleString()}` : "--", desc: "Daily Potential Loss", icon: TrendingDown, color: "text-red-400" },
          { title: "CVaR (Expected Shortfall)", value: varData ? `$${varData.cvar_95.toLocaleString()}` : "--", desc: "Tail Risk Mean", icon: AlertTriangle, color: "text-orange-400" },
          { title: "Portfolio Beta", value: "1.24", desc: "vs S&P 500", icon: Activity, color: "text-blue-400" },
          { title: "Leverage", value: "1.8x", desc: "Max 2.5x", icon: Activity, color: "text-purple-400" },
        ].map((stat, i) => (
          <Card key={i} className="bg-black/40 border-white/10 backdrop-blur-sm">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">{stat.title}</CardTitle>
              <stat.icon className={`h-4 w-4 ${stat.color}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white mb-1">{stat.value}</div>
              <p className="text-xs text-muted-foreground">{stat.desc}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Charts */}
      <div className="grid gap-6 md:grid-cols-2 h-[500px]">
        {/* VaR Distribution */}
        <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
           <CardHeader>
            <CardTitle className="text-white">P&L Distribution (Monte Carlo)</CardTitle>
            <CardDescription>Forward-looking 1-day simulation (10,000 paths)</CardDescription>
          </CardHeader>
           <CardContent className="h-[400px]">
             {varData ? (
                 <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={varData.distribution_curve}>
                         <defs>
                            <linearGradient id="colorProb" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.4}/>
                                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                            </linearGradient>
                         </defs>
                         <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                         <XAxis dataKey="pnl" stroke="#666" fontSize={10} tickFormatter={(val) => `$${Math.round(val/1000)}k`} />
                         <YAxis hide />
                         <Tooltip 
                            contentStyle={{ backgroundColor: 'rgba(0,0,0,0.9)', border: '1px solid rgba(255,255,255,0.2)' }}
                            labelStyle={{ color: '#fff' }}
                         />
                         <ReferenceLine x={-varData.var_95} stroke="red" strokeDasharray="3 3" label={{ value: 'VaR 95%', fill: 'red', fontSize: 10, position: 'insideTopLeft' }} />
                         <Area type="monotone" dataKey="probability" stroke="#8b5cf6" fillOpacity={1} fill="url(#colorProb)" />
                    </AreaChart>
                 </ResponsiveContainer>
             ) : (
                 <div className="flex h-full items-center justify-center text-muted-foreground">Running Simulations...</div>
             )}
           </CardContent>
        </Card>

        {/* Stress Scenarios */}
        <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
           <CardHeader>
            <CardTitle className="text-white">Historical Stress Tests</CardTitle>
            <CardDescription>Portfolio impact during past crises</CardDescription>
          </CardHeader>
           <CardContent className="space-y-6 pt-4">
              {varData && Object.entries(varData.stress_scenarios).map(([scenario, impact]) => (
                  <div key={scenario} className="space-y-2">
                      <div className="flex justify-between items-end">
                          <span className="text-sm font-medium text-gray-300">{scenario}</span>
                          <span className={`font-mono font-bold ${impact < -20 ? 'text-red-500' : 'text-orange-400'}`}>
                              {impact}%
                          </span>
                      </div>
                      <div className="h-2 w-full bg-white/10 rounded-full overflow-hidden">
                          <div 
                            className={`h-full ${impact < -20 ? 'bg-red-500' : 'bg-orange-400'}`} 
                            style={{ width: `${Math.abs(impact) * 2}%` }} // Scale for visual
                          />
                      </div>
                  </div>
              ))}
              {!varData && [1,2,3,4].map(i => (
                  <div key={i} className="h-8 w-full bg-white/5 animate-pulse rounded" />
              ))}
           </CardContent>
        </Card>
      </div>
    </div>
  );
}
