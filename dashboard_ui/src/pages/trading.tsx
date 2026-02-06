import { useEffect } from "react";
import { useStore } from "@/lib/store";
import { 
  BarChart,
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { DollarSign, LineChart, Timer, Percent } from "lucide-react";

export default function TradingTerminalPage() {
  const { fetchTCA, tca } = useStore();

  useEffect(() => {
    fetchTCA();
    const interval = setInterval(fetchTCA, 5000);
    return () => clearInterval(interval);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const venueData = tca 
    ? Object.entries(tca.venue_breakdown).map(([venue, pct]) => ({ 
        name: venue, 
        value: pct * 100 
      })).sort((a,b) => b.value - a.value)
    : [];

  const COLORS = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b'];

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      
      {/* Header */}
      <div>
         <h2 className="text-3xl font-bold tracking-tight text-white glow-text">Trading Terminal Pro</h2>
         <p className="text-muted-foreground">Execution Management & TCA</p>
      </div>
      
      {/* Execution Stats */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {[
          { title: "Slippage (bps)", value: tca?.slippage_bps || "--", unit: "bps", icon: LineChart, color: "text-red-400" },
          { title: "Market Impact", value: tca?.market_impact_bps || "--", unit: "bps", icon: DollarSign, color: "text-orange-400" },
          { title: "Fill Probability", value: tca ? `${(tca.fill_probability * 100).toFixed(0)}%` : "--", unit: "", icon: Percent, color: "text-green-400" },
          { title: "Avg Speed", value: tca?.execution_speed_ms || "--", unit: "ms", icon: Timer, color: "text-blue-400" },
        ].map((stat, i) => (
          <Card key={i} className="bg-black/40 border-white/10 backdrop-blur-sm">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">{stat.title}</CardTitle>
              <stat.icon className={`h-4 w-4 ${stat.color}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">
                {stat.value} <span className="text-xs text-muted-foreground font-normal">{stat.unit}</span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Split */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 h-[500px]">
        {/* Venue Breakdown */}
        <Card className="col-span-1 bg-black/40 border-white/10 backdrop-blur-sm">
           <CardHeader>
            <CardTitle className="text-white">Smart Order Router</CardTitle>
            <CardDescription>Execution Venue Breakdown</CardDescription>
          </CardHeader>
           <CardContent className="h-[400px]">
             {venueData.length > 0 ? (
                 <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={venueData} layout="vertical">
                         <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="rgba(255,255,255,0.1)" />
                         <XAxis type="number" hide />
                         <YAxis dataKey="name" type="category" width={60} tick={{fill: 'white', fontSize: 12}} />
                         <Tooltip 
                            cursor={{fill: 'rgba(255,255,255,0.05)'}}
                            contentStyle={{ backgroundColor: 'rgba(0,0,0,0.9)', border: '1px solid rgba(255,255,255,0.2)' }}
                            itemStyle={{ color: '#fff' }}
                         />
                         <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={32}>
                             {venueData.map((_entry, index) => (
                                 <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                             ))}
                         </Bar>
                    </BarChart>
                 </ResponsiveContainer>
             ) : (
                 <div className="flex h-full items-center justify-center text-muted-foreground">Gathering TCA Data...</div>
             )}
           </CardContent>
        </Card>

        {/* Live Blotter (Mocktable for now) */}
        <Card className="col-span-2 bg-black/40 border-white/10 backdrop-blur-sm">
           <CardHeader>
            <CardTitle className="text-white">Active Blotter</CardTitle>
            <CardDescription>Live Orders & Fills</CardDescription>
          </CardHeader>
          <CardContent>
               <div className="rounded-md border border-white/10 overflow-hidden">
                   <table className="w-full text-sm text-left text-gray-400">
                       <thead className="text-xs uppercase bg-white/5 text-gray-400">
                           <tr>
                               <th className="px-6 py-3">Time</th>
                               <th className="px-6 py-3">Symbol</th>
                               <th className="px-6 py-3">Side</th>
                               <th className="px-6 py-3">Size</th>
                               <th className="px-6 py-3">Price</th>
                               <th className="px-6 py-3">Status</th>
                           </tr>
                       </thead>
                       <tbody>
                           {[1,2,3,4,5].map((i) => (
                               <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                                   <td className="px-6 py-4 font-mono">10:42:{10+i}</td>
                                   <td className="px-6 py-4 font-bold text-white">AAPL</td>
                                   <td className="px-6 py-4 text-green-400">BUY</td>
                                   <td className="px-6 py-4 font-mono">100</td>
                                   <td className="px-6 py-4 font-mono">154.2{i}</td>
                                   <td className="px-6 py-4"><Badge variant="outline" className="border-green-500 text-green-500">FILLED</Badge></td>
                               </tr>
                           ))}
                       </tbody>
                   </table>
               </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
