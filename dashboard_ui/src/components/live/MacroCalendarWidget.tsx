import { useState } from "react";
import { motion } from "framer-motion";
import { Calendar, TrendingUp, TrendingDown, Minus, Clock } from "lucide-react";

type MacroEvent = {
  id: string;
  time: string;
  country: string;
  event: string;
  impact: "HIGH" | "MEDIUM" | "LOW";
  actual: string | null;
  forecast: string;
  previous: string;
  surprise: "BEAT" | "MISS" | "INLINE" | null;
};

const MOCK_EVENTS: MacroEvent[] = [
  { id: "1", time: "08:30", country: "🇺🇸", event: "Nonfarm Payrolls", impact: "HIGH", actual: "275K", forecast: "200K", previous: "229K", surprise: "BEAT" },
  { id: "2", time: "08:30", country: "🇺🇸", event: "Unemployment Rate", impact: "HIGH", actual: "3.9%", forecast: "3.7%", previous: "3.7%", surprise: "MISS" },
  { id: "3", time: "10:00", country: "🇺🇸", event: "ISM Manufacturing PMI", impact: "HIGH", actual: "47.8", forecast: "49.5", previous: "49.1", surprise: "MISS" },
  { id: "4", time: "10:00", country: "🇺🇸", event: "Michigan Consumer Sentiment", impact: "MEDIUM", actual: null, forecast: "76.9", previous: "79.4", surprise: null },
  { id: "5", time: "11:00", country: "🇪🇺", event: "ECB Deposit Rate Decision", impact: "HIGH", actual: "4.00%", forecast: "4.00%", previous: "4.00%", surprise: "INLINE" },
  { id: "6", time: "13:00", country: "🇺🇸", event: "Fed Chair Speech", impact: "HIGH", actual: null, forecast: "—", previous: "—", surprise: null },
  { id: "7", time: "08:00", country: "🇬🇧", event: "UK GDP (QoQ)", impact: "MEDIUM", actual: "-0.3%", forecast: "-0.1%", previous: "0.0%", surprise: "MISS" },
  { id: "8", time: "14:00", country: "🇺🇸", event: "10-Year Note Auction", impact: "MEDIUM", actual: null, forecast: "4.28%", previous: "4.32%", surprise: null },
  { id: "9", time: "07:00", country: "🇩🇪", event: "German CPI (YoY)", impact: "MEDIUM", actual: "2.5%", forecast: "2.6%", previous: "2.9%", surprise: "BEAT" },
  { id: "10", time: "15:00", country: "🇺🇸", event: "Baker Hughes Rig Count", impact: "LOW", actual: null, forecast: "—", previous: "623", surprise: null },
];

export default function MacroCalendarWidget() {
  const [filter, setFilter] = useState<"ALL" | "HIGH" | "MEDIUM" | "LOW">("ALL");

  const filtered = filter === "ALL" ? MOCK_EVENTS : MOCK_EVENTS.filter(e => e.impact === filter);

  const impactStyle = (impact: string) => {
    if (impact === "HIGH") return "bg-rose-500/20 text-rose-300 border-rose-500/30";
    if (impact === "MEDIUM") return "bg-amber-500/20 text-amber-300 border-amber-500/30";
    return "bg-slate-500/20 text-slate-400 border-slate-500/30";
  };

  const surpriseIcon = (s: string | null) => {
    if (s === "BEAT") return <TrendingUp size={12} className="text-emerald-400" />;
    if (s === "MISS") return <TrendingDown size={12} className="text-rose-400" />;
    if (s === "INLINE") return <Minus size={12} className="text-slate-400" />;
    return <Clock size={12} className="text-slate-600 animate-pulse" />;
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header with Filters */}
      <div className="flex items-center justify-between p-3 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <Calendar size={14} className="text-cyan-400" />
          <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">Economic Calendar</span>
        </div>
        <div className="flex gap-1">
          {(["ALL", "HIGH", "MEDIUM", "LOW"] as const).map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-2 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider transition-all ${
                filter === f
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                  : "text-slate-500 hover:text-slate-300 border border-transparent"
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Events List */}
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        {filtered.map((event, i) => (
          <motion.div
            key={event.id}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.03 }}
            className={`flex items-center gap-3 px-3 py-2.5 border-b border-white/[0.04] hover:bg-white/[0.03] transition-colors ${
              !event.actual ? "opacity-70" : ""
            }`}
          >
            {/* Time + Country */}
            <div className="w-16 shrink-0">
              <p className="text-[10px] font-mono text-slate-500">{event.time}</p>
              <p className="text-sm">{event.country}</p>
            </div>

            {/* Impact Indicator */}
            <div className="shrink-0">
              <span className={`inline-block w-14 text-center rounded px-1.5 py-0.5 text-[8px] font-bold border uppercase ${impactStyle(event.impact)}`}>
                {event.impact}
              </span>
            </div>

            {/* Event Name */}
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-slate-200 truncate">{event.event}</p>
            </div>

            {/* Data Columns */}
            <div className="flex items-center gap-3 shrink-0 font-mono text-[11px]">
              <div className="text-center w-14">
                <p className="text-[8px] text-slate-600 uppercase">Actual</p>
                <p className={`font-bold ${
                  event.actual 
                    ? event.surprise === "BEAT" 
                      ? "text-emerald-400" 
                      : event.surprise === "MISS" 
                        ? "text-rose-400" 
                        : "text-slate-300"
                    : "text-slate-600"
                }`}>
                  {event.actual ?? "—"}
                </p>
              </div>
              <div className="text-center w-14">
                <p className="text-[8px] text-slate-600 uppercase">Forecast</p>
                <p className="text-slate-400">{event.forecast}</p>
              </div>
              <div className="text-center w-14">
                <p className="text-[8px] text-slate-600 uppercase">Previous</p>
                <p className="text-slate-500">{event.previous}</p>
              </div>
              <div className="w-5 flex items-center justify-center">
                {surpriseIcon(event.surprise)}
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
