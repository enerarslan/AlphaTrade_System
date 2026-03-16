import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Calendar, TrendingUp, TrendingDown, Minus, Clock, Loader2 } from "lucide-react";
import { api } from "@/lib/api";

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

type MacroObservation = {
  key: string | null;
  series_id: string;
  label: string;
  source: string;
  unit: string;
  observation_date: string;
  value: number | null;
  previous_value: number | null;
  change_pct: number | null;
};

type MacroSummaryResponse = {
  generated_at: string;
  series: unknown[];
  recent_observations: MacroObservation[];
};

const HIGH_IMPACT_SERIES = new Set(["FEDFUNDS", "DGS10", "VIX", "UNRATE", "PAYEMS", "CPIAUCSL", "GDPC1"]);
const MEDIUM_IMPACT_SERIES = new Set(["DGS2", "RSAFS", "DGORDER"]);

const ECB_SERIES = new Set<string>();

function deriveImpact(seriesId: string): "HIGH" | "MEDIUM" | "LOW" {
  if (HIGH_IMPACT_SERIES.has(seriesId)) return "HIGH";
  if (MEDIUM_IMPACT_SERIES.has(seriesId)) return "MEDIUM";
  return "LOW";
}

function deriveCountry(seriesId: string): string {
  if (ECB_SERIES.has(seriesId)) return "🇪🇺";
  return "🇺🇸";
}

function deriveTime(seriesId: string): string {
  if (ECB_SERIES.has(seriesId)) return "07:00";
  return "08:30";
}

function formatValue(value: number | null, unit: string): string | null {
  if (value === null || value === undefined) return null;
  return `${value}${unit}`;
}

function deriveSurprise(changePct: number | null, value: number | null): "BEAT" | "MISS" | "INLINE" | null {
  if (value === null || value === undefined) return null;
  if (changePct === null || changePct === undefined) return "INLINE";
  if (changePct > 0.5) return "BEAT";
  if (changePct < -0.5) return "MISS";
  return "INLINE";
}

function transformObservations(observations: MacroObservation[]): MacroEvent[] {
  return observations.map((obs, i) => ({
    id: obs.series_id ? `${obs.series_id}-${obs.observation_date}` : String(i),
    time: deriveTime(obs.series_id),
    country: deriveCountry(obs.series_id),
    event: obs.label,
    impact: deriveImpact(obs.series_id),
    actual: formatValue(obs.value, obs.unit),
    forecast: "—",
    previous: formatValue(obs.previous_value, obs.unit) ?? "—",
    surprise: deriveSurprise(obs.change_pct, obs.value),
  }));
}

export default function MacroCalendarWidget() {
  const [filter, setFilter] = useState<"ALL" | "HIGH" | "MEDIUM" | "LOW">("ALL");
  const [events, setEvents] = useState<MacroEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchData = async () => {
    try {
      const response = await api.get<MacroSummaryResponse>("/market/macro/summary");
      const transformed = transformObservations(response.data.recent_observations ?? []);
      setEvents(transformed);
      setError(null);
    } catch (err) {
      setError("Failed to load macro data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();

    intervalRef.current = setInterval(() => {
      if (document.visibilityState === "visible") {
        fetchData();
      }
    }, 60_000);

    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const filtered = filter === "ALL" ? events : events.filter(e => e.impact === filter);

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
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 size={20} className="text-cyan-400 animate-spin" />
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-xs text-slate-500">{error}</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-xs text-slate-500">No events available</p>
          </div>
        ) : (
          filtered.map((event, i) => (
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
          ))
        )}
      </div>
    </div>
  );
}
