import { useCallback, useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Brain,
  Calendar,
  ChevronRight,
  Clock,
  Crosshair,
  Filter,
  Lightbulb,
  MessageSquareText,
  Search,
  ShieldAlert,
  Target,
  Zap,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import MiniGauge from "@/components/ui/mini-gauge";
import AnimatedCounter from "@/components/ui/AnimatedCounter";
import { api } from "@/lib/api";

// ─── Types ───────────────────────────────────────────────────────────────────

type ForensicSignal = {
  signal_id: string;
  timestamp: string;
  symbol: string;
  direction: string;
  strength: number;
  confidence: number;
  model_source: string;
};

type RiskEvent = {
  event_type: string;
  severity: string;
  timestamp: string;
};

type TradeDecision = {
  trade_id: string;
  symbol: string;
  side: string;
  strategy: string;
  entry_time: string;
  exit_time: string | null;
  entry_price: number;
  exit_price: number;
  quantity: number;
  pnl: number;
  pnl_pct: number;
  commission: number;
  hold_duration: string;
  slippage_bps: number;
  signals: ForensicSignal[];
  risk_events: RiskEvent[];
  narrative: string;
  outcome: "win" | "loss" | "flat";
};

type StrategyBreakdown = {
  count: number;
  wins: number;
  losses: number;
  total_pnl: number;
  win_rate: number;
};

type SymbolBreakdown = {
  count: number;
  wins: number;
  total_pnl: number;
};

type HourlyPerf = {
  total: number;
  wins: number;
  total_pnl: number;
};

type ForensicPatterns = {
  insights: string[];
  win_rate: number;
  avg_pnl: number;
  best_strategy: string | null;
  strategy_breakdown: Record<string, StrategyBreakdown>;
  symbol_breakdown: Record<string, SymbolBreakdown>;
  hourly_performance: Record<string, HourlyPerf>;
};

type ForensicsResponse = {
  timestamp: string;
  decisions: TradeDecision[];
  total_count: number;
  patterns: ForensicPatterns;
};

// ─── Animation Variants ──────────────────────────────────────────────────────

const stagger = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.06 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: "easeOut" as const } },
} as const;

const slideIn = {
  hidden: { opacity: 0, x: 20 },
  show: { opacity: 1, x: 0, transition: { duration: 0.3, ease: "easeOut" as const } },
} as const;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatTime(iso: string) {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" }) +
      " " + d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  } catch {
    return iso;
  }
}

function outcomeColor(outcome: string) {
  if (outcome === "win") return "text-emerald-400";
  if (outcome === "loss") return "text-rose-400";
  return "text-slate-400";
}

function outcomeBg(outcome: string) {
  if (outcome === "win") return "border-emerald-500/20 bg-emerald-500/[0.04]";
  if (outcome === "loss") return "border-rose-500/20 bg-rose-500/[0.04]";
  return "border-slate-500/20 bg-slate-500/[0.04]";
}

function pnlColor(pnl: number) {
  if (pnl > 0) return "text-emerald-400";
  if (pnl < 0) return "text-rose-400";
  return "text-slate-400";
}

// ─── Sub-Components ──────────────────────────────────────────────────────────

function PatternInsightsPanel({ patterns }: { patterns: ForensicPatterns }) {
  return (
    <Card className="border-amber-500/15 bg-gradient-to-br from-amber-500/[0.03] to-transparent">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Lightbulb size={18} className="text-amber-400" />
          Pattern Intelligence
        </CardTitle>
        <CardDescription>AI-detected patterns across your trading decisions</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {patterns.insights.length === 0 ? (
          <p className="text-sm text-slate-500">Insufficient data for pattern detection.</p>
        ) : (
          patterns.insights.map((insight, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className="flex items-start gap-3 rounded-lg border border-amber-500/10 bg-amber-500/[0.03] px-4 py-3"
            >
              <Brain size={16} className="mt-0.5 shrink-0 text-amber-400" />
              <p className="text-sm text-slate-200">{insight}</p>
            </motion.div>
          ))
        )}
      </CardContent>
    </Card>
  );
}

function StrategyHeatmap({ breakdown }: { breakdown: Record<string, StrategyBreakdown> }) {
  const entries = Object.entries(breakdown).sort((a, b) => b[1].total_pnl - a[1].total_pnl);
  if (entries.length === 0) return null;
  const maxPnl = Math.max(...entries.map(([, v]) => Math.abs(v.total_pnl)), 1);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Target size={16} className="text-cyan-400" />
          Strategy Breakdown
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {entries.map(([name, stats]) => {
          const pct = Math.abs(stats.total_pnl) / maxPnl;
          const isPositive = stats.total_pnl >= 0;
          return (
            <div key={name} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-slate-200">{name.replaceAll("_", " ")}</span>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-slate-500">{stats.count} trades</span>
                  <span className="text-xs text-slate-500">{(stats.win_rate * 100).toFixed(0)}% WR</span>
                  <span className={`font-mono text-xs font-semibold ${pnlColor(stats.total_pnl)}`}>
                    {stats.total_pnl >= 0 ? "+" : ""}${stats.total_pnl.toLocaleString()}
                  </span>
                </div>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.05]">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${pct * 100}%` }}
                  transition={{ duration: 0.6, ease: "easeOut" }}
                  className={`h-full rounded-full ${isPositive ? "bg-emerald-500/60" : "bg-rose-500/60"}`}
                />
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

function HourlyChart({ hourly }: { hourly: Record<string, HourlyPerf> }) {
  const hours = Array.from({ length: 24 }, (_, i) => i);
  const data = hours.map((h) => hourly[String(h)] || { total: 0, wins: 0, total_pnl: 0 });
  const maxPnl = Math.max(...data.map((d) => Math.abs(d.total_pnl)), 1);
  const maxBarHeight = 56;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Clock size={16} className="text-indigo-400" />
          Hourly P&L Distribution
        </CardTitle>
        <CardDescription>When does your alpha generate the most edge?</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-end gap-0.5">
          {hours.map((h) => {
            const d = data[h];
            const height = d.total > 0 ? Math.max(4, (Math.abs(d.total_pnl) / maxPnl) * maxBarHeight) : 2;
            const isPositive = d.total_pnl >= 0;
            const isMarketHours = h >= 9 && h <= 16;
            return (
              <div key={h} className="group relative flex flex-1 flex-col items-center">
                <div className="relative flex h-14 w-full items-end justify-center">
                  <motion.div
                    initial={{ height: 0 }}
                    animate={{ height }}
                    transition={{ duration: 0.5, delay: h * 0.02 }}
                    className={`w-full max-w-[14px] rounded-t-sm ${
                      d.total === 0
                        ? "bg-white/[0.04]"
                        : isPositive
                          ? "bg-emerald-500/50 group-hover:bg-emerald-400/70"
                          : "bg-rose-500/50 group-hover:bg-rose-400/70"
                    } transition-colors`}
                  />
                </div>
                <span className={`mt-1 text-[8px] ${isMarketHours ? "text-slate-400" : "text-slate-600"}`}>
                  {h}
                </span>
                {d.total > 0 && (
                  <div className="pointer-events-none absolute -top-14 left-1/2 z-20 hidden -translate-x-1/2 rounded-lg border border-white/10 bg-slate-900 px-2.5 py-1.5 shadow-xl group-hover:block">
                    <p className="whitespace-nowrap text-[10px] font-medium text-slate-200">{h}:00 UTC</p>
                    <p className={`whitespace-nowrap text-[10px] font-mono font-semibold ${pnlColor(d.total_pnl)}`}>
                      {d.total_pnl >= 0 ? "+" : ""}${d.total_pnl.toLocaleString()}
                    </p>
                    <p className="whitespace-nowrap text-[9px] text-slate-500">{d.wins}/{d.total} wins</p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
        <div className="mt-1 flex items-center justify-center gap-4 text-[9px] text-slate-600">
          <span>Pre-market</span>
          <span className="text-slate-400">Market Hours (9-16)</span>
          <span>After-hours</span>
        </div>
      </CardContent>
    </Card>
  );
}

function DecisionTimeline({
  decisions,
  selectedId,
  onSelect,
}: {
  decisions: TradeDecision[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="space-y-1">
      {decisions.map((d) => {
        const isSelected = d.trade_id === selectedId;
        return (
          <motion.button
            key={d.trade_id}
            onClick={() => onSelect(d.trade_id)}
            whileHover={{ x: 2 }}
            className={`group flex w-full items-center gap-3 rounded-lg border px-3 py-2.5 text-left transition-all ${
              isSelected
                ? "border-cyan-500/30 bg-cyan-500/[0.08] shadow-[0_0_12px_rgba(6,182,212,0.08)]"
                : `${outcomeBg(d.outcome)} hover:bg-white/[0.04]`
            }`}
          >
            {/* Outcome indicator */}
            <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
              d.outcome === "win" ? "bg-emerald-500/15" : d.outcome === "loss" ? "bg-rose-500/15" : "bg-slate-500/15"
            }`}>
              {d.outcome === "win" ? (
                <ArrowUpRight size={14} className="text-emerald-400" />
              ) : d.outcome === "loss" ? (
                <ArrowDownRight size={14} className="text-rose-400" />
              ) : (
                <Activity size={14} className="text-slate-400" />
              )}
            </div>

            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-slate-100">{d.symbol}</span>
                <Badge variant={d.side === "BUY" ? "success" : "error"} className="text-[9px] px-1.5 py-0">
                  {d.side}
                </Badge>
                {d.risk_events.length > 0 && (
                  <AlertTriangle size={10} className="text-amber-400" />
                )}
              </div>
              <p className="truncate text-[10px] text-slate-500">
                {formatTime(d.entry_time)} · {d.strategy.replaceAll("_", " ")}
              </p>
            </div>

            <div className="shrink-0 text-right">
              <p className={`font-mono text-sm font-bold ${outcomeColor(d.outcome)}`}>
                {d.pnl >= 0 ? "+" : ""}${d.pnl.toLocaleString()}
              </p>
              <p className="text-[10px] text-slate-500">{d.hold_duration}</p>
            </div>

            <ChevronRight
              size={14}
              className={`shrink-0 transition-colors ${isSelected ? "text-cyan-400" : "text-slate-600 group-hover:text-slate-400"}`}
            />
          </motion.button>
        );
      })}
    </div>
  );
}

function DecisionDetail({ decision }: { decision: TradeDecision }) {
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={decision.trade_id}
        variants={slideIn}
        initial="hidden"
        animate="show"
        exit="hidden"
        className="space-y-4"
      >
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h3 className="text-2xl font-bold text-slate-100">{decision.symbol}</h3>
              <Badge variant={decision.side === "BUY" ? "success" : "error"}>{decision.side}</Badge>
              <Badge variant="outline" className="text-[10px]">{decision.strategy.replaceAll("_", " ")}</Badge>
            </div>
            <p className="mt-1 text-sm text-slate-500">
              {formatTime(decision.entry_time)}
              {decision.exit_time && ` → ${formatTime(decision.exit_time)}`}
              {" · "}{decision.hold_duration} hold
            </p>
          </div>
          <div className={`rounded-xl border px-4 py-2 text-right ${outcomeBg(decision.outcome)}`}>
            <p className={`text-xl font-bold font-mono ${outcomeColor(decision.outcome)}`}>
              {decision.pnl >= 0 ? "+" : ""}${decision.pnl.toLocaleString()}
            </p>
            <p className={`text-xs font-mono ${outcomeColor(decision.outcome)}`}>
              {decision.pnl_pct >= 0 ? "+" : ""}{decision.pnl_pct.toFixed(2)}%
            </p>
          </div>
        </div>

        {/* Narrative */}
        <div className="rounded-xl border border-indigo-500/15 bg-indigo-500/[0.03] px-5 py-4">
          <div className="mb-2 flex items-center gap-2">
            <MessageSquareText size={14} className="text-indigo-400" />
            <span className="text-[10px] font-bold uppercase tracking-[0.16em] text-indigo-400/80">AI Narrative</span>
          </div>
          <p className="text-sm leading-relaxed text-slate-300">{decision.narrative}</p>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MetricTile label="Entry Price" value={`$${decision.entry_price.toFixed(2)}`} icon={<ArrowUpRight size={14} />} accent="cyan" />
          <MetricTile label="Exit Price" value={`$${decision.exit_price.toFixed(2)}`} icon={<ArrowDownRight size={14} />} accent="cyan" />
          <MetricTile label="Quantity" value={`${decision.quantity} shares`} icon={<BarChart3 size={14} />} accent="emerald" />
          <MetricTile label="Slippage" value={`${decision.slippage_bps.toFixed(1)} bps`} icon={<Zap size={14} />} accent="amber" />
        </div>

        {/* Signal Context */}
        {decision.signals.length > 0 && (
          <Card className="border-cyan-500/10">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Brain size={14} className="text-cyan-400" />
                Triggering Signals
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {decision.signals.map((sig) => (
                <div key={sig.signal_id} className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2.5">
                  <div className="flex items-center gap-3">
                    <div className={`h-2 w-2 rounded-full ${sig.direction === "LONG" ? "bg-emerald-400" : "bg-rose-400"}`} />
                    <div>
                      <span className="text-sm font-medium text-slate-200">{sig.model_source}</span>
                      <Badge variant="outline" className="ml-2 text-[9px]">{sig.direction}</Badge>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      <p className="text-[10px] uppercase text-slate-500">Confidence</p>
                      <p className="font-mono text-sm font-semibold text-cyan-300">{(sig.confidence * 100).toFixed(0)}%</p>
                    </div>
                    <div className="text-right">
                      <p className="text-[10px] uppercase text-slate-500">Strength</p>
                      <p className="font-mono text-sm font-semibold text-slate-200">{(sig.strength * 100).toFixed(0)}%</p>
                    </div>
                    <MiniGauge value={sig.confidence} size={36} />
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        )}

        {/* Risk Events */}
        {decision.risk_events.length > 0 && (
          <Card className="border-rose-500/10">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <ShieldAlert size={14} className="text-rose-400" />
                Risk Events at Trade Time
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {decision.risk_events.map((evt, i) => (
                <div key={i} className="flex items-center justify-between rounded-lg border border-rose-500/10 bg-rose-500/[0.03] px-3 py-2.5">
                  <div className="flex items-center gap-2">
                    <AlertTriangle size={14} className={evt.severity === "CRITICAL" ? "text-rose-400" : "text-amber-400"} />
                    <span className="text-sm text-slate-200">{evt.event_type.replaceAll("_", " ")}</span>
                  </div>
                  <Badge variant={evt.severity === "CRITICAL" ? "error" : "warning"}>{evt.severity}</Badge>
                </div>
              ))}
            </CardContent>
          </Card>
        )}

        {/* Execution Quality */}
        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3 text-center">
            <p className="text-[10px] uppercase tracking-wider text-slate-500">Commission</p>
            <p className="mt-1 font-mono text-sm font-semibold text-amber-300">${decision.commission.toFixed(2)}</p>
          </div>
          <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3 text-center">
            <p className="text-[10px] uppercase tracking-wider text-slate-500">Trade ID</p>
            <p className="mt-1 font-mono text-xs text-slate-400">{decision.trade_id}</p>
          </div>
          <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3 text-center">
            <p className="text-[10px] uppercase tracking-wider text-slate-500">Net Return</p>
            <p className={`mt-1 font-mono text-sm font-semibold ${outcomeColor(decision.outcome)}`}>
              {decision.pnl_pct >= 0 ? "+" : ""}{decision.pnl_pct.toFixed(2)}%
            </p>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

function MetricTile({
  label,
  value,
  icon,
  accent = "cyan",
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  accent?: "cyan" | "emerald" | "amber" | "rose";
}) {
  const colors = {
    cyan: "border-cyan-500/15 text-cyan-400",
    emerald: "border-emerald-500/15 text-emerald-400",
    amber: "border-amber-500/15 text-amber-400",
    rose: "border-rose-500/15 text-rose-400",
  };
  return (
    <div className={`rounded-xl border bg-white/[0.02] px-3 py-2.5 ${colors[accent].split(" ")[0]}`}>
      <div className="flex items-center gap-1.5">
        <span className={colors[accent].split(" ").slice(1).join(" ")}>{icon}</span>
        <p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
      </div>
      <p className="mt-1 font-mono text-sm font-semibold text-slate-200">{value}</p>
    </div>
  );
}

function SymbolLeaderboard({ breakdown }: { breakdown: Record<string, SymbolBreakdown> }) {
  const entries = Object.entries(breakdown)
    .sort((a, b) => b[1].total_pnl - a[1].total_pnl)
    .slice(0, 8);
  if (entries.length === 0) return null;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Crosshair size={16} className="text-emerald-400" />
          Symbol Leaderboard
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-1.5">
          {entries.map(([sym, stats], i) => (
            <div key={sym} className="flex items-center justify-between rounded-lg border border-white/[0.04] bg-white/[0.01] px-3 py-2">
              <div className="flex items-center gap-3">
                <span className="w-5 text-center text-xs font-bold text-slate-600">#{i + 1}</span>
                <span className="font-semibold text-slate-200">{sym}</span>
                <span className="text-xs text-slate-500">{stats.count} trades</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs text-slate-500">{stats.wins}/{stats.count} W</span>
                <span className={`font-mono text-sm font-bold ${pnlColor(stats.total_pnl)}`}>
                  {stats.total_pnl >= 0 ? "+" : ""}${stats.total_pnl.toLocaleString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────

export default function TradeForensicsPage() {
  const [data, setData] = useState<ForensicsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [filterSymbol, _setFilterSymbol] = useState("");
  const [filterOutcome, setFilterOutcome] = useState<"all" | "win" | "loss">("all");
  const [filterStrategy, setFilterStrategy] = useState<string>("all");
  const [days, setDays] = useState(30);
  const [searchQuery, setSearchQuery] = useState("");

  const fetchForensics = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params: Record<string, string | number> = { days, limit: 200 };
      if (filterSymbol) params.symbol = filterSymbol;
      const res = await api.get<ForensicsResponse>("/forensics/decisions", { params });
      setData(res.data);
      if (res.data.decisions.length > 0 && !selectedId) {
        setSelectedId(res.data.decisions[0].trade_id);
      }
    } catch (err) {
      setError("Failed to load forensic data");
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [days, filterSymbol]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    void fetchForensics();
  }, [fetchForensics]);

  const filteredDecisions = useMemo(() => {
    if (!data) return [];
    let decisions = data.decisions;
    if (filterOutcome !== "all") {
      decisions = decisions.filter((d) => d.outcome === filterOutcome);
    }
    if (filterStrategy !== "all") {
      decisions = decisions.filter((d) => d.strategy === filterStrategy);
    }
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      decisions = decisions.filter(
        (d) =>
          d.symbol.toLowerCase().includes(q) ||
          d.strategy.toLowerCase().includes(q) ||
          d.narrative.toLowerCase().includes(q),
      );
    }
    return decisions;
  }, [data, filterOutcome, filterStrategy, searchQuery]);

  const selectedDecision = useMemo(
    () => filteredDecisions.find((d) => d.trade_id === selectedId) ?? filteredDecisions[0] ?? null,
    [filteredDecisions, selectedId],
  );

  const strategies = useMemo(() => {
    if (!data) return [];
    return [...new Set(data.decisions.map((d) => d.strategy))];
  }, [data]);

  const totalPnl = useMemo(
    () => filteredDecisions.reduce((sum, d) => sum + d.pnl, 0),
    [filteredDecisions],
  );

  const winCount = useMemo(
    () => filteredDecisions.filter((d) => d.outcome === "win").length,
    [filteredDecisions],
  );

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-6">
      {/* Hero Banner */}
      <motion.section variants={fadeUp} className="mission-gradient mission-grid relative overflow-hidden rounded-2xl border border-white/[0.08] p-6 shadow-lg backdrop-blur-sm">
        <div className="relative z-10 flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.24em] text-indigo-400/70">Decision Intelligence</p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-100 sm:text-4xl">
              Trade <span className="gradient-text">Forensics</span>
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-slate-400">
              Time-travel through every trade decision. Understand why each action was taken, what the system saw, and what patterns emerge across your alpha.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-2.5 text-center">
              <p className="text-[10px] uppercase tracking-wider text-slate-500">Decisions</p>
              <p className="font-mono text-lg font-bold text-cyan-300">
                <AnimatedCounter value={filteredDecisions.length} />
              </p>
            </div>
            <div className="rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-2.5 text-center">
              <p className="text-[10px] uppercase tracking-wider text-slate-500">Win Rate</p>
              <p className="font-mono text-lg font-bold text-emerald-300">
                {filteredDecisions.length > 0 ? ((winCount / filteredDecisions.length) * 100).toFixed(0) : 0}%
              </p>
            </div>
            <div className="rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-2.5 text-center">
              <p className="text-[10px] uppercase tracking-wider text-slate-500">Net P&L</p>
              <p className={`font-mono text-lg font-bold ${pnlColor(totalPnl)}`}>
                {totalPnl >= 0 ? "+" : ""}$<AnimatedCounter value={Math.abs(totalPnl)} />
              </p>
            </div>
          </div>
        </div>
      </motion.section>

      {/* Filters Bar */}
      <motion.section variants={fadeUp} className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2 rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 py-1.5">
          <Search size={14} className="text-slate-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search trades..."
            className="w-40 bg-transparent text-sm text-slate-200 outline-none placeholder:text-slate-600"
          />
        </div>

        <div className="flex items-center gap-1.5">
          <Filter size={14} className="text-slate-500" />
          {(["all", "win", "loss"] as const).map((f) => (
            <Button
              key={f}
              variant={filterOutcome === f ? "default" : "outline"}
              size="sm"
              className={`h-7 text-xs ${filterOutcome === f ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/30" : ""}`}
              onClick={() => setFilterOutcome(f)}
            >
              {f === "all" ? "All" : f === "win" ? "Winners" : "Losers"}
            </Button>
          ))}
        </div>

        <select
          value={filterStrategy}
          onChange={(e) => setFilterStrategy(e.target.value)}
          className="rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 py-1.5 text-xs text-slate-300 outline-none"
        >
          <option value="all">All Strategies</option>
          {strategies.map((s) => (
            <option key={s} value={s}>{s.replaceAll("_", " ")}</option>
          ))}
        </select>

        <div className="flex items-center gap-1.5">
          <Calendar size={14} className="text-slate-500" />
          {[7, 14, 30, 90].map((d) => (
            <Button
              key={d}
              variant={days === d ? "default" : "outline"}
              size="sm"
              className={`h-7 text-xs ${days === d ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/30" : ""}`}
              onClick={() => setDays(d)}
            >
              {d}d
            </Button>
          ))}
        </div>

        <Button variant="outline" size="sm" className="h-7 gap-1 text-xs" onClick={() => void fetchForensics()}>
          <Activity size={12} />
          Refresh
        </Button>
      </motion.section>

      {/* Main Content */}
      {loading ? (
        <div className="flex min-h-[40vh] items-center justify-center">
          <div className="flex items-center gap-3 rounded-xl border border-white/[0.08] bg-white/[0.02] px-5 py-3">
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-cyan-500 border-t-transparent" />
            <span className="text-sm text-slate-400">Reconstructing decision history...</span>
          </div>
        </div>
      ) : error ? (
        <Card className="border-rose-500/20">
          <CardContent className="flex items-center gap-3 py-8">
            <AlertTriangle size={18} className="text-rose-400" />
            <span className="text-rose-300">{error}</span>
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Timeline + Detail Split */}
          <motion.section variants={fadeUp} className="grid gap-5 lg:grid-cols-[380px_1fr]">
            {/* Timeline Panel */}
            <div className="max-h-[75vh] space-y-3 overflow-y-auto rounded-xl border border-white/[0.06] bg-white/[0.01] p-3">
              <div className="sticky top-0 z-10 flex items-center justify-between bg-slate-950/90 pb-2 backdrop-blur-sm">
                <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
                  Decision Timeline
                </span>
                <span className="text-[10px] text-slate-600">{filteredDecisions.length} trades</span>
              </div>
              {filteredDecisions.length === 0 ? (
                <p className="py-12 text-center text-sm text-slate-500">No trades match current filters</p>
              ) : (
                <DecisionTimeline
                  decisions={filteredDecisions}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                />
              )}
            </div>

            {/* Detail Panel */}
            <div className="min-h-[50vh] rounded-xl border border-white/[0.06] bg-white/[0.01] p-5">
              {selectedDecision ? (
                <DecisionDetail decision={selectedDecision} />
              ) : (
                <div className="flex h-full items-center justify-center">
                  <p className="text-sm text-slate-500">Select a trade to view forensic details</p>
                </div>
              )}
            </div>
          </motion.section>

          {/* Pattern Intelligence + Strategy + Hourly */}
          {data?.patterns && (
            <>
              <motion.section variants={fadeUp}>
                <PatternInsightsPanel patterns={data.patterns} />
              </motion.section>

              <motion.section variants={fadeUp} className="grid gap-5 lg:grid-cols-2">
                <StrategyHeatmap breakdown={data.patterns.strategy_breakdown} />
                <SymbolLeaderboard breakdown={data.patterns.symbol_breakdown} />
              </motion.section>

              <motion.section variants={fadeUp}>
                <HourlyChart hourly={data.patterns.hourly_performance} />
              </motion.section>
            </>
          )}
        </>
      )}
    </motion.div>
  );
}
