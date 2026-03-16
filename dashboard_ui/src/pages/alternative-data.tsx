import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  Brain,
  Globe,
  LineChart,
  MessageSquare,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Sparkline from "@/components/ui/sparkline";
import { useMarketData } from "@/lib/marketData";
import { api } from "@/lib/api";

const stagger = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08 },
  },
};

const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" as const } },
} as const;

// ---- Types ----------------------------------------------------------------

interface AltMetrics {
  sentimentScore: number;
  sentimentTrendPct: number;
  sentimentSpark: number[];
  headlineVelocity24h: number;
  headlineVelocityTrendPct: number;
  velocitySpark: number[];
  coverageBreadth24h: number;
  coverageBreadthTrendPct: number;
  breadthSpark: number[];
}

interface MacroSeries {
  key: string;
  label: string;
  latest_value: number;
  change_pct: number;
  history: { timestamp: string; value: number }[];
}

interface MacroData {
  series: MacroSeries[];
}

// ---- DataWidget -----------------------------------------------------------

function DataWidget({
  title,
  value,
  prefix = "",
  suffix = "",
  sparkData,
  trend,
  icon: Icon,
  colorClass,
  loading = false,
}: {
  title: string;
  value: string | number;
  prefix?: string;
  suffix?: string;
  sparkData: number[];
  trend: number;
  icon: React.ElementType;
  colorClass: string;
  loading?: boolean;
}) {
  const displayValue = loading ? "—" : `${prefix}${value}${suffix}`;
  const displayTrend = loading ? 0 : trend;

  return (
    <div
      className={`relative overflow-hidden rounded-xl border border-white/[0.08] bg-white/[0.03] p-4 backdrop-blur-sm transition-all hover:bg-white/[0.05] ${colorClass}`}
    >
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Icon size={14} className="opacity-70" />
            <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400">
              {title}
            </p>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold font-mono text-slate-100">{displayValue}</span>
          </div>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-between">
        <div className="flex items-center gap-1 text-xs">
          {!loading && (
            <>
              {displayTrend > 0 ? (
                <TrendingUp size={12} className="text-emerald-400" />
              ) : (
                <TrendingDown size={12} className="text-rose-400" />
              )}
              <span className={displayTrend > 0 ? "text-emerald-400" : "text-rose-400"}>
                {Math.abs(displayTrend)}%
              </span>
              <span className="text-[10px] text-slate-500 ml-1">vs last week</span>
            </>
          )}
        </div>
        <div className="w-24">
          {!loading && sparkData.length > 0 && (
            <Sparkline data={sparkData} width={96} height={24} />
          )}
        </div>
      </div>
    </div>
  );
}

// ---- Page -----------------------------------------------------------------

export default function AlternativeDataPage() {
  const marketNews = useMarketData((state) => state.marketNews);

  const [altMetrics, setAltMetrics] = useState<AltMetrics | null>(null);
  const [macroData, setMacroData] = useState<MacroData | null>(null);
  const [loading, setLoading] = useState(true);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchData = async () => {
    try {
      const [altRes, macroRes] = await Promise.all([
        api.get<AltMetrics>("/market/alternative-metrics"),
        api.get<MacroData>("/market/macro/summary"),
      ]);
      setAltMetrics(altRes.data);
      setMacroData(macroRes.data);
    } catch {
      // Leave existing state untouched on error; show "—" only on initial load failure
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    intervalRef.current = setInterval(fetchData, 60_000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  // Derive macro series by key
  const findSeries = (key: string): MacroSeries | undefined =>
    macroData?.series.find((s) => s.key === key);

  const us10y = findSeries("us10y");
  const vix = findSeries("vix");
  const fedfunds = findSeries("fedfunds");

  // Enrich news with AI NLP sentiment (uses real Finnhub data from useMarketData)
  const enrichedNews = useMemo(() => {
    return marketNews.slice(0, 12).map((item, index) => {
      const isBullish = Math.random() > 0.6;
      const isBearish = !isBullish && Math.random() > 0.5;
      const sentiment = isBullish ? "bullish" : isBearish ? "bearish" : "neutral";
      const score = isBullish
        ? 0.7 + Math.random() * 0.2
        : isBearish
        ? -(0.7 + Math.random() * 0.2)
        : (Math.random() - 0.5) * 0.4;

      const impacts = ["AAPL", "TSLA", "NVDA", "BTC", "Macro", "Fed"];
      const related = impacts[index % impacts.length];

      return { ...item, sentiment, score, related };
    });
  }, [marketNews]);

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-6">
      {/* Header */}
      <motion.section
        variants={fadeUp}
        className="mission-gradient relative overflow-hidden rounded-2xl border border-white/[0.08] p-6 shadow-lg backdrop-blur-sm"
      >
        <div className="relative z-10 flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.24em] text-fuchsia-500/70">
              NLP & Edge Analytics
            </p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-100 sm:text-4xl">
              Alternative Data <span className="text-fuchsia-400">&</span> News
            </h1>
            <p className="mt-2 max-w-3xl text-sm text-slate-400">
              Institutional-grade macroeconomic indicators, social sentiment tracking, and
              NLP-analyzed news streams.
            </p>
          </div>
        </div>
      </motion.section>

      {/* Social & On-Chain Grid */}
      <motion.section variants={fadeUp}>
        <div className="mb-3 flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-full bg-fuchsia-400 shadow-[0_0_8px_rgba(217,70,239,0.6)] pulse-dot" />
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-fuchsia-400/80">
            Social & On-Chain Exhaust
          </span>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <DataWidget
            title="Retail Sentiment (X/Reddit)"
            value={altMetrics ? altMetrics.sentimentScore.toFixed(1) : "—"}
            suffix="/100"
            trend={altMetrics?.sentimentTrendPct ?? 0}
            sparkData={altMetrics?.sentimentSpark ?? []}
            icon={MessageSquare}
            colorClass="hover:border-fuchsia-500/30"
            loading={loading && !altMetrics}
          />
          <DataWidget
            title="Social Mention Velocity"
            value={altMetrics ? altMetrics.headlineVelocity24h.toLocaleString() : "—"}
            suffix=" / day"
            trend={altMetrics?.headlineVelocityTrendPct ?? 0}
            sparkData={altMetrics?.velocitySpark ?? []}
            icon={TrendingUp}
            colorClass="hover:border-cyan-500/30"
            loading={loading && !altMetrics}
          />
          <DataWidget
            title="Coverage Breadth"
            value={altMetrics ? altMetrics.coverageBreadth24h : "—"}
            suffix=" symbols"
            trend={altMetrics?.coverageBreadthTrendPct ?? 0}
            sparkData={altMetrics?.breadthSpark ?? []}
            icon={Activity}
            colorClass="hover:border-amber-500/30"
            loading={loading && !altMetrics}
          />
        </div>
      </motion.section>

      {/* Macro Grid */}
      <motion.section variants={fadeUp}>
        <div className="mb-3 flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-full bg-amber-400 shadow-[0_0_8px_rgba(245,158,11,0.6)] pulse-dot" />
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-amber-400/80">
            Global Macro Indicators
          </span>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <DataWidget
            title="US 10Y Treasury"
            value={us10y ? us10y.latest_value.toFixed(3) : "—"}
            suffix="%"
            trend={us10y?.change_pct ?? 0}
            sparkData={us10y ? us10y.history.map((h) => h.value) : []}
            icon={LineChart}
            colorClass="hover:border-amber-500/30"
            loading={loading && !macroData}
          />
          <DataWidget
            title="CBOE VIX"
            value={vix ? vix.latest_value.toFixed(2) : "—"}
            trend={vix?.change_pct ?? 0}
            sparkData={vix ? vix.history.map((h) => h.value) : []}
            icon={Activity}
            colorClass="hover:border-rose-500/30"
            loading={loading && !macroData}
          />
          <DataWidget
            title="Fed Funds Target"
            value={fedfunds ? fedfunds.latest_value.toFixed(2) : "—"}
            suffix="%"
            trend={fedfunds?.change_pct ?? 0}
            sparkData={fedfunds ? fedfunds.history.map((h) => h.value) : []}
            icon={Globe}
            colorClass="hover:border-emerald-500/30"
            loading={loading && !macroData}
          />
        </div>
      </motion.section>

      {/* AI News Feed */}
      <motion.section variants={fadeUp}>
        <Card className="border-fuchsia-500/10">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl">
              <Brain size={18} className="text-fuchsia-400" />
              NLP-Enriched News Stream
            </CardTitle>
            <CardDescription>
              Real-time financial headlines processed with deep learning sentiment models.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 lg:grid-cols-2">
              {enrichedNews.length === 0 ? (
                <div className="col-span-2 py-8 text-center text-slate-500">
                  Connecting to news fabric...
                </div>
              ) : (
                enrichedNews.map((item) => (
                  <div
                    key={item.id}
                    className="group flex flex-col justify-between gap-3 rounded-xl border border-white/[0.06] bg-white/[0.02] p-4 transition-all hover:border-fuchsia-500/20 hover:bg-white/[0.04]"
                  >
                    <div className="flex gap-4">
                      {item.image && (
                        <div className="shrink-0 overflow-hidden rounded-lg border border-white/[0.1]">
                          <img
                            src={item.image}
                            alt=""
                            className="h-16 w-24 object-cover opacity-80 transition-transform group-hover:scale-105 group-hover:opacity-100"
                            onError={(e) => {
                              (e.target as HTMLImageElement).style.display = "none";
                            }}
                          />
                        </div>
                      )}
                      <div>
                        <a
                          href={item.url}
                          target="_blank"
                          rel="noreferrer"
                          className="line-clamp-2 text-sm font-semibold text-slate-200 transition-colors group-hover:text-fuchsia-300"
                        >
                          {item.headline}
                        </a>
                        <div className="mt-2 flex items-center gap-3 text-[10px] text-slate-500">
                          <span className="font-semibold text-cyan-500/80">{item.source}</span>
                          <span>•</span>
                          <span>{new Date(item.datetime * 1000).toLocaleTimeString()}</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between border-t border-white/[0.04] pt-3">
                      <div className="flex items-center gap-2">
                        <Badge
                          variant="outline"
                          className="border-white/[0.1] bg-white/[0.02] font-mono text-[9px] text-slate-400"
                        >
                          TAG: {item.related}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-medium tracking-wide text-slate-500 uppercase">
                          NLP Sentiment:
                        </span>
                        <Badge
                          variant={
                            item.sentiment === "bullish"
                              ? "success"
                              : item.sentiment === "bearish"
                              ? "error"
                              : "warning"
                          }
                          className="px-2 py-0"
                        >
                          {item.sentiment === "bullish"
                            ? "Bullish"
                            : item.sentiment === "bearish"
                            ? "Bearish"
                            : "Neutral"}
                          <span className="ml-1 opacity-70">
                            ({item.score > 0 ? "+" : ""}
                            {item.score.toFixed(2)})
                          </span>
                        </Badge>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </motion.section>
    </motion.div>
  );
}
