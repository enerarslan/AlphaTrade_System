import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  ChevronDown,
  Crosshair,
  Maximize2,
  Minimize2,
  RotateCcw,
  Shield,
  Target,
  TrendingDown,
  X,
  Zap,
} from "lucide-react";
import { api } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface FactorExposures {
  equity_beta: number;
  rate_sensitivity: number;
  vix_sensitivity: number;
  credit_sensitivity: number;
  liquidity_score: number;
  usd_sensitivity: number;
}

interface StressPosition {
  symbol: string;
  quantity: number;
  avg_entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  weight: number;
  sector: string;
  volatility_30d: number;
  factor_exposures: FactorExposures;
}

interface SnapshotResponse {
  timestamp: string;
  portfolio_value: number;
  positions: StressPosition[];
}

interface ScenarioParams {
  vix_shock_pct: number;
  rate_move_bps: number;
  equity_shock_pct: number;
  credit_spread_bps: number;
  liquidity_drain_pct: number;
  usd_move_pct: number;
}

interface LineageEvent {
  timestamp: string;
  stage: string;
  detail: string;
  status: "success" | "warning" | "error";
}

interface LineageResponse {
  symbol: string;
  events: LineageEvent[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_SCENARIO: ScenarioParams = {
  vix_shock_pct: 0,
  rate_move_bps: 0,
  equity_shock_pct: 0,
  credit_spread_bps: 0,
  liquidity_drain_pct: 0,
  usd_move_pct: 0,
};

const PRESETS: Record<string, ScenarioParams> = {
  Custom: { ...DEFAULT_SCENARIO },
  "Black Monday '87": { vix_shock_pct: 150, rate_move_bps: -50, equity_shock_pct: -22, credit_spread_bps: 200, liquidity_drain_pct: -60, usd_move_pct: -5 },
  "GFC 2008": { vix_shock_pct: 180, rate_move_bps: -100, equity_shock_pct: -25, credit_spread_bps: 400, liquidity_drain_pct: -70, usd_move_pct: 8 },
  "COVID Mar 2020": { vix_shock_pct: 200, rate_move_bps: -150, equity_shock_pct: -30, credit_spread_bps: 350, liquidity_drain_pct: -50, usd_move_pct: 5 },
  "Rate Shock": { vix_shock_pct: 40, rate_move_bps: 250, equity_shock_pct: -8, credit_spread_bps: 100, liquidity_drain_pct: -20, usd_move_pct: 3 },
  "Flash Crash": { vix_shock_pct: 80, rate_move_bps: 0, equity_shock_pct: -12, credit_spread_bps: 50, liquidity_drain_pct: -80, usd_move_pct: 0 },
  "Stagflation": { vix_shock_pct: 60, rate_move_bps: 200, equity_shock_pct: -15, credit_spread_bps: 250, liquidity_drain_pct: -35, usd_move_pct: -4 },
};

const SECTOR_LABELS: Record<string, string> = {
  technology: "TECH",
  financial: "FIN",
  healthcare: "HLTH",
  consumer: "CONS",
  energy: "ENER",
  industrial: "IND",
  communication: "COMM",
  etf: "ETF",
  other: "OTH",
};

const SECTOR_ORDER = ["technology", "financial", "healthcare", "consumer", "energy", "industrial", "communication", "etf", "other"];

const SECTOR_COLORS: Record<string, string> = {
  technology: "#06b6d4",
  financial: "#8b5cf6",
  healthcare: "#10b981",
  consumer: "#f59e0b",
  energy: "#ef4444",
  industrial: "#f97316",
  communication: "#ec4899",
  etf: "#94a3b8",
  other: "#64748b",
};

// ---------------------------------------------------------------------------
// Stress Computation (client-side for instant response)
// ---------------------------------------------------------------------------

function computeStressedPnl(pos: StressPosition, scenario: ScenarioParams): number {
  const fe = pos.factor_exposures;
  const returnShock =
    fe.equity_beta * (scenario.equity_shock_pct / 100) +
    fe.rate_sensitivity * (scenario.rate_move_bps / 10000) +
    fe.vix_sensitivity * (scenario.vix_shock_pct / 100) +
    fe.credit_sensitivity * (scenario.credit_spread_bps / 10000) +
    fe.usd_sensitivity * (scenario.usd_move_pct / 100);

  const liquidityCost =
    (1 - fe.liquidity_score) * Math.abs(scenario.liquidity_drain_pct / 100) * 0.5;

  return pos.market_value * (returnShock - liquidityCost);
}

function isScenarioActive(scenario: ScenarioParams): boolean {
  return Object.values(scenario).some((v) => Math.abs(v) > 0.01);
}

// ---------------------------------------------------------------------------
// Color Helpers
// ---------------------------------------------------------------------------

function stressColor(pnl: number, maxAbsPnl: number): string {
  if (maxAbsPnl < 0.01) return "rgba(100, 116, 139, 0.5)";
  const t = Math.max(-1, Math.min(1, pnl / maxAbsPnl));
  if (t >= 0) {
    const i = Math.min(t * 1.5, 1);
    return `rgba(16, 185, 129, ${(0.25 + i * 0.55).toFixed(2)})`;
  }
  const i = Math.min(Math.abs(t) * 1.5, 1);
  return `rgba(244, 63, 94, ${(0.25 + i * 0.55).toFixed(2)})`;
}

function stressBorderColor(pnl: number, maxAbsPnl: number): string {
  if (maxAbsPnl < 0.01) return "rgba(100, 116, 139, 0.3)";
  const t = Math.max(-1, Math.min(1, pnl / maxAbsPnl));
  if (t >= 0) return `rgba(16, 185, 129, ${(0.4 + Math.min(t, 1) * 0.5).toFixed(2)})`;
  return `rgba(244, 63, 94, ${(0.4 + Math.min(Math.abs(t), 1) * 0.5).toFixed(2)})`;
}

// ---------------------------------------------------------------------------
// Narrative Generation (client-side for zero latency)
// ---------------------------------------------------------------------------

function generateNarrative(
  positions: StressPosition[],
  scenario: ScenarioParams,
  portfolioValue: number,
): string {
  if (!isScenarioActive(scenario) || positions.length === 0) {
    return "Drag the scenario sliders to simulate market shocks across your portfolio positions.";
  }

  const stressedEntries = positions.map((pos) => ({
    symbol: pos.symbol,
    sector: pos.sector,
    pnl: computeStressedPnl(pos, scenario),
    weight: pos.weight,
  }));

  const totalImpact = stressedEntries.reduce((s, e) => s + e.pnl, 0);
  const sorted = [...stressedEntries].sort((a, b) => a.pnl - b.pnl);
  const worst = sorted[0];
  const best = sorted[sorted.length - 1];

  const totalNegative = sorted.filter((e) => e.pnl < 0).reduce((s, e) => s + Math.abs(e.pnl), 0);
  const top3Loss = sorted
    .slice(0, 3)
    .reduce((s, e) => s + Math.abs(Math.min(e.pnl, 0)), 0);
  const concentration = totalNegative > 0 ? (top3Loss / totalNegative) * 100 : 0;

  const impactPct = portfolioValue > 0 ? (totalImpact / portfolioValue) * 100 : 0;

  const shocks = [
    { name: "VIX spike", val: Math.abs(scenario.vix_shock_pct), unit: "%" },
    { name: "rate move", val: Math.abs(scenario.rate_move_bps), unit: "bp" },
    { name: "equity drawdown", val: Math.abs(scenario.equity_shock_pct), unit: "%" },
    { name: "credit widening", val: Math.abs(scenario.credit_spread_bps), unit: "bp" },
    { name: "liquidity drain", val: Math.abs(scenario.liquidity_drain_pct), unit: "%" },
    { name: "USD move", val: Math.abs(scenario.usd_move_pct), unit: "%" },
  ]
    .filter((s) => s.val > 0)
    .sort((a, b) => b.val - a.val);

  const dominant = shocks[0];
  const sign = totalImpact >= 0 ? "+" : "";
  const sectorLosses = new Map<string, number>();
  for (const e of stressedEntries) {
    sectorLosses.set(e.sector, (sectorLosses.get(e.sector) ?? 0) + e.pnl);
  }
  const worstSector = [...sectorLosses.entries()].sort((a, b) => a[1] - b[1])[0];

  let narrative = `A ${dominant.val}${dominant.unit} ${dominant.name} scenario yields portfolio impact of ${sign}$${Math.abs(Math.round(totalImpact)).toLocaleString()} (${impactPct >= 0 ? "+" : ""}${impactPct.toFixed(2)}%).`;

  if (worst && worst.pnl < 0) {
    narrative += ` Worst hit: ${worst.symbol} ($${Math.round(worst.pnl).toLocaleString()}).`;
  }

  if (concentration > 50) {
    narrative += ` Top-3 losers concentrate ${concentration.toFixed(0)}% of total drawdown`;
    if (worstSector) {
      narrative += ` — heaviest sector exposure: ${SECTOR_LABELS[worstSector[0]] ?? worstSector[0]}.`;
    } else {
      narrative += ".";
    }
  }

  if (best && best.pnl > 0) {
    narrative += ` Partial hedge via ${best.symbol} (+$${Math.round(best.pnl).toLocaleString()}).`;
  }

  return narrative;
}

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

function fmtDollar(v: number): string {
  const abs = Math.abs(v);
  if (abs >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `$${(v / 1e3).toFixed(1)}K`;
  return `$${v.toFixed(0)}`;
}

function fmtPct(v: number): string {
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

// ---------------------------------------------------------------------------
// Slider Component
// ---------------------------------------------------------------------------

function ScenarioSlider({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
  colorClass,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (v: number) => void;
  colorClass: string;
}) {
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div className="group">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 group-hover:text-slate-300 transition-colors">
          {label}
        </span>
        <span className={`text-xs font-mono font-bold ${Math.abs(value) > 0.01 ? colorClass : "text-slate-500"}`}>
          {value > 0 ? "+" : ""}
          {value}
          {unit}
        </span>
      </div>
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-white/[0.08]
            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5
            [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-slate-200 [&::-webkit-slider-thumb]:shadow-[0_0_8px_rgba(255,255,255,0.3)]
            [&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:hover:scale-125
            [&::-moz-range-thumb]:w-3.5 [&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:rounded-full
            [&::-moz-range-thumb]:bg-slate-200 [&::-moz-range-thumb]:border-0"
          style={{
            background: `linear-gradient(to right, ${Math.abs(value) > 0.01 ? "rgba(244,63,94,0.4)" : "rgba(100,116,139,0.2)"} 0%, ${Math.abs(value) > 0.01 ? "rgba(244,63,94,0.4)" : "rgba(100,116,139,0.2)"} ${pct}%, rgba(255,255,255,0.08) ${pct}%, rgba(255,255,255,0.08) 100%)`,
          }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Bubble Chart Component (SVG)
// ---------------------------------------------------------------------------

function BubbleChart({
  positions,
  scenario,
  selectedSymbol,
  onSelect,
  width,
  height,
}: {
  positions: StressPosition[];
  scenario: ScenarioParams;
  selectedSymbol: string | null;
  onSelect: (symbol: string | null) => void;
  width: number;
  height: number;
}) {
  const [hoveredSymbol, setHoveredSymbol] = useState<string | null>(null);

  const { bubbles, maxAbsPnl, sectorPositions } = useMemo(() => {
    const active = isScenarioActive(scenario);
    const stressedPnls = new Map<string, number>();
    positions.forEach((pos) => {
      stressedPnls.set(pos.symbol, active ? computeStressedPnl(pos, scenario) : pos.unrealized_pnl);
    });

    const pnlValues = [...stressedPnls.values()];
    const mxAbsPnl = Math.max(...pnlValues.map(Math.abs), 1);

    // Group by sector
    const sectorGroups = new Map<string, StressPosition[]>();
    for (const pos of positions) {
      const group = sectorGroups.get(pos.sector) ?? [];
      group.push(pos);
      sectorGroups.set(pos.sector, group);
    }

    const presentSectors = SECTOR_ORDER.filter((s) => sectorGroups.has(s));
    const sectorCount = Math.max(presentSectors.length, 1);

    const padding = { left: 50, right: 20, top: 30, bottom: 45 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;
    const colWidth = chartW / sectorCount;

    // Beta range
    const betas = positions.map((p) => p.factor_exposures.equity_beta);
    const minBeta = Math.min(...betas, 0.5);
    const maxBeta = Math.max(...betas, 1.5);
    const betaRange = Math.max(maxBeta - minBeta, 0.5);

    // Max weight for radius scaling
    const maxWeight = Math.max(...positions.map((p) => p.weight), 0.01);

    const allBubbles: Array<{
      pos: StressPosition;
      cx: number;
      cy: number;
      r: number;
      fill: string;
      stroke: string;
      stressedPnl: number;
    }> = [];

    const sectorPos: Array<{ sector: string; cx: number }> = [];

    presentSectors.forEach((sector, sectorIdx) => {
      const group = sectorGroups.get(sector) ?? [];
      const centerX = padding.left + colWidth * sectorIdx + colWidth / 2;
      sectorPos.push({ sector, cx: centerX });

      // Sort by beta within sector for consistent layout
      const sorted = [...group].sort(
        (a, b) => a.factor_exposures.equity_beta - b.factor_exposures.equity_beta,
      );

      sorted.forEach((pos, i) => {
        const beta = pos.factor_exposures.equity_beta;
        const normalizedBeta = (beta - minBeta) / betaRange;
        const cy = padding.top + chartH * (1 - normalizedBeta);

        // Jitter within column to avoid overlap
        const jitter = (i - (sorted.length - 1) / 2) * 12;
        const cx = centerX + jitter;

        const r = Math.max(8, Math.min(45, Math.sqrt(pos.weight / maxWeight) * 35));
        const pnl = stressedPnls.get(pos.symbol) ?? 0;

        allBubbles.push({
          pos,
          cx,
          cy: Math.max(padding.top + r, Math.min(padding.top + chartH - r, cy)),
          r,
          fill: stressColor(pnl, mxAbsPnl),
          stroke: stressBorderColor(pnl, mxAbsPnl),
          stressedPnl: pnl,
        });
      });
    });

    return { bubbles: allBubbles, maxAbsPnl: mxAbsPnl, sectorPositions: sectorPos };
  }, [positions, scenario, width, height]);

  const hovered = bubbles.find((b) => b.pos.symbol === hoveredSymbol);

  return (
    <svg width={width} height={height} className="select-none">
      {/* Grid lines */}
      {[0.25, 0.5, 0.75].map((t) => (
        <line
          key={t}
          x1={50}
          y1={30 + (height - 75) * t}
          x2={width - 20}
          y2={30 + (height - 75) * t}
          stroke="rgba(255,255,255,0.04)"
          strokeDasharray="4 4"
        />
      ))}

      {/* Sector column dividers */}
      {sectorPositions.map(({ sector, cx }) => {
        const colW = (width - 70) / Math.max(sectorPositions.length, 1);
        return (
          <g key={sector}>
            <line
              x1={cx - colW / 2}
              y1={30}
              x2={cx - colW / 2}
              y2={height - 45}
              stroke="rgba(255,255,255,0.03)"
            />
            <text
              x={cx}
              y={height - 18}
              textAnchor="middle"
              fill={SECTOR_COLORS[sector] ?? "#64748b"}
              fontSize={10}
              fontWeight={700}
              letterSpacing="0.1em"
              fontFamily="monospace"
            >
              {SECTOR_LABELS[sector] ?? sector.toUpperCase()}
            </text>
          </g>
        );
      })}

      {/* Y-axis label */}
      <text
        x={14}
        y={height / 2}
        textAnchor="middle"
        fill="rgba(148,163,184,0.5)"
        fontSize={9}
        fontWeight={600}
        letterSpacing="0.15em"
        transform={`rotate(-90, 14, ${height / 2})`}
      >
        EQUITY BETA
      </text>

      {/* Bubbles */}
      {bubbles.map((b) => {
        const isSelected = selectedSymbol === b.pos.symbol;
        const isHovered = hoveredSymbol === b.pos.symbol;
        return (
          <g key={b.pos.symbol}>
            {/* Glow effect for selected */}
            {isSelected && (
              <motion.circle
                cx={b.cx}
                cy={b.cy}
                r={b.r + 6}
                fill="none"
                stroke="rgba(6,182,212,0.4)"
                strokeWidth={2}
                initial={false}
                animate={{ r: b.r + 6 }}
              />
            )}

            <motion.circle
              cx={b.cx}
              cy={b.cy}
              r={b.r}
              fill={b.fill}
              stroke={isSelected ? "#06b6d4" : b.stroke}
              strokeWidth={isSelected ? 2 : 1}
              initial={false}
              animate={{
                r: isHovered ? b.r + 3 : b.r,
                fillOpacity: isHovered ? 1 : 0.85,
              }}
              transition={{ type: "spring", stiffness: 300, damping: 25 }}
              className="cursor-pointer"
              onMouseEnter={() => setHoveredSymbol(b.pos.symbol)}
              onMouseLeave={() => setHoveredSymbol(null)}
              onClick={() => onSelect(isSelected ? null : b.pos.symbol)}
            />

            {/* Symbol label */}
            <text
              x={b.cx}
              y={b.cy + 1}
              textAnchor="middle"
              dominantBaseline="central"
              fill="white"
              fontSize={b.r > 18 ? 10 : 8}
              fontWeight={700}
              fontFamily="monospace"
              className="pointer-events-none"
              opacity={0.9}
            >
              {b.pos.symbol}
            </text>
          </g>
        );
      })}

      {/* Hover tooltip */}
      {hovered && (
        <g>
          <rect
            x={Math.min(hovered.cx + hovered.r + 8, width - 185)}
            y={Math.max(hovered.cy - 42, 5)}
            width={175}
            height={80}
            rx={8}
            fill="rgba(15,23,42,0.95)"
            stroke="rgba(255,255,255,0.1)"
          />
          <text
            x={Math.min(hovered.cx + hovered.r + 18, width - 175)}
            y={Math.max(hovered.cy - 24, 23)}
            fill="#e2e8f0"
            fontSize={12}
            fontWeight={700}
            fontFamily="monospace"
          >
            {hovered.pos.symbol}
          </text>
          <text
            x={Math.min(hovered.cx + hovered.r + 18, width - 175)}
            y={Math.max(hovered.cy - 7, 40)}
            fill="#94a3b8"
            fontSize={10}
            fontFamily="monospace"
          >
            MV: {fmtDollar(hovered.pos.market_value)} | Wt: {(hovered.pos.weight * 100).toFixed(1)}%
          </text>
          <text
            x={Math.min(hovered.cx + hovered.r + 18, width - 175)}
            y={Math.max(hovered.cy + 9, 56)}
            fill="#94a3b8"
            fontSize={10}
            fontFamily="monospace"
          >
            Beta: {hovered.pos.factor_exposures.equity_beta.toFixed(2)} | Vol: {(hovered.pos.volatility_30d * 100).toFixed(1)}%
          </text>
          <text
            x={Math.min(hovered.cx + hovered.r + 18, width - 175)}
            y={Math.max(hovered.cy + 25, 72)}
            fill={hovered.stressedPnl >= 0 ? "#10b981" : "#f43f5e"}
            fontSize={11}
            fontWeight={700}
            fontFamily="monospace"
          >
            Stress P&L: {hovered.stressedPnl >= 0 ? "+" : ""}${Math.round(hovered.stressedPnl).toLocaleString()}
          </text>
        </g>
      )}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Signal Lineage Drawer
// ---------------------------------------------------------------------------

function LineageDrawer({
  symbol,
  position,
  stressedPnl,
  onClose,
}: {
  symbol: string;
  position: StressPosition;
  stressedPnl: number;
  onClose: () => void;
}) {
  const [events, setEvents] = useState<LineageEvent[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    api
      .get<LineageResponse>(`/stress-map/lineage/${symbol}`)
      .then((res) => {
        if (!cancelled) setEvents(res.data.events ?? []);
      })
      .catch(() => {
        if (!cancelled) setEvents([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [symbol]);

  const stageIcon = (stage: string) => {
    if (stage.includes("Data")) return <Activity size={14} />;
    if (stage.includes("Feature")) return <Zap size={14} />;
    if (stage.includes("Model") || stage.includes("Signal")) return <Target size={14} />;
    if (stage.includes("Risk")) return <Shield size={14} />;
    if (stage.includes("Order") || stage.includes("Execution")) return <ArrowRight size={14} />;
    return <Crosshair size={14} />;
  };

  const statusColor = (status: string) => {
    if (status === "success") return "bg-emerald-500";
    if (status === "warning") return "bg-amber-500";
    return "bg-rose-500";
  };

  return (
    <motion.div
      initial={{ x: "100%", opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: "100%", opacity: 0 }}
      transition={{ type: "spring", damping: 25, stiffness: 200 }}
      className="absolute inset-y-0 right-0 z-20 w-[380px] border-l border-white/[0.08] bg-slate-950/98 backdrop-blur-xl flex flex-col overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/[0.08] bg-white/[0.02] px-4 py-3">
        <div>
          <h3 className="text-sm font-bold text-slate-100 font-mono">{symbol}</h3>
          <p className="text-[10px] text-slate-500 uppercase tracking-wider">Signal Lineage</p>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 text-slate-400 hover:text-slate-200 transition-colors rounded-md hover:bg-white/[0.1]"
        >
          <X size={16} />
        </button>
      </div>

      {/* Position Details */}
      <div className="border-b border-white/[0.06] p-4 space-y-3">
        <div className="grid grid-cols-2 gap-3 text-xs font-mono">
          <div>
            <p className="text-[9px] text-slate-500 uppercase tracking-wider">Quantity</p>
            <p className="text-slate-200 font-bold">{position.quantity}</p>
          </div>
          <div>
            <p className="text-[9px] text-slate-500 uppercase tracking-wider">Market Value</p>
            <p className="text-slate-200 font-bold">{fmtDollar(position.market_value)}</p>
          </div>
          <div>
            <p className="text-[9px] text-slate-500 uppercase tracking-wider">Entry</p>
            <p className="text-slate-200">${position.avg_entry_price.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-[9px] text-slate-500 uppercase tracking-wider">Current</p>
            <p className="text-slate-200">${position.current_price.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-[9px] text-slate-500 uppercase tracking-wider">Unrealized P&L</p>
            <p className={position.unrealized_pnl >= 0 ? "text-emerald-400 font-bold" : "text-rose-400 font-bold"}>
              {fmtDollar(position.unrealized_pnl)} ({fmtPct(position.unrealized_pnl_pct)})
            </p>
          </div>
          <div>
            <p className="text-[9px] text-slate-500 uppercase tracking-wider">Stress Impact</p>
            <p className={stressedPnl >= 0 ? "text-emerald-400 font-bold" : "text-rose-400 font-bold"}>
              {stressedPnl >= 0 ? "+" : ""}${Math.round(stressedPnl).toLocaleString()}
            </p>
          </div>
        </div>

        {/* Factor Exposures Mini Bar */}
        <div>
          <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-2">Factor Exposures</p>
          <div className="space-y-1">
            {[
              { label: "Eq Beta", value: position.factor_exposures.equity_beta, max: 2 },
              { label: "Rates", value: position.factor_exposures.rate_sensitivity, max: 1 },
              { label: "VIX", value: position.factor_exposures.vix_sensitivity, max: 2 },
              { label: "Credit", value: position.factor_exposures.credit_sensitivity, max: 1 },
              { label: "Liq", value: position.factor_exposures.liquidity_score, max: 1 },
              { label: "USD", value: position.factor_exposures.usd_sensitivity, max: 1 },
            ].map((f) => (
              <div key={f.label} className="flex items-center gap-2 text-[10px] font-mono">
                <span className="w-12 text-slate-500">{f.label}</span>
                <div className="flex-1 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-cyan-500/60"
                    style={{ width: `${Math.min(Math.abs(f.value) / f.max, 1) * 100}%` }}
                  />
                </div>
                <span className="w-10 text-right text-slate-400">{f.value.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="flex-1 overflow-y-auto p-4">
        <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-3">Event Timeline</p>
        {loading ? (
          <div className="flex items-center justify-center py-8 text-xs text-slate-500">Loading lineage...</div>
        ) : events.length === 0 ? (
          <div className="flex items-center justify-center py-8 text-xs text-slate-500">No lineage data available</div>
        ) : (
          <div className="relative ml-3">
            {/* Vertical line */}
            <div className="absolute left-[6px] top-2 bottom-2 w-px bg-white/[0.08]" />

            {events.map((event, i) => (
              <div key={`${event.stage}-${i}`} className="relative flex gap-3 pb-4">
                {/* Dot */}
                <div className={`relative z-10 mt-1 w-3.5 h-3.5 rounded-full ${statusColor(event.status)} flex items-center justify-center shrink-0`}>
                  <div className="w-1.5 h-1.5 rounded-full bg-white/50" />
                </div>

                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-slate-400">{stageIcon(event.stage)}</span>
                    <span className="text-xs font-semibold text-slate-200">{event.stage}</span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed break-words">{event.detail}</p>
                  <p className="text-[9px] text-slate-600 font-mono mt-0.5">
                    {new Date(event.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Main Page Component
// ---------------------------------------------------------------------------

export default function StressMapPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 900, height: 550 });
  const [positions, setPositions] = useState<StressPosition[]>([]);
  const [portfolioValue, setPortfolioValue] = useState(0);
  const [scenario, setScenario] = useState<ScenarioParams>({ ...DEFAULT_SCENARIO });
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [presetName, setPresetName] = useState("Custom");
  const [loading, setLoading] = useState(true);
  const [presetOpen, setPresetOpen] = useState(false);

  // Fetch positions with factor exposures
  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    api
      .get<SnapshotResponse>("/stress-map/snapshot")
      .then((res) => {
        if (!cancelled) {
          setPositions(res.data.positions ?? []);
          setPortfolioValue(res.data.portfolio_value ?? 0);
        }
      })
      .catch(() => {
        if (!cancelled) setPositions([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  // Resize observer
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setDimensions({
          width: Math.floor(entry.contentRect.width),
          height: Math.floor(entry.contentRect.height),
        });
      }
    });

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Computed stress results
  const stressResults = useMemo(() => {
    const active = isScenarioActive(scenario);
    const map = new Map<string, number>();
    let totalImpact = 0;
    let worstSymbol = "";
    let worstPnl = 0;
    let bestSymbol = "";
    let bestPnl = 0;

    for (const pos of positions) {
      const pnl = active ? computeStressedPnl(pos, scenario) : 0;
      map.set(pos.symbol, pnl);
      totalImpact += pnl;
      if (pnl < worstPnl) {
        worstPnl = pnl;
        worstSymbol = pos.symbol;
      }
      if (pnl > bestPnl) {
        bestPnl = pnl;
        bestSymbol = pos.symbol;
      }
    }

    const hedgeRatio =
      totalImpact < 0 && bestPnl > 0 ? Math.min((bestPnl / Math.abs(totalImpact)) * 100, 100) : 0;

    return { map, totalImpact, worstSymbol, worstPnl, bestSymbol, bestPnl, hedgeRatio };
  }, [positions, scenario]);

  const narrative = useMemo(
    () => generateNarrative(positions, scenario, portfolioValue),
    [positions, scenario, portfolioValue],
  );

  const selectedPosition = positions.find((p) => p.symbol === selectedSymbol);

  const updateSlider = (key: keyof ScenarioParams, value: number) => {
    setScenario((prev) => ({ ...prev, [key]: value }));
    setPresetName("Custom");
  };

  const applyPreset = (name: string) => {
    const preset = PRESETS[name];
    if (preset) {
      setScenario({ ...preset });
      setPresetName(name);
    }
    setPresetOpen(false);
  };

  const resetScenario = () => {
    setScenario({ ...DEFAULT_SCENARIO });
    setPresetName("Custom");
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="flex flex-col h-[calc(100vh-120px)]"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-rose-500/20 text-rose-400">
              <AlertTriangle size={16} />
            </div>
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-[0.24em] text-rose-500/70">
                Interactive Scenario Analysis
              </p>
              <h1 className="text-xl font-bold tracking-tight text-slate-100">
                Stress Map
              </h1>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Preset Dropdown */}
          <div className="relative">
            <button
              onClick={() => setPresetOpen(!presetOpen)}
              className="flex items-center gap-2 rounded-lg border border-white/[0.1] bg-white/[0.03] px-3 py-1.5 text-xs font-mono text-slate-300 hover:bg-white/[0.06] transition-colors"
            >
              <Zap size={12} className="text-amber-400" />
              {presetName}
              <ChevronDown size={12} className={`transition-transform ${presetOpen ? "rotate-180" : ""}`} />
            </button>
            {presetOpen && (
              <div className="absolute right-0 top-full mt-1 z-30 w-48 rounded-lg border border-white/[0.1] bg-slate-900/98 backdrop-blur-xl shadow-xl">
                {Object.keys(PRESETS).map((name) => (
                  <button
                    key={name}
                    onClick={() => applyPreset(name)}
                    className={`w-full text-left px-3 py-2 text-xs font-mono transition-colors hover:bg-white/[0.06] ${
                      presetName === name ? "text-cyan-400 bg-cyan-500/10" : "text-slate-300"
                    }`}
                  >
                    {name}
                  </button>
                ))}
              </div>
            )}
          </div>

          <button
            onClick={resetScenario}
            className="flex items-center gap-1.5 rounded-lg border border-white/[0.1] bg-white/[0.03] px-3 py-1.5 text-xs text-slate-400 hover:text-slate-200 hover:bg-white/[0.06] transition-colors"
          >
            <RotateCcw size={12} />
            Reset
          </button>
        </div>
      </div>

      {/* Main Layout */}
      <div className="flex-1 flex gap-4 overflow-hidden">
        {/* Bubble Chart Area */}
        <div className="relative flex-1 rounded-xl border border-white/[0.08] bg-white/[0.02] overflow-hidden">
          <div ref={containerRef} className="w-full h-full">
            {loading ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-sm text-slate-500">Loading portfolio positions...</div>
              </div>
            ) : positions.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full gap-2">
                <AlertTriangle size={24} className="text-slate-500" />
                <p className="text-sm text-slate-500">No open positions to visualize</p>
              </div>
            ) : (
              <BubbleChart
                positions={positions}
                scenario={scenario}
                selectedSymbol={selectedSymbol}
                onSelect={setSelectedSymbol}
                width={dimensions.width}
                height={dimensions.height}
              />
            )}
          </div>

          {/* Lineage Drawer */}
          <AnimatePresence>
            {selectedSymbol && selectedPosition && (
              <LineageDrawer
                symbol={selectedSymbol}
                position={selectedPosition}
                stressedPnl={stressResults.map.get(selectedSymbol) ?? 0}
                onClose={() => setSelectedSymbol(null)}
              />
            )}
          </AnimatePresence>
        </div>

        {/* Slider Panel */}
        <div className="w-[260px] shrink-0 flex flex-col gap-4 overflow-y-auto">
          {/* Scenario Controls */}
          <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-4">
            <div className="flex items-center gap-2 mb-4">
              <span className="inline-block h-2 w-2 rounded-full bg-rose-400 shadow-[0_0_8px_rgba(244,63,94,0.6)]" />
              <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-rose-400/80">
                Scenario Controls
              </span>
            </div>

            <div className="space-y-4">
              <ScenarioSlider
                label="VIX Shock"
                value={scenario.vix_shock_pct}
                min={-50}
                max={200}
                step={5}
                unit="%"
                onChange={(v) => updateSlider("vix_shock_pct", v)}
                colorClass="text-rose-400"
              />
              <ScenarioSlider
                label="Rate Move"
                value={scenario.rate_move_bps}
                min={-200}
                max={300}
                step={10}
                unit="bp"
                onChange={(v) => updateSlider("rate_move_bps", v)}
                colorClass="text-amber-400"
              />
              <ScenarioSlider
                label="Equity Drawdown"
                value={scenario.equity_shock_pct}
                min={-30}
                max={0}
                step={1}
                unit="%"
                onChange={(v) => updateSlider("equity_shock_pct", v)}
                colorClass="text-rose-400"
              />
              <ScenarioSlider
                label="Credit Spread"
                value={scenario.credit_spread_bps}
                min={-100}
                max={500}
                step={10}
                unit="bp"
                onChange={(v) => updateSlider("credit_spread_bps", v)}
                colorClass="text-amber-400"
              />
              <ScenarioSlider
                label="Liquidity Drain"
                value={scenario.liquidity_drain_pct}
                min={-80}
                max={0}
                step={5}
                unit="%"
                onChange={(v) => updateSlider("liquidity_drain_pct", v)}
                colorClass="text-purple-400"
              />
              <ScenarioSlider
                label="USD Move"
                value={scenario.usd_move_pct}
                min={-10}
                max={10}
                step={0.5}
                unit="%"
                onChange={(v) => updateSlider("usd_move_pct", v)}
                colorClass="text-cyan-400"
              />
            </div>
          </div>

          {/* Impact Summary */}
          <div className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-4">
            <div className="flex items-center gap-2 mb-3">
              <span className="inline-block h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(6,182,212,0.6)]" />
              <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-400/80">
                Impact Summary
              </span>
            </div>

            <div className="space-y-3">
              <div>
                <p className="text-[9px] text-slate-500 uppercase tracking-wider">Portfolio &Delta;P&L</p>
                <p
                  className={`text-lg font-bold font-mono ${
                    !isScenarioActive(scenario)
                      ? "text-slate-500"
                      : stressResults.totalImpact >= 0
                      ? "text-emerald-400"
                      : "text-rose-400"
                  }`}
                >
                  {isScenarioActive(scenario)
                    ? `${stressResults.totalImpact >= 0 ? "+" : ""}$${Math.abs(Math.round(stressResults.totalImpact)).toLocaleString()}`
                    : "—"}
                </p>
                {isScenarioActive(scenario) && portfolioValue > 0 && (
                  <p className="text-[10px] text-slate-500 font-mono">
                    {fmtPct((stressResults.totalImpact / portfolioValue) * 100)} of portfolio
                  </p>
                )}
              </div>

              {isScenarioActive(scenario) && stressResults.worstSymbol && (
                <div>
                  <p className="text-[9px] text-slate-500 uppercase tracking-wider">Max Loss Position</p>
                  <p className="text-sm font-bold font-mono text-rose-400">
                    {stressResults.worstSymbol}{" "}
                    <span className="text-xs">{fmtDollar(stressResults.worstPnl)}</span>
                  </p>
                </div>
              )}

              {isScenarioActive(scenario) && stressResults.bestSymbol && stressResults.bestPnl > 0 && (
                <div>
                  <p className="text-[9px] text-slate-500 uppercase tracking-wider">Best Hedge</p>
                  <p className="text-sm font-bold font-mono text-emerald-400">
                    {stressResults.bestSymbol}{" "}
                    <span className="text-xs">+{fmtDollar(stressResults.bestPnl)}</span>
                  </p>
                </div>
              )}

              {isScenarioActive(scenario) && stressResults.hedgeRatio > 0 && (
                <div>
                  <p className="text-[9px] text-slate-500 uppercase tracking-wider">Hedge Coverage</p>
                  <div className="flex items-center gap-2 mt-0.5">
                    <div className="flex-1 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full bg-emerald-500/70"
                        style={{ width: `${stressResults.hedgeRatio}%` }}
                      />
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">
                      {stressResults.hedgeRatio.toFixed(0)}%
                    </span>
                  </div>
                </div>
              )}

              <div className="pt-2 border-t border-white/[0.06]">
                <p className="text-[9px] text-slate-500 uppercase tracking-wider">Positions</p>
                <p className="text-sm font-mono text-slate-300">{positions.length}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Narrative Strip */}
      <div className="mt-3 rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3">
        <div className="flex items-start gap-3">
          <div className="shrink-0 mt-0.5">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-fuchsia-500/20 text-fuchsia-400">
              <Target size={12} />
            </div>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed font-mono">
            {narrative}
          </p>
        </div>
      </div>
    </motion.div>
  );
}
