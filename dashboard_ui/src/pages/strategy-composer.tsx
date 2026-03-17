import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type ReactNode,
} from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  Handle,
  Position,
  BaseEdge,
  getBezierPath,
  useReactFlow,
  ReactFlowProvider,
  type Node,
  type Edge,
  type Connection,
  type NodeProps,
  type EdgeProps,
  type NodeTypes,
  type EdgeTypes,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  Brain,
  ChevronRight,
  Copy,
  Database,
  Download,
  FlaskConical,
  GitBranch,
  GripVertical,
  Layers,
  Play,
  Plus,
  RotateCcw,
  Save,
  Settings2,
  Shield,
  Sparkles,
  Target,
  Trash2,
  TrendingUp,
  X,
  Zap,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

// ============================================================================
// Types
// ============================================================================

type NodeCategory =
  | "data"
  | "feature"
  | "model"
  | "alpha"
  | "risk"
  | "execution"
  | "output";

interface NodeConfig {
  [key: string]: string | number | boolean | (string | number)[];
}

interface StrategyNodeData {
  label: string;
  category: NodeCategory;
  subtype: string;
  config: NodeConfig;
  status: "idle" | "running" | "success" | "error";
  metrics?: Record<string, number | string>;
  description: string;
  [key: string]: unknown;
}

// ============================================================================
// Constants — Category metadata
// ============================================================================

const CATEGORY_META: Record<
  NodeCategory,
  { color: string; gradient: string; icon: ReactNode; label: string; borderColor: string }
> = {
  data: {
    color: "#06b6d4",
    gradient: "from-cyan-500/20 to-cyan-600/5",
    icon: <Database size={14} />,
    label: "Data Source",
    borderColor: "border-cyan-500/30",
  },
  feature: {
    color: "#8b5cf6",
    gradient: "from-violet-500/20 to-violet-600/5",
    icon: <Sparkles size={14} />,
    label: "Feature Engineering",
    borderColor: "border-violet-500/30",
  },
  model: {
    color: "#f59e0b",
    gradient: "from-amber-500/20 to-amber-600/5",
    icon: <Brain size={14} />,
    label: "ML Model",
    borderColor: "border-amber-500/30",
  },
  alpha: {
    color: "#10b981",
    gradient: "from-emerald-500/20 to-emerald-600/5",
    icon: <TrendingUp size={14} />,
    label: "Alpha Signal",
    borderColor: "border-emerald-500/30",
  },
  risk: {
    color: "#ef4444",
    gradient: "from-rose-500/20 to-rose-600/5",
    icon: <Shield size={14} />,
    label: "Risk Check",
    borderColor: "border-rose-500/30",
  },
  execution: {
    color: "#3b82f6",
    gradient: "from-blue-500/20 to-blue-600/5",
    icon: <Zap size={14} />,
    label: "Execution",
    borderColor: "border-blue-500/30",
  },
  output: {
    color: "#ec4899",
    gradient: "from-pink-500/20 to-pink-600/5",
    icon: <Target size={14} />,
    label: "Output",
    borderColor: "border-pink-500/30",
  },
};

// ============================================================================
// Node Catalog — Available nodes to drag onto canvas
// ============================================================================

interface CatalogEntry {
  subtype: string;
  label: string;
  category: NodeCategory;
  description: string;
  defaultConfig: NodeConfig;
  inputs: number;
  outputs: number;
}

const NODE_CATALOG: CatalogEntry[] = [
  // Data Sources
  {
    subtype: "market_data",
    label: "Market Data",
    category: "data",
    description: "OHLCV price data from Alpaca/DB",
    defaultConfig: { symbols: ["AAPL", "MSFT", "GOOGL"], timeframe: "1D", source: "alpaca", lookback_days: 252 },
    inputs: 0,
    outputs: 1,
  },
  {
    subtype: "alt_data",
    label: "Alternative Data",
    category: "data",
    description: "News sentiment, social media, dark pool",
    defaultConfig: { feeds: ["news_sentiment", "social_volume"], refresh_sec: 60 },
    inputs: 0,
    outputs: 1,
  },
  {
    subtype: "macro_data",
    label: "Macro / VIX",
    category: "data",
    description: "VIX, yields, macro indicators",
    defaultConfig: { indicators: ["VIX", "US10Y", "DXY"], source: "feed" },
    inputs: 0,
    outputs: 1,
  },
  // Feature Engineering
  {
    subtype: "technical",
    label: "Technical Indicators",
    category: "feature",
    description: "RSI, MACD, Bollinger, ATR, etc.",
    defaultConfig: { indicators: ["RSI_14", "MACD", "BB_20", "ATR_14"], normalize: true },
    inputs: 1,
    outputs: 1,
  },
  {
    subtype: "statistical",
    label: "Statistical Features",
    category: "feature",
    description: "Rolling stats, z-scores, correlations",
    defaultConfig: { windows: [5, 10, 20, 60], features: ["zscore", "rolling_vol", "skew"] },
    inputs: 1,
    outputs: 1,
  },
  {
    subtype: "microstructure",
    label: "Microstructure",
    category: "feature",
    description: "Bid-ask spread, order flow, volume profile",
    defaultConfig: { features: ["spread", "vpin", "order_imbalance"], bar_type: "volume" },
    inputs: 1,
    outputs: 1,
  },
  {
    subtype: "cross_sectional",
    label: "Cross-Sectional",
    category: "feature",
    description: "Relative value, sector momentum, rank features",
    defaultConfig: { rank_features: true, sector_neutral: true },
    inputs: 1,
    outputs: 1,
  },
  // ML Models
  {
    subtype: "xgboost",
    label: "XGBoost",
    category: "model",
    description: "Gradient boosted trees classifier",
    defaultConfig: { n_estimators: 500, max_depth: 6, learning_rate: 0.05, purged_cv_folds: 5 },
    inputs: 1,
    outputs: 1,
  },
  {
    subtype: "lightgbm",
    label: "LightGBM",
    category: "model",
    description: "Light gradient boosting machine",
    defaultConfig: { n_estimators: 400, num_leaves: 31, learning_rate: 0.05, purged_cv_folds: 5 },
    inputs: 1,
    outputs: 1,
  },
  {
    subtype: "lstm",
    label: "LSTM Network",
    category: "model",
    description: "Long short-term memory deep learning",
    defaultConfig: { hidden_size: 128, num_layers: 2, dropout: 0.2, seq_length: 20, epochs: 50 },
    inputs: 1,
    outputs: 1,
  },
  {
    subtype: "ensemble",
    label: "Model Ensemble",
    category: "model",
    description: "Combine multiple model predictions",
    defaultConfig: { method: "weighted_average", weights: "auto", meta_learner: false },
    inputs: 3,
    outputs: 1,
  },
  // Alpha Signals
  {
    subtype: "momentum",
    label: "Momentum Alpha",
    category: "alpha",
    description: "Trend-following signal generation",
    defaultConfig: { fast_period: 10, slow_period: 50, signal_threshold: 0.6, decay: 0.95 },
    inputs: 1,
    outputs: 1,
  },
  {
    subtype: "mean_reversion",
    label: "Mean Reversion",
    category: "alpha",
    description: "Counter-trend reversion signals",
    defaultConfig: { lookback: 20, entry_zscore: 2.0, exit_zscore: 0.5, half_life: 10 },
    inputs: 1,
    outputs: 1,
  },
  {
    subtype: "ml_alpha",
    label: "ML-Based Alpha",
    category: "alpha",
    description: "Model prediction → alpha signal",
    defaultConfig: { threshold: 0.55, scale: "rank", combination: "additive" },
    inputs: 1,
    outputs: 1,
  },
  {
    subtype: "alpha_combiner",
    label: "Alpha Combiner",
    category: "alpha",
    description: "Weighted combination of multiple alphas",
    defaultConfig: { method: "ic_weighted", rebalance_freq: "weekly", max_alpha_corr: 0.7 },
    inputs: 3,
    outputs: 1,
  },
  // Risk
  {
    subtype: "position_limits",
    label: "Position Limits",
    category: "risk",
    description: "Max position size & concentration",
    defaultConfig: { max_position_pct: 5, max_sector_pct: 25, max_gross_leverage: 1.5 },
    inputs: 1,
    outputs: 1,
  },
  {
    subtype: "drawdown_guard",
    label: "Drawdown Guard",
    category: "risk",
    description: "Drawdown monitoring & kill switch",
    defaultConfig: { max_drawdown_pct: 10, trailing_stop_pct: 5, cooldown_hours: 24 },
    inputs: 1,
    outputs: 1,
  },
  {
    subtype: "var_check",
    label: "VaR / Stress",
    category: "risk",
    description: "Value-at-Risk & stress test gates",
    defaultConfig: { var_confidence: 0.99, var_limit_pct: 3, stress_scenarios: ["GFC", "COVID"] },
    inputs: 1,
    outputs: 1,
  },
  {
    subtype: "correlation_monitor",
    label: "Correlation Monitor",
    category: "risk",
    description: "Portfolio correlation regime check",
    defaultConfig: { max_avg_corr: 0.6, alert_threshold: 0.8, window: 60 },
    inputs: 1,
    outputs: 1,
  },
  // Execution
  {
    subtype: "twap",
    label: "TWAP",
    category: "execution",
    description: "Time-weighted average price execution",
    defaultConfig: { duration_min: 30, slice_count: 10, max_participation: 0.1 },
    inputs: 1,
    outputs: 1,
  },
  {
    subtype: "vwap",
    label: "VWAP",
    category: "execution",
    description: "Volume-weighted average price execution",
    defaultConfig: { duration_min: 60, volume_profile: "historical", urgency: "medium" },
    inputs: 1,
    outputs: 1,
  },
  {
    subtype: "smart_router",
    label: "Smart Router",
    category: "execution",
    description: "Optimal venue selection & routing",
    defaultConfig: { venues: ["alpaca"], algo: "adaptive", impact_model: true },
    inputs: 1,
    outputs: 1,
  },
  // Output
  {
    subtype: "portfolio_output",
    label: "Portfolio Allocator",
    category: "output",
    description: "Final portfolio construction & rebalance",
    defaultConfig: { optimizer: "mean_variance", rebalance_freq: "daily", target_vol: 0.15 },
    inputs: 1,
    outputs: 0,
  },
  {
    subtype: "performance_log",
    label: "Performance Logger",
    category: "output",
    description: "Log signals & results for analysis",
    defaultConfig: { log_signals: true, log_trades: true, export_format: "parquet" },
    inputs: 1,
    outputs: 0,
  },
];

// ============================================================================
// Animated Edge — Particles flowing along path
// ============================================================================

function AnimatedFlowEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  data,
}: EdgeProps) {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const isActive = (data as Record<string, unknown>)?.active !== false;
  const edgeColor = (data as Record<string, unknown>)?.color as string ?? "rgba(6, 182, 212, 0.4)";

  return (
    <>
      {/* Background path */}
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          ...style,
          stroke: "rgba(255,255,255,0.06)",
          strokeWidth: 2,
        }}
      />
      {/* Colored overlay */}
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          ...style,
          stroke: edgeColor,
          strokeWidth: 2,
        }}
      />
      {/* Animated particles */}
      {isActive && (
        <>
          <circle r="3" fill={edgeColor} filter="url(#glow)">
            <animateMotion dur="2s" repeatCount="indefinite" path={edgePath} />
          </circle>
          <circle r="3" fill={edgeColor} filter="url(#glow)">
            <animateMotion dur="2s" repeatCount="indefinite" path={edgePath} begin="0.66s" />
          </circle>
          <circle r="3" fill={edgeColor} filter="url(#glow)">
            <animateMotion dur="2s" repeatCount="indefinite" path={edgePath} begin="1.33s" />
          </circle>
        </>
      )}
    </>
  );
}

// ============================================================================
// Strategy Node Component
// ============================================================================

function StrategyNode({ data, selected }: NodeProps<Node<StrategyNodeData>>) {
  const meta = CATEGORY_META[data.category];
  const catalogEntry = NODE_CATALOG.find((e) => e.subtype === data.subtype);
  const inputCount = catalogEntry?.inputs ?? 1;
  const outputCount = catalogEntry?.outputs ?? 1;

  const statusIndicator = () => {
    switch (data.status) {
      case "running":
        return (
          <span className="flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse" />
            <span className="text-[8px] text-amber-400 font-mono">RUN</span>
          </span>
        );
      case "success":
        return (
          <span className="flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            <span className="text-[8px] text-emerald-400 font-mono">OK</span>
          </span>
        );
      case "error":
        return (
          <span className="flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-rose-400" />
            <span className="text-[8px] text-rose-400 font-mono">ERR</span>
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-slate-500" />
            <span className="text-[8px] text-slate-500 font-mono">IDLE</span>
          </span>
        );
    }
  };

  return (
    <div
      className={`relative rounded-xl border bg-slate-950/95 backdrop-blur-sm transition-all duration-200 ${
        selected
          ? `${meta.borderColor} shadow-[0_0_20px_rgba(0,0,0,0.5)] ring-1 ring-white/10`
          : "border-white/[0.08] hover:border-white/[0.15]"
      }`}
      style={{
        minWidth: 200,
        boxShadow: selected
          ? `0 0 30px ${meta.color}15, 0 4px 20px rgba(0,0,0,0.4)`
          : "0 4px 20px rgba(0,0,0,0.3)",
      }}
    >
      {/* Input handles */}
      {inputCount > 0 &&
        Array.from({ length: inputCount }).map((_, i) => (
          <Handle
            key={`input-${i}`}
            type="target"
            position={Position.Left}
            id={`input-${i}`}
            style={{
              top: `${((i + 1) / (inputCount + 1)) * 100}%`,
              width: 10,
              height: 10,
              background: "rgba(15,23,42,0.9)",
              border: `2px solid ${meta.color}`,
              borderRadius: "50%",
            }}
          />
        ))}

      {/* Output handles */}
      {outputCount > 0 &&
        Array.from({ length: outputCount }).map((_, i) => (
          <Handle
            key={`output-${i}`}
            type="source"
            position={Position.Right}
            id={`output-${i}`}
            style={{
              top: `${((i + 1) / (outputCount + 1)) * 100}%`,
              width: 10,
              height: 10,
              background: meta.color,
              border: "2px solid rgba(15,23,42,0.9)",
              borderRadius: "50%",
            }}
          />
        ))}

      {/* Header */}
      <div
        className={`flex items-center justify-between rounded-t-xl bg-gradient-to-r ${meta.gradient} px-3 py-2 border-b border-white/[0.06]`}
      >
        <div className="flex items-center gap-2">
          <div
            className="flex h-6 w-6 items-center justify-center rounded-md"
            style={{ backgroundColor: `${meta.color}25`, color: meta.color }}
          >
            {meta.icon}
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.15em]" style={{ color: `${meta.color}99` }}>
              {meta.label}
            </p>
            <p className="text-xs font-bold text-slate-100">{data.label}</p>
          </div>
        </div>
        {statusIndicator()}
      </div>

      {/* Body */}
      <div className="px-3 py-2.5 space-y-2">
        <p className="text-[10px] text-slate-500 leading-relaxed">{data.description}</p>

        {/* Mini config preview */}
        <div className="space-y-1">
          {Object.entries(data.config)
            .slice(0, 3)
            .map(([key, val]) => (
              <div key={key} className="flex items-center justify-between text-[9px] font-mono">
                <span className="text-slate-500">{key}</span>
                <span className="text-slate-300 truncate max-w-[100px]">
                  {Array.isArray(val) ? val.join(", ") : String(val)}
                </span>
              </div>
            ))}
          {Object.keys(data.config).length > 3 && (
            <p className="text-[8px] text-slate-600 text-right">
              +{Object.keys(data.config).length - 3} more params
            </p>
          )}
        </div>

        {/* Metrics (when running/success) */}
        {data.metrics && Object.keys(data.metrics).length > 0 && (
          <div className="mt-1 pt-1.5 border-t border-white/[0.04] grid grid-cols-2 gap-x-3 gap-y-0.5">
            {Object.entries(data.metrics).map(([key, val]) => (
              <div key={key} className="flex items-center justify-between text-[9px] font-mono">
                <span className="text-slate-500">{key}</span>
                <span
                  className={
                    typeof val === "number" && val >= 0
                      ? "text-emerald-400"
                      : typeof val === "number" && val < 0
                      ? "text-rose-400"
                      : "text-cyan-400"
                  }
                >
                  {typeof val === "number" ? val.toFixed(2) : val}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Node & Edge type registry
// ============================================================================

const nodeTypes: NodeTypes = {
  strategyNode: StrategyNode as NodeTypes["strategyNode"],
};

const edgeTypes: EdgeTypes = {
  animatedFlow: AnimatedFlowEdge as EdgeTypes["animatedFlow"],
};

// ============================================================================
// Template strategies
// ============================================================================

interface StrategyTemplate {
  name: string;
  description: string;
  nodes: Node<StrategyNodeData>[];
  edges: Edge[];
}

const TEMPLATES: StrategyTemplate[] = [
  {
    name: "Momentum Pipeline",
    description: "Classic momentum strategy with technical features → XGBoost → risk → TWAP",
    nodes: [
      {
        id: "t1-data",
        type: "strategyNode",
        position: { x: 50, y: 200 },
        data: {
          label: "Market Data",
          category: "data",
          subtype: "market_data",
          config: { symbols: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"], timeframe: "1D", source: "alpaca", lookback_days: 252 },
          status: "idle",
          description: "OHLCV price data from Alpaca/DB",
        },
      },
      {
        id: "t1-tech",
        type: "strategyNode",
        position: { x: 350, y: 120 },
        data: {
          label: "Technical Indicators",
          category: "feature",
          subtype: "technical",
          config: { indicators: ["RSI_14", "MACD", "BB_20", "ATR_14", "ADX_14"], normalize: true },
          status: "idle",
          description: "RSI, MACD, Bollinger, ATR, ADX",
        },
      },
      {
        id: "t1-stat",
        type: "strategyNode",
        position: { x: 350, y: 310 },
        data: {
          label: "Statistical Features",
          category: "feature",
          subtype: "statistical",
          config: { windows: [5, 10, 20, 60], features: ["zscore", "rolling_vol", "skew"] },
          status: "idle",
          description: "Rolling stats, z-scores, volatility",
        },
      },
      {
        id: "t1-xgb",
        type: "strategyNode",
        position: { x: 680, y: 200 },
        data: {
          label: "XGBoost",
          category: "model",
          subtype: "xgboost",
          config: { n_estimators: 500, max_depth: 6, learning_rate: 0.05, purged_cv_folds: 5 },
          status: "idle",
          description: "Gradient boosted trees classifier",
        },
      },
      {
        id: "t1-mom",
        type: "strategyNode",
        position: { x: 980, y: 200 },
        data: {
          label: "Momentum Alpha",
          category: "alpha",
          subtype: "momentum",
          config: { fast_period: 10, slow_period: 50, signal_threshold: 0.6, decay: 0.95 },
          status: "idle",
          description: "Trend-following signal generation",
        },
      },
      {
        id: "t1-risk",
        type: "strategyNode",
        position: { x: 1280, y: 140 },
        data: {
          label: "Position Limits",
          category: "risk",
          subtype: "position_limits",
          config: { max_position_pct: 5, max_sector_pct: 25, max_gross_leverage: 1.5 },
          status: "idle",
          description: "Max position size & concentration",
        },
      },
      {
        id: "t1-dd",
        type: "strategyNode",
        position: { x: 1280, y: 330 },
        data: {
          label: "Drawdown Guard",
          category: "risk",
          subtype: "drawdown_guard",
          config: { max_drawdown_pct: 10, trailing_stop_pct: 5, cooldown_hours: 24 },
          status: "idle",
          description: "Drawdown monitoring & kill switch",
        },
      },
      {
        id: "t1-twap",
        type: "strategyNode",
        position: { x: 1580, y: 200 },
        data: {
          label: "TWAP",
          category: "execution",
          subtype: "twap",
          config: { duration_min: 30, slice_count: 10, max_participation: 0.1 },
          status: "idle",
          description: "Time-weighted average price execution",
        },
      },
      {
        id: "t1-out",
        type: "strategyNode",
        position: { x: 1880, y: 200 },
        data: {
          label: "Portfolio Allocator",
          category: "output",
          subtype: "portfolio_output",
          config: { optimizer: "mean_variance", rebalance_freq: "daily", target_vol: 0.15 },
          status: "idle",
          description: "Final portfolio construction",
        },
      },
    ],
    edges: [
      { id: "t1-e1", source: "t1-data", target: "t1-tech", type: "animatedFlow", data: { color: "rgba(6,182,212,0.5)" } },
      { id: "t1-e2", source: "t1-data", target: "t1-stat", type: "animatedFlow", data: { color: "rgba(6,182,212,0.5)" } },
      { id: "t1-e3", source: "t1-tech", target: "t1-xgb", type: "animatedFlow", data: { color: "rgba(139,92,246,0.5)" } },
      { id: "t1-e4", source: "t1-stat", target: "t1-xgb", type: "animatedFlow", data: { color: "rgba(139,92,246,0.5)" } },
      { id: "t1-e5", source: "t1-xgb", target: "t1-mom", type: "animatedFlow", data: { color: "rgba(245,158,11,0.5)" } },
      { id: "t1-e6", source: "t1-mom", target: "t1-risk", type: "animatedFlow", data: { color: "rgba(16,185,129,0.5)" } },
      { id: "t1-e7", source: "t1-mom", target: "t1-dd", type: "animatedFlow", data: { color: "rgba(16,185,129,0.5)" } },
      { id: "t1-e8", source: "t1-risk", target: "t1-twap", type: "animatedFlow", data: { color: "rgba(239,68,68,0.5)" } },
      { id: "t1-e9", source: "t1-dd", target: "t1-twap", type: "animatedFlow", data: { color: "rgba(239,68,68,0.5)" } },
      { id: "t1-e10", source: "t1-twap", target: "t1-out", type: "animatedFlow", data: { color: "rgba(59,130,246,0.5)" } },
    ],
  },
  {
    name: "Mean Reversion + LSTM",
    description: "Deep learning mean reversion with microstructure features → ensemble → VaR risk",
    nodes: [
      {
        id: "t2-data",
        type: "strategyNode",
        position: { x: 50, y: 200 },
        data: {
          label: "Market Data",
          category: "data",
          subtype: "market_data",
          config: { symbols: ["AAPL", "JPM", "XOM"], timeframe: "5min", source: "alpaca", lookback_days: 60 },
          status: "idle",
          description: "Intraday OHLCV data",
        },
      },
      {
        id: "t2-micro",
        type: "strategyNode",
        position: { x: 350, y: 120 },
        data: {
          label: "Microstructure",
          category: "feature",
          subtype: "microstructure",
          config: { features: ["spread", "vpin", "order_imbalance"], bar_type: "volume" },
          status: "idle",
          description: "Order flow & volume profile",
        },
      },
      {
        id: "t2-stat",
        type: "strategyNode",
        position: { x: 350, y: 310 },
        data: {
          label: "Statistical Features",
          category: "feature",
          subtype: "statistical",
          config: { windows: [5, 10, 20], features: ["zscore", "hurst", "halflife"] },
          status: "idle",
          description: "Mean reversion statistical features",
        },
      },
      {
        id: "t2-lstm",
        type: "strategyNode",
        position: { x: 680, y: 200 },
        data: {
          label: "LSTM Network",
          category: "model",
          subtype: "lstm",
          config: { hidden_size: 128, num_layers: 2, dropout: 0.2, seq_length: 20, epochs: 50 },
          status: "idle",
          description: "Sequence model for pattern detection",
        },
      },
      {
        id: "t2-mr",
        type: "strategyNode",
        position: { x: 980, y: 200 },
        data: {
          label: "Mean Reversion",
          category: "alpha",
          subtype: "mean_reversion",
          config: { lookback: 20, entry_zscore: 2.0, exit_zscore: 0.5, half_life: 10 },
          status: "idle",
          description: "Counter-trend reversion signals",
        },
      },
      {
        id: "t2-var",
        type: "strategyNode",
        position: { x: 1280, y: 200 },
        data: {
          label: "VaR / Stress",
          category: "risk",
          subtype: "var_check",
          config: { var_confidence: 0.99, var_limit_pct: 3, stress_scenarios: ["GFC", "COVID", "FlashCrash"] },
          status: "idle",
          description: "VaR & stress test gates",
        },
      },
      {
        id: "t2-vwap",
        type: "strategyNode",
        position: { x: 1580, y: 200 },
        data: {
          label: "VWAP",
          category: "execution",
          subtype: "vwap",
          config: { duration_min: 60, volume_profile: "historical", urgency: "medium" },
          status: "idle",
          description: "VWAP execution algorithm",
        },
      },
      {
        id: "t2-out",
        type: "strategyNode",
        position: { x: 1880, y: 200 },
        data: {
          label: "Portfolio Allocator",
          category: "output",
          subtype: "portfolio_output",
          config: { optimizer: "min_variance", rebalance_freq: "intraday", target_vol: 0.10 },
          status: "idle",
          description: "Conservative portfolio construction",
        },
      },
    ],
    edges: [
      { id: "t2-e1", source: "t2-data", target: "t2-micro", type: "animatedFlow", data: { color: "rgba(6,182,212,0.5)" } },
      { id: "t2-e2", source: "t2-data", target: "t2-stat", type: "animatedFlow", data: { color: "rgba(6,182,212,0.5)" } },
      { id: "t2-e3", source: "t2-micro", target: "t2-lstm", type: "animatedFlow", data: { color: "rgba(139,92,246,0.5)" } },
      { id: "t2-e4", source: "t2-stat", target: "t2-lstm", type: "animatedFlow", data: { color: "rgba(139,92,246,0.5)" } },
      { id: "t2-e5", source: "t2-lstm", target: "t2-mr", type: "animatedFlow", data: { color: "rgba(245,158,11,0.5)" } },
      { id: "t2-e6", source: "t2-mr", target: "t2-var", type: "animatedFlow", data: { color: "rgba(16,185,129,0.5)" } },
      { id: "t2-e7", source: "t2-var", target: "t2-vwap", type: "animatedFlow", data: { color: "rgba(239,68,68,0.5)" } },
      { id: "t2-e8", source: "t2-vwap", target: "t2-out", type: "animatedFlow", data: { color: "rgba(59,130,246,0.5)" } },
    ],
  },
  {
    name: "Multi-Alpha Ensemble",
    description: "Multiple alpha sources combined with ML ensemble → full risk stack → smart routing",
    nodes: [
      {
        id: "t3-data",
        type: "strategyNode",
        position: { x: 50, y: 250 },
        data: {
          label: "Market Data",
          category: "data",
          subtype: "market_data",
          config: { symbols: ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM"], timeframe: "1D", source: "alpaca", lookback_days: 504 },
          status: "idle",
          description: "Broad universe daily data",
        },
      },
      {
        id: "t3-alt",
        type: "strategyNode",
        position: { x: 50, y: 450 },
        data: {
          label: "Alternative Data",
          category: "data",
          subtype: "alt_data",
          config: { feeds: ["news_sentiment", "social_volume", "insider_flow"], refresh_sec: 300 },
          status: "idle",
          description: "News, social, insider signals",
        },
      },
      {
        id: "t3-tech",
        type: "strategyNode",
        position: { x: 380, y: 150 },
        data: {
          label: "Technical Indicators",
          category: "feature",
          subtype: "technical",
          config: { indicators: ["RSI_14", "MACD", "BB_20", "ATR_14", "OBV", "VWAP"], normalize: true },
          status: "idle",
          description: "Full technical indicator suite",
        },
      },
      {
        id: "t3-cross",
        type: "strategyNode",
        position: { x: 380, y: 350 },
        data: {
          label: "Cross-Sectional",
          category: "feature",
          subtype: "cross_sectional",
          config: { rank_features: true, sector_neutral: true },
          status: "idle",
          description: "Relative value & sector momentum",
        },
      },
      {
        id: "t3-xgb",
        type: "strategyNode",
        position: { x: 720, y: 100 },
        data: {
          label: "XGBoost",
          category: "model",
          subtype: "xgboost",
          config: { n_estimators: 800, max_depth: 7, learning_rate: 0.03, purged_cv_folds: 5 },
          status: "idle",
          description: "Primary boosting model",
        },
      },
      {
        id: "t3-lgb",
        type: "strategyNode",
        position: { x: 720, y: 280 },
        data: {
          label: "LightGBM",
          category: "model",
          subtype: "lightgbm",
          config: { n_estimators: 600, num_leaves: 63, learning_rate: 0.03, purged_cv_folds: 5 },
          status: "idle",
          description: "Secondary boosting model",
        },
      },
      {
        id: "t3-lstm",
        type: "strategyNode",
        position: { x: 720, y: 460 },
        data: {
          label: "LSTM Network",
          category: "model",
          subtype: "lstm",
          config: { hidden_size: 256, num_layers: 3, dropout: 0.3, seq_length: 30, epochs: 100 },
          status: "idle",
          description: "Deep sequence model",
        },
      },
      {
        id: "t3-ens",
        type: "strategyNode",
        position: { x: 1050, y: 280 },
        data: {
          label: "Model Ensemble",
          category: "model",
          subtype: "ensemble",
          config: { method: "stacking", weights: "auto", meta_learner: true },
          status: "idle",
          description: "Stacked ensemble with meta-learner",
        },
      },
      {
        id: "t3-alpha",
        type: "strategyNode",
        position: { x: 1350, y: 280 },
        data: {
          label: "ML-Based Alpha",
          category: "alpha",
          subtype: "ml_alpha",
          config: { threshold: 0.55, scale: "rank", combination: "additive" },
          status: "idle",
          description: "Model predictions → alpha signal",
        },
      },
      {
        id: "t3-risk1",
        type: "strategyNode",
        position: { x: 1650, y: 170 },
        data: {
          label: "Position Limits",
          category: "risk",
          subtype: "position_limits",
          config: { max_position_pct: 4, max_sector_pct: 20, max_gross_leverage: 1.2 },
          status: "idle",
          description: "Conservative position limits",
        },
      },
      {
        id: "t3-risk2",
        type: "strategyNode",
        position: { x: 1650, y: 380 },
        data: {
          label: "Correlation Monitor",
          category: "risk",
          subtype: "correlation_monitor",
          config: { max_avg_corr: 0.5, alert_threshold: 0.7, window: 60 },
          status: "idle",
          description: "Correlation regime detection",
        },
      },
      {
        id: "t3-exec",
        type: "strategyNode",
        position: { x: 1950, y: 280 },
        data: {
          label: "Smart Router",
          category: "execution",
          subtype: "smart_router",
          config: { venues: ["alpaca"], algo: "adaptive", impact_model: true },
          status: "idle",
          description: "Adaptive smart order routing",
        },
      },
      {
        id: "t3-out",
        type: "strategyNode",
        position: { x: 2250, y: 280 },
        data: {
          label: "Portfolio Allocator",
          category: "output",
          subtype: "portfolio_output",
          config: { optimizer: "risk_parity", rebalance_freq: "daily", target_vol: 0.12 },
          status: "idle",
          description: "Risk-parity allocation",
        },
      },
    ],
    edges: [
      { id: "t3-e1", source: "t3-data", target: "t3-tech", type: "animatedFlow", data: { color: "rgba(6,182,212,0.5)" } },
      { id: "t3-e2", source: "t3-data", target: "t3-cross", type: "animatedFlow", data: { color: "rgba(6,182,212,0.5)" } },
      { id: "t3-e3", source: "t3-alt", target: "t3-cross", type: "animatedFlow", data: { color: "rgba(6,182,212,0.5)" } },
      { id: "t3-e4", source: "t3-tech", target: "t3-xgb", type: "animatedFlow", data: { color: "rgba(139,92,246,0.5)" } },
      { id: "t3-e5", source: "t3-tech", target: "t3-lgb", type: "animatedFlow", data: { color: "rgba(139,92,246,0.5)" } },
      { id: "t3-e6", source: "t3-cross", target: "t3-lgb", type: "animatedFlow", data: { color: "rgba(139,92,246,0.5)" } },
      { id: "t3-e7", source: "t3-cross", target: "t3-lstm", type: "animatedFlow", data: { color: "rgba(139,92,246,0.5)" } },
      { id: "t3-e8", source: "t3-xgb", target: "t3-ens", sourceHandle: "output-0", targetHandle: "input-0", type: "animatedFlow", data: { color: "rgba(245,158,11,0.5)" } },
      { id: "t3-e9", source: "t3-lgb", target: "t3-ens", sourceHandle: "output-0", targetHandle: "input-1", type: "animatedFlow", data: { color: "rgba(245,158,11,0.5)" } },
      { id: "t3-e10", source: "t3-lstm", target: "t3-ens", sourceHandle: "output-0", targetHandle: "input-2", type: "animatedFlow", data: { color: "rgba(245,158,11,0.5)" } },
      { id: "t3-e11", source: "t3-ens", target: "t3-alpha", type: "animatedFlow", data: { color: "rgba(245,158,11,0.5)" } },
      { id: "t3-e12", source: "t3-alpha", target: "t3-risk1", type: "animatedFlow", data: { color: "rgba(16,185,129,0.5)" } },
      { id: "t3-e13", source: "t3-alpha", target: "t3-risk2", type: "animatedFlow", data: { color: "rgba(16,185,129,0.5)" } },
      { id: "t3-e14", source: "t3-risk1", target: "t3-exec", type: "animatedFlow", data: { color: "rgba(239,68,68,0.5)" } },
      { id: "t3-e15", source: "t3-risk2", target: "t3-exec", type: "animatedFlow", data: { color: "rgba(239,68,68,0.5)" } },
      { id: "t3-e16", source: "t3-exec", target: "t3-out", type: "animatedFlow", data: { color: "rgba(59,130,246,0.5)" } },
    ],
  },
];

// ============================================================================
// Properties Panel
// ============================================================================

function PropertiesPanel({
  node,
  onUpdate,
  onDelete,
  onDuplicate,
  onClose,
}: {
  node: Node<StrategyNodeData>;
  onUpdate: (id: string, data: Partial<StrategyNodeData>) => void;
  onDelete: (id: string) => void;
  onDuplicate: (node: Node<StrategyNodeData>) => void;
  onClose: () => void;
}) {
  const meta = CATEGORY_META[node.data.category];
  const [editConfig, setEditConfig] = useState<NodeConfig>({ ...node.data.config });

  useEffect(() => {
    setEditConfig({ ...node.data.config });
  }, [node.id, node.data.config]);

  const handleConfigChange = (key: string, value: string) => {
    const current = node.data.config[key];
    let parsed: string | number | boolean | string[];

    if (Array.isArray(current)) {
      parsed = value.split(",").map((s) => s.trim());
    } else if (typeof current === "number") {
      parsed = Number(value);
      if (isNaN(parsed)) return;
    } else if (typeof current === "boolean") {
      parsed = value === "true";
    } else {
      parsed = value;
    }

    const newConfig = { ...editConfig, [key]: parsed };
    setEditConfig(newConfig);
    onUpdate(node.id, { config: newConfig });
  };

  return (
    <motion.div
      initial={{ x: 320, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 320, opacity: 0 }}
      transition={{ type: "spring", damping: 25, stiffness: 200 }}
      className="absolute right-0 top-0 bottom-0 z-30 w-[320px] border-l border-white/[0.08] bg-slate-950/98 backdrop-blur-xl flex flex-col overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/[0.08] px-4 py-3">
        <div className="flex items-center gap-2">
          <div
            className="flex h-7 w-7 items-center justify-center rounded-lg"
            style={{ backgroundColor: `${meta.color}20`, color: meta.color }}
          >
            {meta.icon}
          </div>
          <div>
            <p className="text-xs font-bold text-slate-100">{node.data.label}</p>
            <p className="text-[9px] uppercase tracking-wider" style={{ color: `${meta.color}80` }}>
              {meta.label}
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 text-slate-400 hover:text-slate-200 transition-colors rounded-md hover:bg-white/[0.08]"
        >
          <X size={14} />
        </button>
      </div>

      {/* Description */}
      <div className="border-b border-white/[0.06] px-4 py-3">
        <p className="text-[10px] text-slate-400 leading-relaxed">{node.data.description}</p>
      </div>

      {/* Config */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        <div className="flex items-center gap-1.5 mb-1">
          <Settings2 size={12} className="text-slate-500" />
          <p className="text-[9px] font-bold uppercase tracking-[0.2em] text-slate-500">
            Parameters
          </p>
        </div>

        {Object.entries(editConfig).map(([key, val]) => (
          <label key={key} className="block">
            <span className="block text-[10px] font-mono text-slate-500 mb-1">
              {key}
            </span>
            {typeof val === "boolean" ? (
              <select
                className="w-full h-8 rounded-lg border border-white/[0.1] bg-white/[0.04] px-2.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-white/[0.2]"
                value={String(val)}
                onChange={(e) => handleConfigChange(key, e.target.value)}
              >
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            ) : (
              <input
                type={typeof val === "number" ? "number" : "text"}
                className="w-full h-8 rounded-lg border border-white/[0.1] bg-white/[0.04] px-2.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-white/[0.2]"
                value={Array.isArray(val) ? val.join(", ") : String(val)}
                onChange={(e) => handleConfigChange(key, e.target.value)}
                step={typeof val === "number" && val < 1 ? 0.01 : 1}
              />
            )}
          </label>
        ))}
      </div>

      {/* Actions */}
      <div className="border-t border-white/[0.08] px-4 py-3 space-y-2">
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            className="flex-1 gap-1.5 h-8 text-xs"
            onClick={() => onDuplicate(node)}
          >
            <Copy size={12} />
            Duplicate
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="flex-1 gap-1.5 h-8 text-xs text-rose-400 hover:text-rose-300 border-rose-500/20 hover:border-rose-500/40"
            onClick={() => onDelete(node.id)}
          >
            <Trash2 size={12} />
            Delete
          </Button>
        </div>
      </div>
    </motion.div>
  );
}

// ============================================================================
// Sidebar — Node palette
// ============================================================================

function NodePalette({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  const [search, setSearch] = useState("");
  const [expandedCategory, setExpandedCategory] = useState<NodeCategory | null>("data");

  const filteredCatalog = useMemo(() => {
    if (!search) return NODE_CATALOG;
    const q = search.toLowerCase();
    return NODE_CATALOG.filter(
      (n) =>
        n.label.toLowerCase().includes(q) ||
        n.description.toLowerCase().includes(q) ||
        n.category.includes(q),
    );
  }, [search]);

  const grouped = useMemo(() => {
    const groups = new Map<NodeCategory, CatalogEntry[]>();
    for (const entry of filteredCatalog) {
      const list = groups.get(entry.category) ?? [];
      list.push(entry);
      groups.set(entry.category, list);
    }
    return groups;
  }, [filteredCatalog]);

  const onDragStart = (e: DragEvent, entry: CatalogEntry) => {
    e.dataTransfer.setData("application/strategy-node", JSON.stringify(entry));
    e.dataTransfer.effectAllowed = "move";
  };

  if (collapsed) {
    return (
      <div className="w-12 border-r border-white/[0.06] bg-slate-950/90 flex flex-col items-center py-3 gap-2">
        <button
          onClick={onToggle}
          className="p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-white/[0.06] transition-colors"
          title="Open palette"
        >
          <Plus size={16} />
        </button>
        <div className="w-6 border-t border-white/[0.08] my-1" />
        {(Object.entries(CATEGORY_META) as [NodeCategory, typeof CATEGORY_META[NodeCategory]][]).map(
          ([cat, meta]) => (
            <div
              key={cat}
              className="flex h-8 w-8 items-center justify-center rounded-lg transition-colors hover:bg-white/[0.06]"
              style={{ color: meta.color }}
              title={meta.label}
            >
              {meta.icon}
            </div>
          ),
        )}
      </div>
    );
  }

  return (
    <div className="w-[260px] border-r border-white/[0.06] bg-slate-950/90 flex flex-col overflow-hidden shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/[0.08] px-3 py-2.5">
        <div className="flex items-center gap-2">
          <Layers size={14} className="text-cyan-400" />
          <p className="text-xs font-bold text-slate-200">Node Palette</p>
        </div>
        <button
          onClick={onToggle}
          className="p-1 text-slate-400 hover:text-slate-200 transition-colors rounded"
        >
          <X size={14} />
        </button>
      </div>

      {/* Search */}
      <div className="px-3 py-2 border-b border-white/[0.06]">
        <input
          type="text"
          placeholder="Search nodes..."
          className="w-full h-7 rounded-md border border-white/[0.08] bg-white/[0.03] px-2 text-[11px] text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-white/[0.15]"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {/* Categories */}
      <div className="flex-1 overflow-y-auto py-1">
        {(
          ["data", "feature", "model", "alpha", "risk", "execution", "output"] as NodeCategory[]
        ).map((cat) => {
          const entries = grouped.get(cat);
          if (!entries || entries.length === 0) return null;
          const meta = CATEGORY_META[cat];
          const isExpanded = expandedCategory === cat || search.length > 0;

          return (
            <div key={cat}>
              <button
                onClick={() => setExpandedCategory(isExpanded && !search ? null : cat)}
                className="flex w-full items-center justify-between px-3 py-2 text-left hover:bg-white/[0.03] transition-colors"
              >
                <div className="flex items-center gap-2">
                  <div
                    className="flex h-5 w-5 items-center justify-center rounded"
                    style={{ backgroundColor: `${meta.color}20`, color: meta.color }}
                  >
                    {meta.icon}
                  </div>
                  <span className="text-[11px] font-semibold text-slate-300">{meta.label}</span>
                  <Badge variant="outline" className="text-[8px] px-1 py-0">
                    {entries.length}
                  </Badge>
                </div>
                <ChevronRight
                  size={12}
                  className={`text-slate-500 transition-transform ${isExpanded ? "rotate-90" : ""}`}
                />
              </button>

              {isExpanded && (
                <div className="pb-1">
                  {entries.map((entry) => (
                    <div
                      key={entry.subtype}
                      draggable
                      onDragStart={(e) => onDragStart(e, entry)}
                      className="group mx-2 mb-1 flex cursor-grab items-start gap-2 rounded-lg border border-transparent px-2.5 py-2 transition-all hover:border-white/[0.08] hover:bg-white/[0.03] active:cursor-grabbing"
                    >
                      <GripVertical
                        size={12}
                        className="mt-0.5 text-slate-600 group-hover:text-slate-400 shrink-0"
                      />
                      <div className="min-w-0">
                        <p className="text-[11px] font-medium text-slate-200">{entry.label}</p>
                        <p className="text-[9px] text-slate-500 leading-relaxed mt-0.5">
                          {entry.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Tip */}
      <div className="border-t border-white/[0.06] px-3 py-2">
        <p className="text-[9px] text-slate-600 text-center">
          Drag nodes onto the canvas to build your strategy pipeline
        </p>
      </div>
    </div>
  );
}

// ============================================================================
// Simulation Controls Bar
// ============================================================================

function SimulationBar({
  nodeCount,
  edgeCount,
  isSimulating,
  simulationProgress,
  onSimulate,
  onReset,
  onSave,
  onExport,
}: {
  nodeCount: number;
  edgeCount: number;
  isSimulating: boolean;
  simulationProgress: number;
  onSimulate: () => void;
  onReset: () => void;
  onSave: () => void;
  onExport: () => void;
}) {
  return (
    <div className="flex items-center justify-between border-t border-white/[0.08] bg-slate-950/90 px-4 py-2">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3 text-[10px] font-mono text-slate-500">
          <span>
            <span className="text-slate-300 font-bold">{nodeCount}</span> nodes
          </span>
          <span>
            <span className="text-slate-300 font-bold">{edgeCount}</span> edges
          </span>
        </div>

        {isSimulating && (
          <div className="flex items-center gap-2">
            <div className="w-32 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-cyan-500 to-emerald-500 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${simulationProgress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
            <span className="text-[10px] font-mono text-cyan-400">{simulationProgress}%</span>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          className="h-7 gap-1.5 text-[10px]"
          onClick={onExport}
        >
          <Download size={11} />
          Export
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="h-7 gap-1.5 text-[10px]"
          onClick={onSave}
        >
          <Save size={11} />
          Save
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="h-7 gap-1.5 text-[10px]"
          onClick={onReset}
        >
          <RotateCcw size={11} />
          Clear
        </Button>
        <Button
          size="sm"
          className={`h-8 gap-1.5 text-xs font-bold ${
            isSimulating
              ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
              : "bg-gradient-to-r from-cyan-600 to-emerald-600 text-white hover:from-cyan-500 hover:to-emerald-500"
          }`}
          onClick={onSimulate}
          disabled={nodeCount === 0}
        >
          {isSimulating ? (
            <>
              <Activity size={13} className="animate-pulse" />
              Simulating...
            </>
          ) : (
            <>
              <Play size={13} />
              Run Backtest
            </>
          )}
        </Button>
      </div>
    </div>
  );
}

// ============================================================================
// Main Composer (inner, needs ReactFlowProvider)
// ============================================================================

let nodeIdCounter = 0;
function nextId() {
  nodeIdCounter += 1;
  return `node-${Date.now()}-${nodeIdCounter}`;
}

function ComposerInner() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();

  const [nodes, setNodes, onNodesChange] = useNodesState<Node<StrategyNodeData>>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  const [paletteCollapsed, setPaletteCollapsed] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationProgress, setSimulationProgress] = useState(0);
  const [showTemplates, setShowTemplates] = useState(true);

  // Selected node object
  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedNodeId) ?? null,
    [nodes, selectedNodeId],
  );

  // Connect edges
  const onConnect = useCallback(
    (connection: Connection) => {
      // Determine edge color based on source node category
      const sourceNode = nodes.find((n) => n.id === connection.source);
      const color = sourceNode
        ? `${CATEGORY_META[(sourceNode.data as StrategyNodeData).category].color}80`
        : "rgba(6,182,212,0.5)";

      setEdges((eds) =>
        addEdge(
          {
            ...connection,
            type: "animatedFlow",
            data: { color, active: true },
            markerEnd: { type: MarkerType.ArrowClosed, color, width: 15, height: 15 },
          },
          eds,
        ),
      );
    },
    [nodes, setEdges],
  );

  // Handle node selection
  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNodeId(node.id);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
  }, []);

  // Drop handler
  const onDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      const raw = e.dataTransfer.getData("application/strategy-node");
      if (!raw) return;

      const entry: CatalogEntry = JSON.parse(raw);
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });

      const newNode: Node<StrategyNodeData> = {
        id: nextId(),
        type: "strategyNode",
        position,
        data: {
          label: entry.label,
          category: entry.category,
          subtype: entry.subtype,
          config: { ...entry.defaultConfig },
          status: "idle",
          description: entry.description,
        },
      };

      setNodes((nds) => [...nds, newNode]);
      setShowTemplates(false);
    },
    [screenToFlowPosition, setNodes],
  );

  // Node operations
  const updateNodeData = useCallback(
    (id: string, updates: Partial<StrategyNodeData>) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, ...updates } as StrategyNodeData } : n,
        ),
      );
    },
    [setNodes],
  );

  const deleteNode = useCallback(
    (id: string) => {
      setNodes((nds) => nds.filter((n) => n.id !== id));
      setEdges((eds) => eds.filter((e) => e.source !== id && e.target !== id));
      if (selectedNodeId === id) setSelectedNodeId(null);
    },
    [selectedNodeId, setNodes, setEdges],
  );

  const duplicateNode = useCallback(
    (node: Node<StrategyNodeData>) => {
      const newNode: Node<StrategyNodeData> = {
        ...node,
        id: nextId(),
        position: { x: node.position.x + 40, y: node.position.y + 40 },
        data: { ...node.data, config: { ...node.data.config }, status: "idle" as const },
        selected: false,
      };
      setNodes((nds) => [...nds, newNode]);
    },
    [setNodes],
  );

  // Load template
  const loadTemplate = useCallback(
    (template: StrategyTemplate) => {
      setNodes(template.nodes.map((n) => ({ ...n, data: { ...n.data } })));
      setEdges(template.edges.map((e) => ({ ...e })));
      setShowTemplates(false);
      setSelectedNodeId(null);
    },
    [setNodes, setEdges],
  );

  // Clear canvas
  const clearCanvas = useCallback(() => {
    setNodes([]);
    setEdges([]);
    setSelectedNodeId(null);
    setShowTemplates(true);
    setIsSimulating(false);
    setSimulationProgress(0);
  }, [setNodes, setEdges]);

  // Simulate backtest
  const runSimulation = useCallback(() => {
    if (isSimulating || nodes.length === 0) return;
    setIsSimulating(true);
    setSimulationProgress(0);

    // Reset all node statuses
    setNodes((nds) =>
      nds.map((n) => ({ ...n, data: { ...n.data, status: "idle" as const, metrics: undefined } })),
    );

    // Simulate progress through pipeline stages
    const stages: NodeCategory[] = ["data", "feature", "model", "alpha", "risk", "execution", "output"];
    let currentStage = 0;

    const interval = setInterval(() => {
      if (currentStage >= stages.length) {
        clearInterval(interval);
        setIsSimulating(false);
        setSimulationProgress(100);

        // Set final metrics on output nodes
        setNodes((nds) =>
          nds.map((n) => {
            if ((n.data as StrategyNodeData).category === "output") {
              return {
                ...n,
                data: {
                  ...n.data,
                  status: "success" as const,
                  metrics: {
                    sharpe: 1.42 + Math.random() * 0.5,
                    returns: 12.5 + Math.random() * 8,
                    maxDD: -(5 + Math.random() * 5),
                    winRate: 55 + Math.random() * 10,
                  },
                },
              };
            }
            return n;
          }),
        );
        return;
      }

      const stage = stages[currentStage];
      setSimulationProgress(Math.round(((currentStage + 1) / stages.length) * 100));

      setNodes((nds) =>
        nds.map((n) => {
          const d = n.data as StrategyNodeData;
          if (d.category === stage) {
            const metrics: Record<string, number> = {};
            if (stage === "model") {
              metrics["accuracy"] = 0.58 + Math.random() * 0.12;
              metrics["IC"] = 0.03 + Math.random() * 0.04;
            } else if (stage === "alpha") {
              metrics["signal_str"] = 0.4 + Math.random() * 0.3;
              metrics["turnover"] = 0.1 + Math.random() * 0.2;
            } else if (stage === "risk") {
              metrics["VaR_95"] = -(1 + Math.random() * 2);
              metrics["pass"] = 1;
            } else if (stage === "feature") {
              metrics["features"] = 50 + Math.round(Math.random() * 150);
            } else if (stage === "execution") {
              metrics["slippage_bp"] = 1 + Math.random() * 3;
              metrics["fill_rate"] = 0.95 + Math.random() * 0.05;
            }
            return {
              ...n,
              data: {
                ...d,
                status: "success" as const,
                metrics: Object.keys(metrics).length > 0 ? metrics : undefined,
              },
            };
          }
          if (stages.indexOf(d.category) < currentStage) {
            return n; // Already processed
          }
          if (stages.indexOf(d.category) === currentStage + 1) {
            return { ...n, data: { ...d, status: "running" as const } };
          }
          return n;
        }),
      );

      currentStage++;
    }, 800);

    // Start first stage
    setNodes((nds) =>
      nds.map((n) => {
        const d = n.data as StrategyNodeData;
        if (d.category === "data") {
          return { ...n, data: { ...d, status: "running" as const } };
        }
        return n;
      }),
    );
  }, [isSimulating, nodes.length, setNodes]);

  // Export
  const handleExport = useCallback(() => {
    const pipeline = {
      name: "custom_strategy",
      created_at: new Date().toISOString(),
      nodes: nodes.map((n) => ({
        id: n.id,
        category: (n.data as StrategyNodeData).category,
        subtype: (n.data as StrategyNodeData).subtype,
        label: (n.data as StrategyNodeData).label,
        config: (n.data as StrategyNodeData).config,
        position: n.position,
      })),
      edges: edges.map((e) => ({
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle,
        targetHandle: e.targetHandle,
      })),
    };

    const blob = new Blob([JSON.stringify(pipeline, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `strategy_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [nodes, edges]);

  // Save to localStorage
  const handleSave = useCallback(() => {
    const state = {
      nodes: nodes.map((n) => ({ ...n, selected: false })),
      edges,
    };
    localStorage.setItem("alphatrade_strategy_composer", JSON.stringify(state));
  }, [nodes, edges]);

  // Load from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("alphatrade_strategy_composer");
    if (saved) {
      try {
        const state = JSON.parse(saved);
        if (state.nodes?.length > 0) {
          setNodes(state.nodes);
          setEdges(state.edges ?? []);
          setShowTemplates(false);
        }
      } catch {
        // corrupt state, ignore
      }
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex h-full w-full overflow-hidden">
      {/* Node Palette */}
      <NodePalette
        collapsed={paletteCollapsed}
        onToggle={() => setPaletteCollapsed(!paletteCollapsed)}
      />

      {/* Canvas + bottom bar */}
      <div className="flex-1 flex flex-col overflow-hidden relative">
        <div
          ref={reactFlowWrapper}
          className="flex-1"
          onDragOver={onDragOver}
          onDrop={onDrop}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            defaultEdgeOptions={{
              type: "animatedFlow",
              animated: true,
            }}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            minZoom={0.2}
            maxZoom={2}
            snapToGrid
            snapGrid={[20, 20]}
            className="!bg-transparent"
            proOptions={{ hideAttribution: true }}
          >
            {/* SVG filters for glow effects */}
            <svg width={0} height={0}>
              <defs>
                <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
            </svg>
            <Background color="rgba(255,255,255,0.03)" gap={20} size={1} />
            <Controls
              className="!bg-slate-900/90 !border-white/[0.08] !rounded-lg [&>button]:!bg-transparent [&>button]:!border-white/[0.06] [&>button]:!text-slate-400 [&>button:hover]:!bg-white/[0.06] [&>button:hover]:!text-slate-200"
              showInteractive={false}
            />
            <MiniMap
              className="!bg-slate-900/80 !border-white/[0.08] !rounded-lg"
              nodeColor={(n) => {
                const d = n.data as StrategyNodeData;
                return CATEGORY_META[d.category]?.color ?? "#64748b";
              }}
              maskColor="rgba(0,0,0,0.7)"
            />
          </ReactFlow>

          {/* Empty state — Template selector */}
          {showTemplates && nodes.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
              <div className="pointer-events-auto w-[700px] space-y-6">
                <div className="text-center">
                  <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-500/20 to-emerald-500/20 border border-cyan-500/20">
                    <GitBranch size={28} className="text-cyan-400" />
                  </div>
                  <h2 className="text-2xl font-bold text-slate-100">Strategy Composer</h2>
                  <p className="mt-2 text-sm text-slate-400 max-w-md mx-auto">
                    Build trading strategies visually. Drag nodes from the palette or start with a template.
                  </p>
                </div>

                <div className="grid grid-cols-3 gap-3">
                  {TEMPLATES.map((tmpl) => (
                    <button
                      key={tmpl.name}
                      onClick={() => loadTemplate(tmpl)}
                      className="group rounded-xl border border-white/[0.08] bg-slate-950/80 p-4 text-left transition-all hover:border-cyan-500/30 hover:bg-white/[0.03]"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <FlaskConical size={14} className="text-cyan-400" />
                        <p className="text-xs font-bold text-slate-200 group-hover:text-cyan-300 transition-colors">
                          {tmpl.name}
                        </p>
                      </div>
                      <p className="text-[10px] text-slate-500 leading-relaxed">
                        {tmpl.description}
                      </p>
                      <div className="mt-3 flex items-center gap-2 text-[9px] font-mono text-slate-600">
                        <span>{tmpl.nodes.length} nodes</span>
                        <span>&middot;</span>
                        <span>{tmpl.edges.length} edges</span>
                      </div>
                    </button>
                  ))}
                </div>

                <p className="text-center text-[10px] text-slate-600">
                  Or drag nodes from the left palette to start from scratch
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Simulation Bar */}
        <SimulationBar
          nodeCount={nodes.length}
          edgeCount={edges.length}
          isSimulating={isSimulating}
          simulationProgress={simulationProgress}
          onSimulate={runSimulation}
          onReset={clearCanvas}
          onSave={handleSave}
          onExport={handleExport}
        />

        {/* Properties Panel */}
        <AnimatePresence>
          {selectedNode && (
            <PropertiesPanel
              node={selectedNode}
              onUpdate={updateNodeData}
              onDelete={deleteNode}
              onDuplicate={duplicateNode}
              onClose={() => setSelectedNodeId(null)}
            />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

// ============================================================================
// Page export (wrapped in provider)
// ============================================================================

export default function StrategyComposerPage() {
  return (
    <div className="-mx-4 -my-5 lg:-mx-6" style={{ height: "calc(100vh - 96px)" }}>
      <ReactFlowProvider>
        <ComposerInner />
      </ReactFlowProvider>
    </div>
  );
}
