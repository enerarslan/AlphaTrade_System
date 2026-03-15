import { useMemo } from "react";
import { Treemap, ResponsiveContainer, Tooltip } from "recharts";

type Position = {
  symbol: string;
  quantity: number;
  avg_entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  cost_basis: number;
};

interface PortfolioTreemapProps {
  positions: Position[];
  height?: number;
}

function getColor(pct: number): string {
  if (pct >= 5) return "rgba(16, 185, 129, 0.7)";
  if (pct >= 2) return "rgba(16, 185, 129, 0.5)";
  if (pct >= 0) return "rgba(16, 185, 129, 0.25)";
  if (pct >= -2) return "rgba(244, 63, 94, 0.25)";
  if (pct >= -5) return "rgba(244, 63, 94, 0.5)";
  return "rgba(244, 63, 94, 0.7)";
}

function TreemapCell(props: {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  name?: string;
  pnlPct?: number;
  pnl?: number;
  marketValue?: number;
  [key: string]: unknown;
}) {
  const { x = 0, y = 0, width = 0, height = 0, name, pnlPct = 0, marketValue } = props;

  if (width < 30 || height < 25) return null;

  const fill = getColor(pnlPct);
  const symbolFontSize = width > 80 ? 14 : 11;

  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={4}
        fill={fill}
        stroke="rgba(255,255,255,0.08)"
        strokeWidth={1}
      />
      {width > 50 && height > 40 && (
        <>
          <text
            x={x + width / 2}
            y={y + height / 2 - 8}
            textAnchor="middle"
            fill="#e2e8f0"
            fontSize={symbolFontSize}
            fontWeight="bold"
          >
            {name}
          </text>
          <text
            x={x + width / 2}
            y={y + height / 2 + 10}
            textAnchor="middle"
            fill={pnlPct >= 0 ? "#6ee7b7" : "#fda4af"}
            fontSize={11}
            fontFamily="monospace"
          >
            {pnlPct >= 0 ? "+" : ""}
            {pnlPct?.toFixed(1)}%
          </text>
          {width > 90 && height > 60 && (
            <text
              x={x + width / 2}
              y={y + height / 2 + 26}
              textAnchor="middle"
              fill="#94a3b8"
              fontSize={9}
            >
              ${Math.abs(marketValue ?? 0).toLocaleString()}
            </text>
          )}
        </>
      )}
    </g>
  );
}

function TreemapTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: Record<string, unknown> }> }) {
  if (!active || !payload?.[0]) return null;
  const data = payload[0].payload as {
    name?: string;
    marketValue?: number;
    pnl?: number;
    pnlPct?: number;
    quantity?: number;
    entryPrice?: number;
    currentPrice?: number;
  };

  return (
    <div className="rounded-lg border border-white/[0.08] bg-slate-900/95 px-3 py-2 shadow-xl backdrop-blur-sm">
      <p className="font-bold text-slate-200">{data.name}</p>
      <div className="mt-1 space-y-0.5 text-xs">
        <p className="text-slate-400">
          Market Value:{" "}
          <span className="font-mono text-cyan-300">
            ${data.marketValue?.toLocaleString()}
          </span>
        </p>
        <p className="text-slate-400">
          P&L:{" "}
          <span className={`font-mono ${(data.pnl ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
            {(data.pnl ?? 0) >= 0 ? "+" : ""}${data.pnl?.toLocaleString()}
          </span>
        </p>
        <p className="text-slate-400">
          Return:{" "}
          <span className={`font-mono ${(data.pnlPct ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
            {(data.pnlPct ?? 0) >= 0 ? "+" : ""}
            {data.pnlPct?.toFixed(2)}%
          </span>
        </p>
        <p className="text-slate-400">
          Qty: <span className="font-mono text-slate-200">{data.quantity}</span>
        </p>
        <p className="text-slate-400">
          Entry:{" "}
          <span className="font-mono text-slate-200">${data.entryPrice?.toFixed(2)}</span>
          {" "}→ Current:{" "}
          <span className="font-mono text-slate-200">${data.currentPrice?.toFixed(2)}</span>
        </p>
      </div>
    </div>
  );
}

export default function PortfolioTreemap({ positions, height = 350 }: PortfolioTreemapProps) {
  const treemapData = useMemo(() => {
    if (positions.length === 0) return [];
    return [
      {
        name: "Portfolio",
        children: positions.map((p) => ({
          name: p.symbol,
          size: Math.abs(p.market_value),
          pnl: p.unrealized_pnl,
          pnlPct: p.unrealized_pnl_pct,
          quantity: p.quantity,
          entryPrice: p.avg_entry_price,
          currentPrice: p.current_price,
          marketValue: p.market_value,
        })),
      },
    ];
  }, [positions]);

  if (positions.length === 0) {
    return (
      <div className="flex items-center justify-center" style={{ height }}>
        <p className="text-sm text-slate-500">No positions for treemap visualization.</p>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <Treemap
        data={treemapData}
        dataKey="size"
        aspectRatio={4 / 3}
        stroke="rgba(255,255,255,0.06)"
        content={<TreemapCell />}
      >
        <Tooltip content={<TreemapTooltip />} />
      </Treemap>
    </ResponsiveContainer>
  );
}
