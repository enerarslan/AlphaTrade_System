import { useEffect, useMemo, useRef, useState } from "react";
import {
  AreaSeries,
  ColorType,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type Time,
} from "lightweight-charts";

interface EquityCurveChartProps {
  data: Array<{ time: string; value: number }>;
  height?: number;
  showToolbar?: boolean;
}

const TIME_RANGES = ["1D", "1W", "1M", "3M", "YTD", "1Y", "ALL"] as const;

function filterByRange(data: Array<{ time: string; value: number }>, range: string) {
  if (range === "ALL" || data.length === 0) return data;
  const now = new Date();
  let cutoff: Date;

  switch (range) {
    case "1D":
      cutoff = new Date(now.getTime() - 86400_000);
      break;
    case "1W":
      cutoff = new Date(now.getTime() - 7 * 86400_000);
      break;
    case "1M":
      cutoff = new Date(now.getTime() - 30 * 86400_000);
      break;
    case "3M":
      cutoff = new Date(now.getTime() - 90 * 86400_000);
      break;
    case "YTD":
      cutoff = new Date(now.getFullYear(), 0, 1);
      break;
    case "1Y":
      cutoff = new Date(now.getTime() - 365 * 86400_000);
      break;
    default:
      return data;
  }

  const cutoffStr = cutoff.toISOString().slice(0, 10);
  return data.filter((d) => d.time >= cutoffStr);
}

export default function EquityCurveChart({
  data,
  height = 320,
  showToolbar = true,
}: EquityCurveChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);
  const [range, setRange] = useState<string>("ALL");

  const filteredData = useMemo(() => filterByRange(data, range), [data, range]);

  const isPositive = filteredData.length >= 2
    ? filteredData[filteredData.length - 1].value >= filteredData[0].value
    : true;

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "rgba(148, 163, 184, 0.7)",
        fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
        fontSize: 10,
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.03)" },
        horzLines: { color: "rgba(255,255,255,0.03)" },
      },
      crosshair: {
        vertLine: { color: "rgba(6,182,212,0.3)", width: 1, style: 3 },
        horzLine: { color: "rgba(6,182,212,0.3)", width: 1, style: 3 },
      },
      rightPriceScale: {
        borderColor: "rgba(255,255,255,0.06)",
      },
      timeScale: {
        borderColor: "rgba(255,255,255,0.06)",
        timeVisible: true,
      },
      width: containerRef.current.clientWidth,
      height,
    });

    const areaSeries = chart.addSeries(AreaSeries, {
      lineColor: "#34d399",
      topColor: "rgba(52,211,153,0.25)",
      bottomColor: "rgba(52,211,153,0.02)",
      lineWidth: 2,
    });

    chartRef.current = chart;
    seriesRef.current = areaSeries;

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        chart.applyOptions({ width: entry.contentRect.width });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [height]);

  useEffect(() => {
    if (!seriesRef.current) return;

    seriesRef.current.applyOptions({
      lineColor: isPositive ? "#34d399" : "#f87171",
      topColor: isPositive ? "rgba(52,211,153,0.25)" : "rgba(248,113,113,0.25)",
      bottomColor: isPositive ? "rgba(52,211,153,0.02)" : "rgba(248,113,113,0.02)",
    });

    seriesRef.current.setData(
      filteredData.map((point) => ({ time: point.time as Time, value: point.value })),
    );

    chartRef.current?.timeScale().fitContent();
  }, [filteredData, isPositive]);

  return (
    <div>
      {showToolbar && (
        <div className="mb-2 flex gap-1">
          {TIME_RANGES.map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={`rounded-md px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider transition-colors ${
                range === r
                  ? "bg-cyan-500/20 text-cyan-300"
                  : "text-slate-500 hover:bg-white/5 hover:text-slate-300"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      )}
      <div ref={containerRef} className="rounded-xl" />
    </div>
  );
}
