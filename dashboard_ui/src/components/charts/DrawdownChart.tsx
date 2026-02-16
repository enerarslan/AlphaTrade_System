import { useEffect, useRef } from "react";
import { createChart, AreaSeries, type IChartApi, type ISeriesApi, type AreaSeriesOptions } from "lightweight-charts";

interface DrawdownChartProps {
  /** Array of { time: "YYYY-MM-DD", value: number } where value is a negative drawdown % (e.g. -0.12 for -12%) */
  data: Array<{ time: string; value: number }>;
  height?: number;
}

export default function DrawdownChart({ data, height = 260 }: DrawdownChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      height,
      layout: {
        background: { color: "transparent" },
        textColor: "#94a3b8",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.03)" },
        horzLines: { color: "rgba(255,255,255,0.03)" },
      },
      crosshair: {
        vertLine: { color: "rgba(244,63,94,0.3)", labelBackgroundColor: "#1e293b" },
        horzLine: { color: "rgba(244,63,94,0.3)", labelBackgroundColor: "#1e293b" },
      },
      rightPriceScale: {
        borderColor: "rgba(255,255,255,0.06)",
      },
      timeScale: {
        borderColor: "rgba(255,255,255,0.06)",
        timeVisible: false,
      },
    });

    const series = chart.addSeries(AreaSeries, {
      lineColor: "#f87171",
      topColor: "rgba(248,113,113,0.25)",
      bottomColor: "rgba(248,113,113,0.02)",
      lineWidth: 2,
      priceFormat: {
        type: "percent" as const,
      },
    } as AreaSeriesOptions);

    chartRef.current = chart;
    seriesRef.current = series;

    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
    };
  }, [height]);

  // Update data
  useEffect(() => {
    if (!seriesRef.current || data.length === 0) return;
    const formatted = data.map((d) => ({
      time: d.time as string,
      value: d.value * 100, // convert to percentage for display
    }));
    seriesRef.current.setData(formatted as Array<{ time: string; value: number }>);
    chartRef.current?.timeScale().fitContent();
  }, [data]);

  return (
    <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-white/[0.02]">
      <div ref={containerRef} />
      {data.length > 0 && (
        <div className="flex items-center justify-between border-t border-white/[0.06] px-3 py-1.5">
          <span className="text-[10px] text-slate-500">
            Max Drawdown: {(Math.min(...data.map((d) => d.value)) * 100).toFixed(2)}%
          </span>
          <span className="text-[10px] text-slate-500">
            Current: {(data[data.length - 1]?.value !== undefined ? data[data.length - 1].value * 100 : 0).toFixed(2)}%
          </span>
        </div>
      )}
    </div>
  );
}
