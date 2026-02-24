import { useEffect, useRef } from "react";
import {
  ColorType,
  createChart,
  HistogramSeries,
  type IChartApi,
  type ISeriesApi,
  type Time,
} from "lightweight-charts";

interface PnLBarChartProps {
  data: Array<{ time: string; value: number }>;
  height?: number;
}

export default function PnLBarChart({ data, height = 200 }: PnLBarChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);

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
      rightPriceScale: { borderColor: "rgba(255,255,255,0.06)" },
      timeScale: { borderColor: "rgba(255,255,255,0.06)" },
      width: containerRef.current.clientWidth,
      height,
    });

    const histSeries = chart.addSeries(HistogramSeries, {
      color: "#34d399",
      priceFormat: { type: "price", precision: 0, minMove: 1 },
    });

    chartRef.current = chart;
    seriesRef.current = histSeries;

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

    seriesRef.current.setData(
      data.map((point) => ({
        time: point.time as Time,
        value: point.value,
        color: point.value >= 0 ? "rgba(52,211,153,0.8)" : "rgba(248,113,113,0.8)",
      })),
    );

    chartRef.current?.timeScale().fitContent();
  }, [data]);

  return <div ref={containerRef} className="rounded-xl" />;
}
