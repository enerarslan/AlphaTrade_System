import { useEffect, useRef, useState } from "react";
import {
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";
import { api } from "@/lib/api";

type MarketBar = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

type SignalRow = {
  signal_id: string;
  timestamp: string;
  direction: string;
};

type CandleData = { time: Time; open: number; high: number; low: number; close: number };
type VolumeData = { time: Time; value: number; color: string };
type RsiData = { time: Time; value: number };

function toChartTime(timestamp: string): Time {
  return Math.floor(new Date(timestamp).getTime() / 1000) as Time;
}

function computeRsi(bars: MarketBar[], period = 14): RsiData[] {
  if (bars.length === 0) {
    return [];
  }

  const rsis: RsiData[] = [];
  let avgGain = 0;
  let avgLoss = 0;

  for (let i = 1; i < bars.length; i += 1) {
    const delta = bars[i].close - bars[i - 1].close;
    const gain = Math.max(delta, 0);
    const loss = Math.max(-delta, 0);

    if (i <= period) {
      avgGain += gain;
      avgLoss += loss;
      if (i === period) {
        avgGain /= period;
        avgLoss /= period;
        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        rsis.push({
          time: toChartTime(bars[i].timestamp),
          value: 100 - 100 / (1 + rs),
        });
      }
      continue;
    }

    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    rsis.push({
      time: toChartTime(bars[i].timestamp),
      value: 100 - 100 / (1 + rs),
    });
  }

  return rsis;
}

export default function AdvancedChartWidget({
  height = 500,
  symbol = "AAPL",
}: {
  height?: number;
  symbol?: string;
}) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [bars, setBars] = useState<MarketBar[]>([]);
  const [signals, setSignals] = useState<SignalRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [barsResponse, signalsResponse] = await Promise.all([
          api.get<{ data: MarketBar[] }>("/market/bars", {
            params: { symbol, timeframe: "15Min", limit: 200 },
          }),
          api.get<SignalRow[]>("/signals", { params: { symbol, limit: 50 } }),
        ]);

        if (cancelled) {
          return;
        }

        setBars(barsResponse.data.data ?? []);
        setSignals(signalsResponse.data ?? []);
      } catch {
        if (!cancelled) {
          setBars([]);
          setSignals([]);
          setError("Market chart data unavailable");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [symbol]);

  useEffect(() => {
    if (!chartContainerRef.current || loading || bars.length === 0) {
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
      return;
    }

    const chart = createChart(chartContainerRef.current, {
      height,
      layout: {
        background: { color: "transparent" },
        textColor: "#94a3b8",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(255, 255, 255, 0.04)" },
        horzLines: { color: "rgba(255, 255, 255, 0.04)" },
      },
      crosshair: {
        mode: 0,
        vertLine: { width: 1, color: "rgba(148, 163, 184, 0.3)", style: 3 },
        horzLine: { width: 1, color: "rgba(148, 163, 184, 0.3)", style: 3 },
      },
      rightPriceScale: {
        borderColor: "rgba(255, 255, 255, 0.1)",
        autoScale: true,
      },
      timeScale: {
        borderColor: "rgba(255, 255, 255, 0.1)",
        timeVisible: true,
        secondsVisible: false,
        fixLeftEdge: true,
        fixRightEdge: true,
      },
      autoSize: true,
    });
    chartRef.current = chart;

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#10b981",
      downColor: "#f43f5e",
      borderVisible: false,
      wickUpColor: "#10b981",
      wickDownColor: "#f43f5e",
      priceFormat: { type: "price", precision: 2, minMove: 0.01 },
    });

    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: "#26a69a",
      priceFormat: { type: "volume" },
      priceScaleId: "",
    });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    const rsiSeries = chart.addSeries(LineSeries, {
      color: "#d946ef",
      lineWidth: 2,
      priceScaleId: "rsi",
    });
    chart.priceScale("rsi").applyOptions({
      scaleMargins: { top: 0.0, bottom: 0.85 },
      textColor: "#d946ef",
    });
    rsiSeries.createPriceLine({
      price: 70,
      color: "rgba(217, 70, 239, 0.3)",
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: false,
    });
    rsiSeries.createPriceLine({
      price: 30,
      color: "rgba(217, 70, 239, 0.3)",
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: false,
    });

    const candleData: CandleData[] = bars.map((bar) => ({
      time: toChartTime(bar.timestamp),
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    }));
    const volumeData: VolumeData[] = bars.map((bar) => ({
      time: toChartTime(bar.timestamp),
      value: bar.volume,
      color:
        bar.close >= bar.open
          ? "rgba(16, 185, 129, 0.4)"
          : "rgba(244, 63, 94, 0.4)",
    }));
    const rsiData = computeRsi(bars);

    candleSeries.setData(candleData);
    volumeSeries.setData(volumeData);
    rsiSeries.setData(rsiData);

    const markers: SeriesMarker<Time>[] = signals.map((signal) => {
      const isBuy =
        signal.direction.toUpperCase() === "BUY"
        || signal.direction.toUpperCase() === "LONG";
      return {
        time: toChartTime(signal.timestamp),
        position: isBuy ? "belowBar" : "aboveBar",
        color: isBuy ? "#06b6d4" : "#f59e0b",
        shape: isBuy ? "arrowUp" : "arrowDown",
        text: isBuy ? "Signal Long" : "Signal Short",
        size: 1,
      };
    });
    createSeriesMarkers(candleSeries, markers);

    chart.timeScale().fitContent();

    const handleResize = () => {
      chart.applyOptions({ width: chartContainerRef.current?.clientWidth });
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [bars, height, loading, signals]);

  return (
    <div className="relative h-full w-full">
      <div className="pointer-events-none absolute top-2 z-10 mx-4 flex items-center gap-4 text-[10px] font-mono">
        <span className="font-bold uppercase tracking-widest text-slate-100">
          {symbol}
        </span>
        <div className="flex items-center gap-1.5 opacity-60">
          <span className="text-emerald-500">Vol</span>
          <span className="text-fuchsia-500">RSI(14)</span>
        </div>
      </div>
      <div ref={chartContainerRef} className="h-full w-full" />
      {(loading || error || bars.length === 0) && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-950/40 text-xs font-medium text-slate-300 backdrop-blur-sm">
          {loading ? "Loading market bars..." : error ?? "No chart data available"}
        </div>
      )}
    </div>
  );
}
