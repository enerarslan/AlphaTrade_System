import { useEffect, useRef } from "react";
import { createChart, CandlestickSeries, HistogramSeries, LineSeries, createSeriesMarkers, type IChartApi, type SeriesMarker, type Time } from "lightweight-charts";

type CandleData = { time: string; open: number; high: number; low: number; close: number };
type VolumeData = { time: string; value: number; color: string };
type RsiData = { time: string; value: number };
type MarkerData = { time: string; side: "BUY" | "SELL"; text: string };

function generateMockData(days: number) {
  const candles: CandleData[] = [];
  const volumes: VolumeData[] = [];
  const rsi: RsiData[] = [];
  const markers: MarkerData[] = [];
  
  let price = 150;
  
  // Use strictly sequential dates to avoid DST duplicates
  const startRaw = new Date();
  startRaw.setUTCHours(0, 0, 0, 0);
  const startTime = startRaw.getTime() - days * 86400000;
  
  // Calculate raw prices
  for (let i = 0; i <= days; i++) {
    const d = new Date(startTime + i * 86400000);
    const dateStr = d.toISOString().split('T')[0];
    
    const open = price + (Math.random() - 0.5) * 5;
    const close = open + (Math.random() - 0.5) * 6;
    const high = Math.max(open, close) + Math.random() * 2;
    const low = Math.min(open, close) - Math.random() * 2;
    price = close;
    
    candles.push({ time: dateStr, open, high, low, close });
    
    const isUp = close > open;
    const vol = Math.floor(Math.random() * 1000000) + 500000;
    volumes.push({
      time: dateStr, 
      value: vol, 
      color: isUp ? 'rgba(16, 185, 129, 0.4)' : 'rgba(244, 63, 94, 0.4)'
    });
    
    // Fake RSI oscillating 30-70 roughly
    const rsiVal = 50 + (Math.sin(i * 0.2) * 20) + (Math.random() - 0.5) * 10;
    rsi.push({ time: dateStr, value: Math.max(0, Math.min(100, rsiVal)) });
    
    // Add random algo markers
    if (i % 15 === 0) {
       markers.push({ 
         time: dateStr, 
         side: isUp ? "BUY" : "SELL", 
         text: isUp ? "Algo Long" : "Algo Short" 
       });
    }
  }
  
  return { candles, volumes, rsi, markers };
}

export default function AdvancedChartWidget({ height = 500, symbol = "AAPL" }: { height?: number, symbol?: string }) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      height,
      layout: {
        background: { color: "transparent" },
        textColor: "#94a3b8", // slate-400
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
        fixLeftEdge: true,
        fixRightEdge: true,
      },
      autoSize: true,
    });
    chartRef.current = chart;

    // 1. Candlestick Series (Main Pane)
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#10b981", // emerald-500
      downColor: "#f43f5e", // rose-500
      borderVisible: false,
      wickUpColor: "#10b981",
      wickDownColor: "#f43f5e",
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    });

    // 2. Volume Series (Overlaid on Main Pane at bottom)
    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: "#26a69a",
      priceFormat: { type: "volume" },
      priceScaleId: "", // Set as an overlay
    });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    // 3. RSI Series (Secondary Pane / Top)
    // Lightweight-charts doesn't natively do multi-pane easily without creating two charts. 
    // To keep it high-performance, we overlay RSI with a separate price scale.
    const rsiSeries = chart.addSeries(LineSeries, {
      color: "#d946ef", // fuchsia-500
      lineWidth: 2,
      priceScaleId: "rsi",
    });

    // Configure the second pane (RSI) at the top or bottom. We'll put RSI above volume.
    chart.priceScale("rsi").applyOptions({
      scaleMargins: { top: 0.0, bottom: 0.85 }, // Put RSI at the very top of the chart space
      textColor: "#d946ef",
    });
    
    // Add 30/70 RSI reference lines
    rsiSeries.createPriceLine({ price: 70, color: "rgba(217, 70, 239, 0.3)", lineWidth: 1, lineStyle: 2, axisLabelVisible: false });
    rsiSeries.createPriceLine({ price: 30, color: "rgba(217, 70, 239, 0.3)", lineWidth: 1, lineStyle: 2, axisLabelVisible: false });

    // Load Data
    const mockData = generateMockData(200);
    candleSeries.setData(mockData.candles);
    volumeSeries.setData(mockData.volumes);
    rsiSeries.setData(mockData.rsi);

    // Apply execution markers to Candlesticks using v5 plugin
    const lwMarkers: SeriesMarker<Time>[] = mockData.markers.map(m => ({
      time: m.time as Time,
      position: m.side === "BUY" ? "belowBar" : "aboveBar",
      color: m.side === "BUY" ? "#06b6d4" : "#f59e0b", // cyan/amber for algo markers
      shape: m.side === "BUY" ? "arrowUp" : "arrowDown",
      text: m.text,
      size: 1,
    }));
    
    createSeriesMarkers(candleSeries, lwMarkers);

    chart.timeScale().fitContent();

    const handleResize = () => {
      chart.applyOptions({ width: chartContainerRef.current?.clientWidth });
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [height]);

  return (
    <div className="relative h-full w-full">
       <div className="absolute top-2 flex items-center gap-4 text-[10px] font-mono pointer-events-none z-10 mx-4">
         <span className="font-bold text-slate-100 uppercase tracking-widest">{symbol}</span>
         <div className="flex items-center gap-1.5 opacity-60">
           <span className="text-emerald-500">Vol</span>
           <span className="text-fuchsia-500">RSI(14)</span>
         </div>
       </div>
       <div ref={chartContainerRef} className="h-full w-full" />
    </div>
  );
}
