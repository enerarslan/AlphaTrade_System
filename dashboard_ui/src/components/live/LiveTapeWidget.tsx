import { useEffect, useState, useRef } from "react";
import { ArrowUpRight, ArrowDownRight } from "lucide-react";

type TradeEvent = {
  id: string;
  time: Date;
  price: number;
  size: number;
  side: "BUY" | "SELL";
};

export default function LiveTapeWidget({ symbol = "AAPL" }: { symbol?: string }) {
  void symbol;
  const [trades, setTrades] = useState<TradeEvent[]>([]);
  const tapeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let basePrice = 150.25;

    const interval = setInterval(() => {
      // Simulate 1 to 3 trades hitting the tape every tick
      const numTrades = Math.floor(Math.random() * 3) + 1;
      const newTrades: TradeEvent[] = [];

      for (let i = 0; i < numTrades; i++) {
        const isBuy = Math.random() > 0.5;
        basePrice += (isBuy ? 1 : -1) * (Math.random() * 0.05);
        
        newTrades.push({
          id: Math.random().toString(36).substring(7),
          time: new Date(),
          price: basePrice,
          size: Math.floor(Math.random() * 500) + 10,
          side: isBuy ? "BUY" : "SELL"
        });
      }

      setTrades((prev) => [...newTrades, ...prev].slice(0, 50)); // Keep last 50
    }, 250); // fast tape

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex h-full flex-col overflow-hidden text-xs font-mono">
      <div className="flex items-center justify-between border-b border-white/[0.06] pb-2 text-[10px] uppercase text-slate-500">
        <span className="w-1/4 text-left">Time</span>
        <span className="w-1/4 text-center">Price</span>
        <span className="w-1/4 text-right">Size</span>
        <span className="w-1/4 text-right">Side</span>
      </div>
      
      <div 
        ref={tapeRef}
        className="flex-1 overflow-y-auto overflow-x-hidden pt-1 pr-1"
      >
        {trades.map((trade) => (
          <div 
            key={trade.id} 
            className="group flex items-center justify-between py-1 hover:bg-white/[0.04] transition-colors border-b border-white/[0.02]"
          >
            <span className="w-1/4 text-left text-slate-500">
              {trade.time.toLocaleTimeString(undefined, { hour12: false, fractionalSecondDigits: 1 })}
            </span>
            <span className={`w-1/4 text-center font-bold ${trade.side === "BUY" ? "text-emerald-400" : "text-rose-400"}`}>
              {trade.price.toFixed(2)}
            </span>
            <span className="w-1/4 text-right text-slate-300">
              {trade.size.toLocaleString()}
            </span>
            <span className="w-1/4 flex justify-end items-center gap-1">
              {trade.side === "BUY" ? (
                <>
                  <span className="text-[10px] text-emerald-500">BUY</span>
                  <ArrowUpRight size={12} className="text-emerald-500" />
                </>
              ) : (
                <>
                  <span className="text-[10px] text-rose-500">SELL</span>
                  <ArrowDownRight size={12} className="text-rose-500" />
                </>
              )}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
