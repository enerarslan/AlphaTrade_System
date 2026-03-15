import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";

type OrderBookRow = {
  price: number;
  size: number;
  total: number;
  depthImpact: number; // 0 to 1 for the background bar width
};

export default function OrderBookWidget({ symbol = "AAPL" }: { symbol?: string }) {
  void symbol;
  // We'll generate a highly dynamic simulated Level 2 Order Book
  const [bids, setBids] = useState<OrderBookRow[]>([]);
  const [asks, setAsks] = useState<OrderBookRow[]>([]);
  const [lastPrice, setLastPrice] = useState<number>(150.25);
  const [spread, setSpread] = useState<number>(0.05);

  useEffect(() => {
    // Simulated high-frequency updates (approx 10-15 updates per second)
    let basePrice = 150.25;
    
    const generateLevel2 = (center: number) => {
      const generatedBids: OrderBookRow[] = [];
      const generatedAsks: OrderBookRow[] = [];
      let currentBidTotal = 0;
      let currentAskTotal = 0;
      
      for (let i = 0; i < 15; i++) {
        // Asks (Sellers above center)
        const askPrice = center + (i * 0.05) + (Math.random() * 0.02);
        const askSize = Math.floor(Math.random() * 1500) + 100;
        currentAskTotal += askSize;
        generatedAsks.push({ price: askPrice, size: askSize, total: currentAskTotal, depthImpact: 0 });

        // Bids (Buyers below center)
        const bidPrice = center - (i * 0.05) - (Math.random() * 0.02);
        const bidSize = Math.floor(Math.random() * 1500) + 100;
        currentBidTotal += bidSize;
        generatedBids.push({ price: bidPrice, size: bidSize, total: currentBidTotal, depthImpact: 0 });
      }

      // Calculate depth impact based on max total
      const maxAskTotal = generatedAsks[generatedAsks.length - 1].total;
      generatedAsks.forEach(row => row.depthImpact = row.total / maxAskTotal);
      
      const maxBidTotal = generatedBids[generatedBids.length - 1].total;
      generatedBids.forEach(row => row.depthImpact = row.total / maxBidTotal);

      // Asks should be ordered descending (highest price at top)
      return { 
        newBids: generatedBids, 
        newAsks: generatedAsks.reverse() 
      };
    };

    const interval = setInterval(() => {
      // Random walk the base price
      basePrice += (Math.random() - 0.5) * 0.15;
      
      const { newBids, newAsks } = generateLevel2(basePrice);
      
      setLastPrice(basePrice);
      setSpread(newAsks[newAsks.length - 1].price - newBids[0].price);
      setBids(newBids);
      setAsks(newAsks);
    }, 120); // Extremely fast update interval for that high-frequency feel

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex h-full flex-col overflow-hidden text-xs font-mono">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/[0.06] pb-2 text-[10px] uppercase text-slate-500">
        <span className="w-1/3 text-left">Size</span>
        <span className="w-1/3 text-center">Price</span>
        <span className="w-1/3 text-right">Total</span>
      </div>

      {/* Asks (Red) */}
      <div className="flex-1 overflow-hidden" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-end' }}>
        {asks.map((ask, i) => (
          <div key={`ask-${i}`} className="group relative flex cursor-pointer items-center justify-between py-[2px] hover:bg-white/[0.04]">
            <div 
               className="absolute right-0 top-0 bottom-0 bg-rose-500/10 transition-all duration-75" 
               style={{ width: `${ask.depthImpact * 100}%` }} 
            />
            <span className="z-10 w-1/3 text-left pl-1 text-slate-300">{ask.size.toLocaleString()}</span>
            <span className="z-10 w-1/3 text-center font-bold text-rose-400">{ask.price.toFixed(2)}</span>
            <span className="z-10 w-1/3 text-right pr-1 text-slate-500">{ask.total.toLocaleString()}</span>
          </div>
        ))}
      </div>

      {/* Spread / Inside Market */}
      <div className="my-1 flex items-center justify-between border-y border-white/[0.06] bg-white/[0.02] py-2 px-2">
        <span className="text-slate-400 text-[10px] uppercase tracking-wider">Spread</span>
        <div className="flex items-center gap-3">
           <span className="text-sm font-bold text-slate-100">{lastPrice.toFixed(2)}</span>
           <Badge variant="outline" className="border-cyan-500/30 bg-cyan-500/10 text-cyan-400 font-mono text-[10px]">
             {spread.toFixed(2)}
           </Badge>
        </div>
      </div>

      {/* Bids (Green) */}
      <div className="flex-1 overflow-hidden">
        {bids.map((bid, i) => (
          <div key={`bid-${i}`} className="group relative flex cursor-pointer items-center justify-between py-[2px] hover:bg-white/[0.04]">
            <div 
               className="absolute right-0 top-0 bottom-0 bg-emerald-500/10 transition-all duration-75" 
               style={{ width: `${bid.depthImpact * 100}%` }} 
            />
            <span className="z-10 w-1/3 text-left pl-1 text-slate-300">{bid.size.toLocaleString()}</span>
            <span className="z-10 w-1/3 text-center font-bold text-emerald-400">{bid.price.toFixed(2)}</span>
            <span className="z-10 w-1/3 text-right pr-1 text-slate-500">{bid.total.toLocaleString()}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
