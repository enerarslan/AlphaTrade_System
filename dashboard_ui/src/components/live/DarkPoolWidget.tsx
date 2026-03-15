import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

type DarkPoolPrint = {
  id: string;
  time: string;
  symbol: string;
  price: number;
  size: number;
  venue: "DARK" | "LIT" | "MIDPOINT";
  side: "BUY" | "SELL" | "CROSS";
  isBlock: boolean;
  icebergDetected: boolean;
};

const VENUES = ["DARK", "LIT", "MIDPOINT"] as const;
const SIDES = ["BUY", "SELL", "CROSS"] as const;
const SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "JPM"];

function randomPrint(): DarkPoolPrint {
  const sym = SYMBOLS[Math.floor(Math.random() * SYMBOLS.length)];
  const size = Math.floor(Math.random() * 50000) + 500;
  const isBlock = size > 25000;
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    time: new Date().toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }),
    symbol: sym,
    price: +(150 + Math.random() * 100).toFixed(2),
    size,
    venue: VENUES[Math.floor(Math.random() * VENUES.length)],
    side: SIDES[Math.floor(Math.random() * SIDES.length)],
    isBlock,
    icebergDetected: isBlock && Math.random() > 0.6,
  };
}

function formatSize(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toString();
}

export default function DarkPoolWidget() {
  const [prints, setPrints] = useState<DarkPoolPrint[]>([]);
  const [stats, setStats] = useState({ totalVolume: 0, darkRatio: 0, blockCount: 0, icebergs: 0 });
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let totalVol = 0;
    let darkVol = 0;
    let blocks = 0;
    let icebergs = 0;

    const interval = setInterval(() => {
      const newPrint = randomPrint();
      totalVol += newPrint.size;
      if (newPrint.venue === "DARK") darkVol += newPrint.size;
      if (newPrint.isBlock) blocks++;
      if (newPrint.icebergDetected) icebergs++;

      setPrints(prev => [newPrint, ...prev].slice(0, 50));
      setStats({
        totalVolume: totalVol,
        darkRatio: totalVol > 0 ? (darkVol / totalVol) * 100 : 0,
        blockCount: blocks,
        icebergs,
      });
    }, 800 + Math.random() * 1200);

    return () => clearInterval(interval);
  }, []);

  const sideColor = (side: string) => {
    if (side === "BUY") return "text-emerald-400";
    if (side === "SELL") return "text-rose-400";
    return "text-amber-400";
  };

  const venueColor = (venue: string) => {
    if (venue === "DARK") return "bg-purple-500/20 text-purple-300 border-purple-500/30";
    if (venue === "MIDPOINT") return "bg-cyan-500/20 text-cyan-300 border-cyan-500/30";
    return "bg-slate-500/20 text-slate-300 border-slate-500/30";
  };

  return (
    <div className="flex flex-col h-full">
      {/* Stats Bar */}
      <div className="grid grid-cols-4 gap-2 p-3 border-b border-white/[0.06]">
        {[
          { label: "Total Vol", value: formatSize(stats.totalVolume), color: "text-cyan-400" },
          { label: "Dark %", value: `${stats.darkRatio.toFixed(1)}%`, color: "text-purple-400" },
          { label: "Blocks", value: stats.blockCount.toString(), color: "text-amber-400" },
          { label: "Icebergs", value: stats.icebergs.toString(), color: stats.icebergs > 0 ? "text-rose-400" : "text-slate-500" },
        ].map(s => (
          <div key={s.label} className="text-center">
            <p className="text-[8px] uppercase tracking-widest text-slate-500 font-semibold">{s.label}</p>
            <p className={`font-mono text-sm font-bold ${s.color}`}>{s.value}</p>
          </div>
        ))}
      </div>

      {/* Feed */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin">
        <table className="w-full text-[11px] font-mono">
          <thead className="sticky top-0 bg-slate-950/95 backdrop-blur-sm">
            <tr className="text-[9px] uppercase tracking-wider text-slate-500 border-b border-white/[0.06]">
              <th className="py-1.5 px-2 text-left font-semibold">Time</th>
              <th className="py-1.5 px-2 text-left font-semibold">Sym</th>
              <th className="py-1.5 px-2 text-right font-semibold">Price</th>
              <th className="py-1.5 px-2 text-right font-semibold">Size</th>
              <th className="py-1.5 px-2 text-center font-semibold">Side</th>
              <th className="py-1.5 px-2 text-center font-semibold">Venue</th>
              <th className="py-1.5 px-2 text-center font-semibold">Flags</th>
            </tr>
          </thead>
          <tbody>
            <AnimatePresence initial={false}>
              {prints.map(p => (
                <motion.tr
                  key={p.id}
                  initial={{ opacity: 0, backgroundColor: p.isBlock ? "rgba(217, 70, 239, 0.15)" : "rgba(6, 182, 212, 0.08)" }}
                  animate={{ opacity: 1, backgroundColor: "transparent" }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 1.5 }}
                  className={`border-b border-white/[0.03] hover:bg-white/[0.03] ${p.isBlock ? "font-semibold" : ""}`}
                >
                  <td className="py-1 px-2 text-slate-500">{p.time}</td>
                  <td className="py-1 px-2 text-slate-200 font-bold">{p.symbol}</td>
                  <td className="py-1 px-2 text-right text-slate-300">${p.price.toFixed(2)}</td>
                  <td className={`py-1 px-2 text-right ${p.isBlock ? "text-amber-300" : "text-slate-400"}`}>
                    {formatSize(p.size)}
                  </td>
                  <td className={`py-1 px-2 text-center ${sideColor(p.side)}`}>{p.side}</td>
                  <td className="py-1 px-2 text-center">
                    <span className={`inline-block rounded px-1.5 py-0.5 text-[9px] border ${venueColor(p.venue)}`}>
                      {p.venue}
                    </span>
                  </td>
                  <td className="py-1 px-2 text-center space-x-1">
                    {p.isBlock && (
                      <span className="inline-block rounded bg-amber-500/20 text-amber-300 border border-amber-500/30 px-1 py-0.5 text-[8px] font-bold">
                        BLOCK
                      </span>
                    )}
                    {p.icebergDetected && (
                      <span className="inline-block rounded bg-rose-500/20 text-rose-300 border border-rose-500/30 px-1 py-0.5 text-[8px] font-bold animate-pulse">
                        🧊 ICE
                      </span>
                    )}
                  </td>
                </motion.tr>
              ))}
            </AnimatePresence>
          </tbody>
        </table>
      </div>
    </div>
  );
}
