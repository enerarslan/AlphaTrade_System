import { useEffect, useMemo } from "react";
import { TrendingDown, TrendingUp } from "lucide-react";
import { SECTOR_MAP, useMarketData } from "@/lib/marketData";

const SECTOR_COLOR_BY_KEY: Record<string, string> = Object.fromEntries(
  Object.entries(SECTOR_MAP).map(([key, value]) => [key, value.color]),
);

export default function MarketTicker() {
  const stockQuotes = useMarketData((state) => state.stockQuotes);
  const fearGreed = useMarketData((state) => state.fearGreed);
  const startAutoRefresh = useMarketData((state) => state.startAutoRefresh);

  useEffect(() => {
    const cleanup = startAutoRefresh();
    return cleanup;
  }, [startAutoRefresh]);

  const items = useMemo(() => {
    const quotes = Array.from(stockQuotes.values());
    if (quotes.length === 0) return [];

    return quotes
      .filter((q) => q.price > 0)
      .sort((a, b) => Math.abs(b.changePercent) - Math.abs(a.changePercent))
      .slice(0, 20)
      .map((q) => ({
        ...q,
        sectorColor: SECTOR_COLOR_BY_KEY[q.sector] ?? "#94a3b8",
      }));
  }, [stockQuotes]);

  if (items.length === 0 && !fearGreed) return null;

  const allItems = [...items, ...items];

  return (
    <div className="relative w-full overflow-hidden border-b border-white/[0.06] bg-slate-950/80">
      <div className="flex animate-signal-tape whitespace-nowrap py-1.5">
        {allItems.map((item, i) => (
          <div key={`${item.symbol}-${i}`} className="mr-5 flex items-center gap-1.5 px-1">
            <span
              className="inline-block h-1 w-1 rounded-full"
              style={{ backgroundColor: item.sectorColor }}
            />
            <span className="text-[10px] font-bold tracking-wider text-slate-400">{item.symbol}</span>
            <span className="font-mono text-xs font-semibold text-slate-200">
              ${item.price.toFixed(2)}
            </span>
            <span className={`flex items-center gap-0.5 font-mono text-[10px] font-semibold ${item.changePercent >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
              {item.changePercent >= 0 ? <TrendingUp size={9} /> : <TrendingDown size={9} />}
              {item.changePercent >= 0 ? "+" : ""}{item.changePercent.toFixed(2)}%
            </span>
            <span className="ml-1 text-slate-700/50">-</span>
          </div>
        ))}
        {fearGreed && (
          <>
            <div className="mr-5 flex items-center gap-1.5 px-1">
              <span className="text-[10px] font-bold tracking-wider text-amber-400">F&G</span>
              <span className={`font-mono text-xs font-semibold ${fearGreed.value >= 60 ? "text-emerald-400" : fearGreed.value >= 40 ? "text-amber-400" : "text-rose-400"}`}>
                {fearGreed.value}
              </span>
              <span className="text-[9px] text-slate-500">{fearGreed.classification}</span>
              <span className="ml-1 text-slate-700/50">-</span>
            </div>
            <div className="mr-5 flex items-center gap-1.5 px-1">
              <span className="text-[10px] font-bold tracking-wider text-amber-400">F&G</span>
              <span className={`font-mono text-xs font-semibold ${fearGreed.value >= 60 ? "text-emerald-400" : fearGreed.value >= 40 ? "text-amber-400" : "text-rose-400"}`}>
                {fearGreed.value}
              </span>
              <span className="text-[9px] text-slate-500">{fearGreed.classification}</span>
              <span className="ml-1 text-slate-700/50">-</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
