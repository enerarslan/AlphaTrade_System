import { useMemo } from "react";
import { SECTOR_MAP, useMarketData } from "@/lib/marketData";

export default function SectorHeatmap() {
  const stockQuotes = useMarketData((state) => state.stockQuotes);
  const quotes = useMemo(() => Array.from(stockQuotes.values()), [stockQuotes]);

  const sectors = useMemo(() => {
    const bySector = new Map<string, typeof quotes>();
    for (const quote of quotes) {
      const list = bySector.get(quote.sector) ?? [];
      list.push(quote);
      bySector.set(quote.sector, list);
    }

    return Object.entries(SECTOR_MAP).map(([key, data]) => {
      const sectorQuotes = bySector.get(key) ?? [];
      const fetched = sectorQuotes.length;
      const total = data.symbols.length;
      const avgChange = fetched > 0
        ? sectorQuotes.reduce((sum, quote) => sum + quote.changePercent, 0) / fetched
        : 0;
      const totalVolume = sectorQuotes.reduce((sum, quote) => sum + quote.volume, 0);
      const topMover = sectorQuotes.length > 0
        ? sectorQuotes.reduce((top, quote) => (Math.abs(quote.changePercent) > Math.abs(top.changePercent) ? quote : top), sectorQuotes[0])
        : undefined;

      return { key, ...data, avgChange, totalVolume, fetched, total, topMover };
    });
  }, [quotes]);

  if (quotes.length === 0) {
    return (
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="h-20 animate-shimmer rounded-xl border border-white/[0.04] bg-white/[0.02]" />
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      {sectors.map((sector) => {
        const isUp = sector.avgChange >= 0;
        const intensity = Math.min(Math.abs(sector.avgChange) * 14, 100);
        const bg = isUp
          ? `rgba(16, 185, 129, ${intensity / 300})`
          : `rgba(244, 63, 94, ${intensity / 300})`;

        return (
          <div
            key={sector.key}
            className="relative overflow-hidden rounded-xl border border-white/[0.06] p-3 transition-colors hover:border-white/[0.12]"
            style={{ backgroundColor: bg }}
          >
            <div
              className="absolute left-0 top-0 h-full w-[3px] rounded-l-xl"
              style={{ backgroundColor: sector.color }}
            />

            <div className="flex items-start justify-between">
              <div>
                <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-300">{sector.label}</p>
                <p className={`mt-1 font-mono text-lg font-bold ${isUp ? "text-emerald-400" : "text-rose-400"}`}>
                  {isUp ? "+" : ""}{sector.avgChange.toFixed(2)}%
                </p>
              </div>
              <span className="text-[9px] text-slate-500">{sector.fetched}/{sector.total}</span>
            </div>

            {sector.topMover && (
              <div className="mt-1.5 flex items-center justify-between">
                <span className="text-[9px] text-slate-500">Top:</span>
                <span className={`font-mono text-[10px] font-semibold ${sector.topMover.changePercent >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {sector.topMover.symbol} {sector.topMover.changePercent >= 0 ? "+" : ""}{sector.topMover.changePercent.toFixed(2)}%
                </span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
