import { useMemo, useState } from "react";
import { ChevronDown, ChevronRight, TrendingDown, TrendingUp } from "lucide-react";
import { SECTOR_MAP, useMarketData, type StockQuote } from "@/lib/marketData";

type SortKey = "symbol" | "price" | "changePercent" | "volume";
type SortDir = "asc" | "desc";

type SortHeaderProps = {
  label: string;
  field: SortKey;
  className?: string;
  sortKey: SortKey;
  sortDir: SortDir;
  onSort: (field: SortKey) => void;
};

const SECTOR_COLOR_BY_KEY: Record<string, string> = Object.fromEntries(
  Object.entries(SECTOR_MAP).map(([key, value]) => [key, value.color]),
);

function SortHeader({ label, field, className = "", sortKey, sortDir, onSort }: SortHeaderProps) {
  return (
    <th
      className={`cursor-pointer select-none py-2 text-[9px] font-bold uppercase tracking-[0.15em] text-slate-500 transition-colors hover:text-cyan-400 ${className}`}
      onClick={() => onSort(field)}
    >
      {label}
      {sortKey === field && <span className="ml-0.5 text-cyan-400">{sortDir === "asc" ? "^" : "v"}</span>}
    </th>
  );
}

function QuoteRow({ quote }: { quote: StockQuote }) {
  const isUp = quote.change >= 0;
  const sectorColor = SECTOR_COLOR_BY_KEY[quote.sector] ?? "#94a3b8";

  return (
    <tr className="group border-b border-white/[0.04] transition-colors hover:bg-white/[0.03]">
      <td className="py-2 pl-3 pr-2">
        <div className="flex items-center gap-2">
          <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ backgroundColor: sectorColor }} />
          <span className="text-xs font-bold tracking-wide text-slate-100">{quote.symbol}</span>
        </div>
      </td>
      <td className="py-2 text-right font-mono text-xs font-semibold text-slate-100">
        ${quote.price.toFixed(2)}
      </td>
      <td className={`py-2 text-right font-mono text-xs font-semibold ${isUp ? "text-emerald-400" : "text-rose-400"}`}>
        {isUp ? "+" : ""}{quote.change.toFixed(2)}
      </td>
      <td className="py-2 pr-2 text-right">
        <span
          className={`inline-flex items-center gap-0.5 rounded-md px-1.5 py-0.5 font-mono text-[10px] font-bold ${
            isUp ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"
          }`}
        >
          {isUp ? <TrendingUp size={9} /> : <TrendingDown size={9} />}
          {isUp ? "+" : ""}{quote.changePercent.toFixed(2)}%
        </span>
      </td>
      <td className="hidden py-2 pr-3 text-right font-mono text-[10px] text-slate-500 xl:table-cell">
        {quote.volume > 0 ? `${(quote.volume / 1e6).toFixed(1)}M` : "—"}
      </td>
      <td className="hidden py-2 pr-3 text-right font-mono text-[10px] text-slate-500 xl:table-cell">
        {quote.high > 0 ? quote.high.toFixed(2) : "—"}
      </td>
      <td className="hidden py-2 pr-3 text-right font-mono text-[10px] text-slate-500 xl:table-cell">
        {quote.low > 0 ? quote.low.toFixed(2) : "—"}
      </td>
    </tr>
  );
}

export default function WatchlistWidget() {
  const stockQuotes = useMarketData((state) => state.stockQuotes);
  const lastUpdate = useMarketData((state) => state.lastUpdate);

  const [sortKey, setSortKey] = useState<SortKey>("symbol");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [collapsedSectors, setCollapsedSectors] = useState<Set<string>>(new Set());
  const [viewMode, setViewMode] = useState<"sector" | "flat">("sector");

  const quotes = useMemo(() => Array.from(stockQuotes.values()), [stockQuotes]);
  const loaded = quotes.length;
  const total = 46;

  const sortedQuotes = useMemo(() => {
    const sorted = [...quotes].sort((a, b) => {
      const mult = sortDir === "asc" ? 1 : -1;
      if (sortKey === "symbol") return a.symbol.localeCompare(b.symbol) * mult;
      return ((a[sortKey] ?? 0) - (b[sortKey] ?? 0)) * mult;
    });
    return sorted;
  }, [quotes, sortKey, sortDir]);

  const sectorGroups = useMemo(() => {
    const groups: Record<string, { data: (typeof SECTOR_MAP)[string]; quotes: StockQuote[] }> = {};
    const quotesBySymbol = new Map(sortedQuotes.map((quote) => [quote.symbol, quote]));

    for (const [key, data] of Object.entries(SECTOR_MAP)) {
      const sectorQuotes = data.symbols
        .map((symbol) => quotesBySymbol.get(symbol))
        .filter((quote): quote is StockQuote => Boolean(quote));

      if (sectorQuotes.length > 0) {
        groups[key] = { data, quotes: sectorQuotes };
      }
    }

    return groups;
  }, [sortedQuotes]);

  const gainers = useMemo(() => quotes.filter((quote) => quote.changePercent > 0).length, [quotes]);
  const losers = useMemo(() => quotes.filter((quote) => quote.changePercent < 0).length, [quotes]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((dir) => (dir === "asc" ? "desc" : "asc"));
      return;
    }

    setSortKey(key);
    setSortDir(key === "symbol" ? "asc" : "desc");
  }

  function toggleSector(sector: string) {
    setCollapsedSectors((prev) => {
      const next = new Set(prev);
      if (next.has(sector)) next.delete(sector);
      else next.add(sector);
      return next;
    });
  }

  if (loaded === 0) {
    return (
      <div className="space-y-1">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="h-8 animate-shimmer rounded-lg border border-white/[0.04] bg-white/[0.02]" />
        ))}
        <p className="pt-2 text-center text-[9px] text-slate-500">Connecting to Finnhub... fetching 46 symbols</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="inline-block h-2 w-2 rounded-full bg-emerald-400 pulse-dot shadow-[0_0_6px_rgba(16,185,129,0.5)]" />
          <span className="text-[9px] font-bold uppercase tracking-[0.2em] text-emerald-400/80">LIVE · {loaded}/{total}</span>
          <span className="text-[9px] text-slate-600">{lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : ""}</span>
        </div>

        <div className="flex items-center gap-2">
          <span className="font-mono text-[10px] text-emerald-400">^ {gainers}</span>
          <span className="font-mono text-[10px] text-rose-400">¡ {losers}</span>
          <button
            onClick={() => setViewMode((mode) => (mode === "sector" ? "flat" : "sector"))}
            className="rounded-md border border-white/[0.08] bg-white/[0.03] px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-slate-400 transition-colors hover:border-cyan-500/30 hover:text-cyan-400"
          >
            {viewMode === "sector" ? "Flat" : "Sector"}
          </button>
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-white/[0.02]">
        <table className="w-full text-left">
          <thead className="border-b border-white/[0.08] bg-white/[0.02]">
            <tr>
              <SortHeader label="Symbol" field="symbol" className="pl-3" sortKey={sortKey} sortDir={sortDir} onSort={toggleSort} />
              <SortHeader label="Price" field="price" className="text-right" sortKey={sortKey} sortDir={sortDir} onSort={toggleSort} />
              <th className="py-2 text-right text-[9px] font-bold uppercase tracking-[0.15em] text-slate-500">Chg</th>
              <SortHeader label="Chg %" field="changePercent" className="text-right pr-2" sortKey={sortKey} sortDir={sortDir} onSort={toggleSort} />
              <SortHeader label="Vol" field="volume" className="hidden text-right pr-3 xl:table-cell" sortKey={sortKey} sortDir={sortDir} onSort={toggleSort} />
              <th className="hidden py-2 pr-3 text-right text-[9px] font-bold uppercase tracking-[0.15em] text-slate-500 xl:table-cell">High</th>
              <th className="hidden py-2 pr-3 text-right text-[9px] font-bold uppercase tracking-[0.15em] text-slate-500 xl:table-cell">Low</th>
            </tr>
          </thead>
          <tbody>
            {viewMode === "sector"
              ? Object.entries(sectorGroups).map(([key, group]) => {
                  const collapsed = collapsedSectors.has(key);
                  const avgChange = group.quotes.reduce((sum, quote) => sum + quote.changePercent, 0) / group.quotes.length;
                  const isUp = avgChange >= 0;

                  return (
                    <tr key={key} className="contents">
                      <td colSpan={7}>
                        <button
                          onClick={() => toggleSector(key)}
                          className="flex w-full items-center gap-2 border-b border-white/[0.06] bg-white/[0.02] px-3 py-1.5 text-left transition-colors hover:bg-white/[0.04]"
                        >
                          {collapsed ? <ChevronRight size={12} className="text-slate-500" /> : <ChevronDown size={12} className="text-slate-500" />}
                          <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: group.data.color }} />
                          <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-300">{group.data.label}</span>
                          <span className="text-[9px] text-slate-500">({group.quotes.length})</span>
                          <span className={`ml-auto font-mono text-[10px] font-semibold ${isUp ? "text-emerald-400" : "text-rose-400"}`}>
                            {isUp ? "+" : ""}{avgChange.toFixed(2)}%
                          </span>
                        </button>

                        {!collapsed && (
                          <table className="w-full text-left">
                            <tbody>
                              {group.quotes.map((quote) => (
                                <QuoteRow key={quote.symbol} quote={quote} />
                              ))}
                            </tbody>
                          </table>
                        )}
                      </td>
                    </tr>
                  );
                })
              : sortedQuotes.map((quote) => <QuoteRow key={quote.symbol} quote={quote} />)}
          </tbody>
        </table>
      </div>
    </div>
  );
}
