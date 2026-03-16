import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";

type BlockTradeRow = {
  id: string;
  symbol: string;
  timestamp: string;
  price: number;
  size: number;
  notional: number | null;
  venue: string | null;
  source: string;
  flags: string[];
};

type BlockTradesResponse = {
  generated_at: string;
  count: number;
  data: BlockTradeRow[];
};

type EnrichedBlockTrade = BlockTradeRow & {
  side: "BUY" | "SELL" | "CROSS";
  icebergDetected: boolean;
};

function formatSize(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return value.toString();
}

export default function DarkPoolWidget() {
  const [prints, setPrints] = useState<BlockTradeRow[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const response = await api.get<BlockTradesResponse>("/market/block-trades", {
          params: {
            symbols: "AAPL,MSFT,NVDA,GOOGL,AMZN,TSLA,META,JPM",
            limit: 50,
            min_size: 1000,
          },
        });
        if (!cancelled) {
          setPrints(response.data.data ?? []);
        }
      } catch {
        if (!cancelled) {
          setPrints([]);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    setLoading(true);
    void load();
    const timer = setInterval(() => {
      if (typeof document !== "undefined" && document.visibilityState !== "visible") {
        return;
      }
      void load();
    }, 5000);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const enrichedPrints = useMemo<EnrichedBlockTrade[]>(() => {
    const frequency = new Map<string, number>();
    prints.forEach((print) => {
      const key = `${print.symbol}:${print.price.toFixed(2)}`;
      frequency.set(key, (frequency.get(key) ?? 0) + 1);
    });

    let previousPrice: number | null = null;
    return prints.map((print) => {
      const key = `${print.symbol}:${print.price.toFixed(2)}`;
      const side: "BUY" | "SELL" | "CROSS" =
        previousPrice == null
          ? "CROSS"
          : print.price > previousPrice
            ? "BUY"
            : print.price < previousPrice
              ? "SELL"
              : "CROSS";
      previousPrice = print.price;
      return {
        ...print,
        side,
        icebergDetected: (frequency.get(key) ?? 0) >= 3,
      };
    });
  }, [prints]);

  const stats = useMemo(() => {
    const totalVolume = enrichedPrints.reduce((sum, print) => sum + print.size, 0);
    const darkVolume = enrichedPrints
      .filter((print) => (print.venue ?? "").toUpperCase().includes("DARK"))
      .reduce((sum, print) => sum + print.size, 0);
    const blockCount = enrichedPrints.length;
    const icebergs = enrichedPrints.filter((print) => print.icebergDetected).length;
    return {
      totalVolume,
      darkRatio: totalVolume > 0 ? (darkVolume / totalVolume) * 100 : 0,
      blockCount,
      icebergs,
    };
  }, [enrichedPrints]);

  const sideColor = (side: string) => {
    if (side === "BUY") return "text-emerald-400";
    if (side === "SELL") return "text-rose-400";
    return "text-amber-400";
  };

  const venueColor = (venue: string | null) => {
    const normalized = (venue ?? "TAPE").toUpperCase();
    if (normalized.includes("DARK")) {
      return "bg-purple-500/20 text-purple-300 border-purple-500/30";
    }
    if (normalized.includes("MID")) {
      return "bg-cyan-500/20 text-cyan-300 border-cyan-500/30";
    }
    return "bg-slate-500/20 text-slate-300 border-slate-500/30";
  };

  if (loading && enrichedPrints.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-slate-500">
        Loading institutional flow...
      </div>
    );
  }

  if (enrichedPrints.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-slate-500">
        No large prints available.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="grid grid-cols-4 gap-2 border-b border-white/[0.06] p-3">
        {[
          {
            label: "Total Vol",
            value: formatSize(stats.totalVolume),
            color: "text-cyan-400",
          },
          {
            label: "Dark %",
            value: `${stats.darkRatio.toFixed(1)}%`,
            color: "text-purple-400",
          },
          {
            label: "Blocks",
            value: stats.blockCount.toString(),
            color: "text-amber-400",
          },
          {
            label: "Icebergs",
            value: stats.icebergs.toString(),
            color: stats.icebergs > 0 ? "text-rose-400" : "text-slate-500",
          },
        ].map((metric) => (
          <div key={metric.label} className="text-center">
            <p className="text-[8px] font-semibold uppercase tracking-widest text-slate-500">
              {metric.label}
            </p>
            <p className={`font-mono text-sm font-bold ${metric.color}`}>
              {metric.value}
            </p>
          </div>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <table className="w-full text-[11px] font-mono">
          <thead className="sticky top-0 bg-slate-950/95 backdrop-blur-sm">
            <tr className="border-b border-white/[0.06] text-[9px] uppercase tracking-wider text-slate-500">
              <th className="px-2 py-1.5 text-left font-semibold">Time</th>
              <th className="px-2 py-1.5 text-left font-semibold">Sym</th>
              <th className="px-2 py-1.5 text-right font-semibold">Price</th>
              <th className="px-2 py-1.5 text-right font-semibold">Size</th>
              <th className="px-2 py-1.5 text-center font-semibold">Side</th>
              <th className="px-2 py-1.5 text-center font-semibold">Venue</th>
              <th className="px-2 py-1.5 text-center font-semibold">Flags</th>
            </tr>
          </thead>
          <tbody>
            {enrichedPrints.map((print) => (
              <tr
                key={print.id}
                className="border-b border-white/[0.03] hover:bg-white/[0.03]"
              >
                <td className="px-2 py-1 text-slate-500">
                  {new Date(print.timestamp).toLocaleTimeString(undefined, {
                    hour12: false,
                    minute: "2-digit",
                    second: "2-digit",
                  })}
                </td>
                <td className="px-2 py-1 font-bold text-slate-200">{print.symbol}</td>
                <td className="px-2 py-1 text-right text-slate-300">
                  ${print.price.toFixed(2)}
                </td>
                <td className="px-2 py-1 text-right text-amber-300">
                  {formatSize(print.size)}
                </td>
                <td className={`px-2 py-1 text-center ${sideColor(print.side)}`}>
                  {print.side}
                </td>
                <td className="px-2 py-1 text-center">
                  <span
                    className={`inline-block rounded border px-1.5 py-0.5 text-[9px] ${venueColor(
                      print.venue,
                    )}`}
                  >
                    {(print.venue ?? print.source).toUpperCase()}
                  </span>
                </td>
                <td className="space-x-1 px-2 py-1 text-center">
                  <span className="inline-block rounded border border-amber-500/30 bg-amber-500/20 px-1 py-0.5 text-[8px] font-bold text-amber-300">
                    BLOCK
                  </span>
                  {print.icebergDetected && (
                    <span className="inline-block rounded border border-rose-500/30 bg-rose-500/20 px-1 py-0.5 text-[8px] font-bold text-rose-300">
                      ICE
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
