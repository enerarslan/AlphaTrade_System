import { useEffect, useMemo, useState } from "react";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { api } from "@/lib/api";

type TradeRow = {
  trade_id: string | null;
  symbol: string;
  timestamp: string;
  price: number;
  size: number;
  exchange: string | null;
  tape: string | null;
  notional: number | null;
};

type TradesResponse = {
  symbol: string;
  source: string;
  count: number;
  data: TradeRow[];
};

type TradeEvent = TradeRow & {
  side: "BUY" | "SELL";
};

export default function LiveTapeWidget({ symbol = "AAPL" }: { symbol?: string }) {
  const [trades, setTrades] = useState<TradeRow[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const response = await api.get<TradesResponse>("/market/trades", {
          params: { symbol, limit: 60 },
        });
        if (!cancelled) {
          setTrades(response.data.data ?? []);
        }
      } catch {
        if (!cancelled) {
          setTrades([]);
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
    }, 1500);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [symbol]);

  const normalizedTrades = useMemo<TradeEvent[]>(() => {
    let previousPrice: number | null = null;
    return trades.map((trade) => {
      const side: "BUY" | "SELL" =
        previousPrice == null || trade.price >= previousPrice ? "BUY" : "SELL";
      previousPrice = trade.price;
      return { ...trade, side };
    });
  }, [trades]);

  if (loading && normalizedTrades.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-slate-500">
        Loading recent prints...
      </div>
    );
  }

  if (normalizedTrades.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-slate-500">
        No recent trade prints available.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-hidden text-xs font-mono">
      <div className="flex items-center justify-between border-b border-white/[0.06] pb-2 text-[10px] uppercase text-slate-500">
        <span className="w-1/4 text-left">Time</span>
        <span className="w-1/4 text-center">Price</span>
        <span className="w-1/4 text-right">Size</span>
        <span className="w-1/4 text-right">Side</span>
      </div>

      <div className="flex-1 overflow-y-auto overflow-x-hidden pt-1 pr-1">
        {normalizedTrades.map((trade, index) => (
          <div
            key={`${trade.trade_id ?? trade.timestamp}-${index}`}
            className="group flex items-center justify-between border-b border-white/[0.02] py-1 transition-colors hover:bg-white/[0.04]"
          >
            <span className="w-1/4 text-left text-slate-500">
              {new Date(trade.timestamp).toLocaleTimeString(undefined, {
                hour12: false,
                minute: "2-digit",
                second: "2-digit",
              })}
            </span>
            <span
              className={`w-1/4 text-center font-bold ${
                trade.side === "BUY" ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {trade.price.toFixed(2)}
            </span>
            <span className="w-1/4 text-right text-slate-300">
              {trade.size.toLocaleString()}
            </span>
            <span className="flex w-1/4 items-center justify-end gap-1">
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
