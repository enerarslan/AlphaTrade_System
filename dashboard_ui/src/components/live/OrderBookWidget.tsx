import { useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";

type DepthLevel = {
  price: number;
  size: number;
  total: number;
};

type DepthResponse = {
  symbol: string;
  timestamp: string | null;
  source: string;
  mid_price: number | null;
  spread: number | null;
  bids: DepthLevel[];
  asks: DepthLevel[];
};

export default function OrderBookWidget({ symbol = "AAPL" }: { symbol?: string }) {
  const [depth, setDepth] = useState<DepthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const response = await api.get<DepthResponse>("/market/depth", {
          params: { symbol, levels: 15, quote_window: 120 },
        });
        if (!cancelled) {
          setDepth(response.data);
        }
      } catch {
        if (!cancelled) {
          setDepth(null);
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
    }, 2500);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [symbol]);

  const maxAskTotal = useMemo(
    () => Math.max(...(depth?.asks.map((row) => row.total) ?? [0]), 1),
    [depth],
  );
  const maxBidTotal = useMemo(
    () => Math.max(...(depth?.bids.map((row) => row.total) ?? [0]), 1),
    [depth],
  );

  if (loading && !depth) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-slate-500">
        Loading quote ladder...
      </div>
    );
  }

  if (!depth || (depth.bids.length === 0 && depth.asks.length === 0)) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-slate-500">
        No live quote depth available.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-hidden text-xs font-mono">
      <div className="flex items-center justify-between border-b border-white/[0.06] pb-2 text-[10px] uppercase text-slate-500">
        <span className="w-1/3 text-left">Size</span>
        <span className="w-1/3 text-center">Price</span>
        <span className="w-1/3 text-right">Total</span>
      </div>

      <div
        className="flex-1 overflow-hidden"
        style={{ display: "flex", flexDirection: "column", justifyContent: "flex-end" }}
      >
        {depth.asks.map((ask, index) => (
          <div
            key={`ask-${ask.price}-${index}`}
            className="group relative flex items-center justify-between py-[2px] hover:bg-white/[0.04]"
          >
            <div
              className="absolute right-0 top-0 bottom-0 bg-rose-500/10"
              style={{ width: `${(ask.total / maxAskTotal) * 100}%` }}
            />
            <span className="z-10 w-1/3 pl-1 text-left text-slate-300">
              {ask.size.toLocaleString()}
            </span>
            <span className="z-10 w-1/3 text-center font-bold text-rose-400">
              {ask.price.toFixed(2)}
            </span>
            <span className="z-10 w-1/3 pr-1 text-right text-slate-500">
              {ask.total.toLocaleString()}
            </span>
          </div>
        ))}
      </div>

      <div className="my-1 flex items-center justify-between border-y border-white/[0.06] bg-white/[0.02] px-2 py-2">
        <span className="text-[10px] uppercase tracking-wider text-slate-400">
          {depth.source}
        </span>
        <div className="flex items-center gap-3">
          <span className="text-sm font-bold text-slate-100">
            {depth.mid_price?.toFixed(2) ?? "--"}
          </span>
          <Badge
            variant="outline"
            className="border-cyan-500/30 bg-cyan-500/10 font-mono text-[10px] text-cyan-400"
          >
            {depth.spread != null ? depth.spread.toFixed(4) : "--"}
          </Badge>
        </div>
      </div>

      <div className="flex-1 overflow-hidden">
        {depth.bids.map((bid, index) => (
          <div
            key={`bid-${bid.price}-${index}`}
            className="group relative flex items-center justify-between py-[2px] hover:bg-white/[0.04]"
          >
            <div
              className="absolute right-0 top-0 bottom-0 bg-emerald-500/10"
              style={{ width: `${(bid.total / maxBidTotal) * 100}%` }}
            />
            <span className="z-10 w-1/3 pl-1 text-left text-slate-300">
              {bid.size.toLocaleString()}
            </span>
            <span className="z-10 w-1/3 text-center font-bold text-emerald-400">
              {bid.price.toFixed(2)}
            </span>
            <span className="z-10 w-1/3 pr-1 text-right text-slate-500">
              {bid.total.toLocaleString()}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
