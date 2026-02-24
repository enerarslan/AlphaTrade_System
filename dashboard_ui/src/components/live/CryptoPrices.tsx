import { TrendingUp, TrendingDown } from "lucide-react";
import { type CryptoPrice } from "@/lib/marketData";
import Sparkline from "@/components/ui/sparkline";

function CryptoCard({ coin }: { coin: CryptoPrice }) {
  const isUp = coin.price_change_percentage_24h >= 0;
  const spark = coin.sparkline_in_7d?.price?.slice(-48) ?? [];

  return (
    <div className="group relative overflow-hidden rounded-xl border border-white/[0.06] bg-white/[0.03] p-3 transition-all hover:border-cyan-500/15 hover:bg-white/[0.05]">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <img src={coin.image} alt={coin.name} className="h-6 w-6 rounded-full" />
          <div>
            <p className="text-xs font-semibold text-slate-200">{coin.symbol.toUpperCase()}</p>
            <p className="text-[9px] text-slate-500">{coin.name}</p>
          </div>
        </div>
        <div className="text-right">
          <p className="font-mono text-sm font-bold text-slate-100">
            ${coin.current_price >= 1 ? coin.current_price.toLocaleString(undefined, { maximumFractionDigits: 2 }) : coin.current_price.toFixed(6)}
          </p>
          <p className={`flex items-center justify-end gap-0.5 font-mono text-[10px] font-semibold ${isUp ? "text-emerald-400" : "text-rose-400"}`}>
            {isUp ? <TrendingUp size={10} /> : <TrendingDown size={10} />}
            {isUp ? "+" : ""}{coin.price_change_percentage_24h?.toFixed(2)}%
          </p>
        </div>
      </div>
      {spark.length > 2 && (
        <div className="mt-2">
          <Sparkline data={spark} width={160} height={32} color={isUp ? "#10b981" : "#f43f5e"} />
        </div>
      )}
      <div className="mt-2 flex justify-between text-[9px] text-slate-500">
        <span>MCap: ${(coin.market_cap / 1e9).toFixed(1)}B</span>
        <span>Vol: ${(coin.total_volume / 1e9).toFixed(2)}B</span>
      </div>
    </div>
  );
}

export default function CryptoPrices() {
  const cryptoPrices: CryptoPrice[] = [];

  if (cryptoPrices.length === 0) {
    return (
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-28 animate-shimmer rounded-xl border border-white/[0.06] bg-white/[0.02]" />
        ))}
      </div>
    );
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
      {cryptoPrices.slice(0, 5).map((coin) => (
        <CryptoCard key={coin.id} coin={coin} />
      ))}
    </div>
  );
}
