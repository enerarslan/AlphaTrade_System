import { useMarketData, type MarketNewsItem } from "@/lib/marketData";
import { ExternalLink, Newspaper } from "lucide-react";

function NewsCard({ item }: { item: MarketNewsItem }) {
  const date = new Date(item.datetime * 1000);
  const ago = getTimeAgo(date);

  return (
    <a
      href={item.url}
      target="_blank"
      rel="noopener noreferrer"
      className="group flex gap-3 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3 transition-all hover:border-cyan-500/15 hover:bg-white/[0.04]"
    >
      {item.image && (
        <img
          src={item.image}
          alt=""
          className="h-14 w-20 rounded-lg object-cover opacity-70 transition-opacity group-hover:opacity-100"
          onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
        />
      )}
      <div className="flex-1 min-w-0">
        <p className="line-clamp-2 text-xs font-medium text-slate-200 group-hover:text-cyan-300 transition-colors">
          {item.headline}
        </p>
        <div className="mt-1 flex items-center gap-2 text-[9px] text-slate-500">
          <span className="font-semibold text-cyan-500/70">{item.source}</span>
          <span>•</span>
          <span>{ago}</span>
          <ExternalLink size={8} className="ml-auto opacity-0 group-hover:opacity-60 transition-opacity" />
        </div>
      </div>
    </a>
  );
}

function getTimeAgo(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function MarketNewsFeed({ limit = 8 }: { limit?: number }) {
  const marketNews = useMarketData((state) => state.marketNews);

  if (marketNews.length === 0) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="h-16 animate-shimmer rounded-xl border border-white/[0.06] bg-white/[0.02]" />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 mb-2">
        <Newspaper size={14} className="text-cyan-400" />
        <span className="text-[10px] font-semibold uppercase tracking-[0.15em] text-slate-500">Live Market News</span>
      </div>
      {marketNews.slice(0, limit).map((item) => (
        <NewsCard key={item.id} item={item} />
      ))}
    </div>
  );
}
