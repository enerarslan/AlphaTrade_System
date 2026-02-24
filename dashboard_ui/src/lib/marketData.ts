import { create } from "zustand";
import axios from "axios";

export interface CryptoPrice {
  id: string;
  symbol: string;
  name: string;
  current_price: number;
  price_change_percentage_24h: number;
  market_cap: number;
  total_volume: number;
  sparkline_in_7d: { price: number[] };
  image: string;
}

export interface FearGreedData {
  value: number;
  classification: string;
  timestamp: string;
}

export interface MarketNewsItem {
  category: string;
  datetime: number;
  headline: string;
  id: number;
  image: string;
  source: string;
  summary: string;
  url: string;
}

export interface StockQuote {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  sector: string;
}

export const SECTOR_MAP: Record<string, { symbols: string[]; color: string; label: string }> = {
  technology: { symbols: ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM"], color: "#06b6d4", label: "Technology" },
  financial: { symbols: ["JPM", "BAC", "GS", "MS", "V", "MA", "BLK", "C"], color: "#8b5cf6", label: "Financial" },
  healthcare: { symbols: ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY"], color: "#10b981", label: "Healthcare" },
  consumer: { symbols: ["WMT", "HD", "KO", "PEP", "MCD", "NKE", "SBUX"], color: "#f59e0b", label: "Consumer" },
  energy: { symbols: ["XOM", "CVX", "COP"], color: "#ef4444", label: "Energy" },
  industrial: { symbols: ["CAT", "BA", "UPS", "HON", "GE"], color: "#f97316", label: "Industrial" },
  communication: { symbols: ["DIS", "NFLX", "T", "VZ"], color: "#ec4899", label: "Communication" },
  etf: { symbols: ["SPY", "QQQ", "IWM", "TLT", "GLD"], color: "#94a3b8", label: "ETFs" },
};

export const ALL_SYMBOLS = Object.values(SECTOR_MAP).flatMap((s) => s.symbols);

export function getSectorForSymbol(symbol: string): string {
  for (const [sector, data] of Object.entries(SECTOR_MAP)) {
    if (data.symbols.includes(symbol)) return sector;
  }
  return "other";
}

const ALPHA_VANTAGE_KEY = import.meta.env.VITE_ALPHA_VANTAGE_API_KEY;
const FINNHUB_KEY = import.meta.env.VITE_FINNHUB_API_KEY;
const FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1";

const warnedMissingKeys = new Set<string>();
const warnedTransientIssues = new Set<string>();

const MARKET_REFRESH_MS = 12_000;
const NEWS_REFRESH_MS = 60_000;
const FEAR_GREED_REFRESH_MS = 45_000;
const QUOTE_BATCH_SIZE = 3;

let quoteFetchIndex = 0;
let marketFetchInFlight = false;
let marketRefreshInterval: ReturnType<typeof setInterval> | null = null;
let marketRefreshSubscribers = 0;
let nextNewsFetchAt = 0;
let nextFearGreedFetchAt = 0;

function warnOnce(key: string, message: string) {
  if (!warnedTransientIssues.has(key)) {
    warnedTransientIssues.add(key);
    console.warn(message);
  }
}

function hasApiKey(provider: "finnhub" | "alphaVantage"): boolean {
  const key = provider === "finnhub" ? FINNHUB_KEY : ALPHA_VANTAGE_KEY;
  if (key) return true;

  if (!warnedMissingKeys.has(provider)) {
    warnedMissingKeys.add(provider);
    console.warn(`[MarketData] Missing API key for ${provider}. Set it in dashboard_ui/.env`);
  }

  return false;
}

function hasQuoteChanged(prev: StockQuote | undefined, next: StockQuote): boolean {
  if (!prev) return true;
  return (
    prev.price !== next.price
    || prev.change !== next.change
    || prev.changePercent !== next.changePercent
    || prev.volume !== next.volume
    || prev.high !== next.high
    || prev.low !== next.low
  );
}

function hasNewsChanged(prev: MarketNewsItem[], next: MarketNewsItem[]): boolean {
  if (prev.length !== next.length) return true;
  for (let i = 0; i < prev.length; i += 1) {
    if (
      prev[i].id !== next[i].id
      || prev[i].datetime !== next[i].datetime
      || prev[i].headline !== next[i].headline
    ) {
      return true;
    }
  }
  return false;
}

export async function fetchFearGreed(): Promise<FearGreedData | null> {
  try {
    const { data } = await axios.get<{ data: Array<{ value: string; value_classification: string; timestamp: string }> }>(
      FEAR_GREED_URL,
      { timeout: 8000 },
    );
    const item = data.data?.[0];
    if (!item) return null;
    return {
      value: Number(item.value),
      classification: item.value_classification,
      timestamp: item.timestamp,
    };
  } catch {
    console.warn("[MarketData] Fear & Greed fetch failed");
    return null;
  }
}

export async function fetchMarketNews(): Promise<MarketNewsItem[]> {
  if (!hasApiKey("finnhub")) return [];

  try {
    const { data } = await axios.get<MarketNewsItem[]>(
      `https://finnhub.io/api/v1/news?category=general&token=${FINNHUB_KEY}`,
      { timeout: 8000 },
    );
    return (data ?? []).slice(0, 20);
  } catch {
    console.warn("[MarketData] Finnhub news fetch failed");
    return [];
  }
}

export async function fetchFinnhubQuote(symbol: string): Promise<StockQuote | null> {
  if (!hasApiKey("finnhub")) return null;

  try {
    const { data } = await axios.get<{ c: number; d: number; dp: number; h: number; l: number; v: number }>(
      `https://finnhub.io/api/v1/quote?symbol=${symbol}&token=${FINNHUB_KEY}`,
      { timeout: 5000 },
    );
    if (!data || data.c === 0) return null;
    return {
      symbol,
      price: data.c,
      change: data.d ?? 0,
      changePercent: data.dp ?? 0,
      volume: data.v ?? 0,
      high: data.h ?? 0,
      low: data.l ?? 0,
      sector: getSectorForSymbol(symbol),
    };
  } catch {
    return null;
  }
}

export async function fetchAlphaVantageQuote(symbol: string): Promise<StockQuote | null> {
  if (!hasApiKey("alphaVantage")) return null;

  try {
    const { data } = await axios.get<{ "Global Quote": Record<string, string> }>(
      `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${ALPHA_VANTAGE_KEY}`,
      { timeout: 8000 },
    );
    const q = data["Global Quote"];
    if (!q || !q["05. price"]) return null;
    return {
      symbol: q["01. symbol"] ?? symbol,
      price: parseFloat(q["05. price"]),
      change: parseFloat(q["09. change"]),
      changePercent: parseFloat(q["10. change percent"]?.replace("%", "") ?? "0"),
      volume: parseInt(q["06. volume"] ?? "0", 10),
      high: parseFloat(q["03. high"] ?? "0"),
      low: parseFloat(q["04. low"] ?? "0"),
      sector: getSectorForSymbol(symbol),
    };
  } catch {
    return null;
  }
}

interface MarketDataState {
  fearGreed: FearGreedData | null;
  marketNews: MarketNewsItem[];
  stockQuotes: Map<string, StockQuote>;
  lastUpdate: number;
  fetchAllMarketData: () => Promise<void>;
  startAutoRefresh: () => () => void;
}

function handleVisibilityChange() {
  if (typeof document === "undefined" || document.visibilityState !== "visible") return;
  void useMarketData.getState().fetchAllMarketData();
}

export const useMarketData = create<MarketDataState>((set, get) => ({
  fearGreed: null,
  marketNews: [],
  stockQuotes: new Map(),
  lastUpdate: 0,

  fetchAllMarketData: async () => {
    if (marketFetchInFlight) return;
    marketFetchInFlight = true;

    try {
      const now = Date.now();
      const shouldFetchFearGreed = now >= nextFearGreedFetchAt;
      const shouldFetchNews = now >= nextNewsFetchAt;
      const currentState = get();

      const [fgResult, newsResult] = await Promise.allSettled([
        shouldFetchFearGreed ? fetchFearGreed() : Promise.resolve(null),
        shouldFetchNews ? fetchMarketNews() : Promise.resolve<MarketNewsItem[] | null>(null),
      ]);

      const batch = ALL_SYMBOLS.slice(quoteFetchIndex, quoteFetchIndex + QUOTE_BATCH_SIZE);
      quoteFetchIndex = (quoteFetchIndex + QUOTE_BATCH_SIZE) >= ALL_SYMBOLS.length
        ? 0
        : quoteFetchIndex + QUOTE_BATCH_SIZE;

      const quoteResults = await Promise.allSettled(
        batch.map((sym) => fetchFinnhubQuote(sym)),
      );

      let quotesChanged = false;
      const updatedQuotes = new Map(currentState.stockQuotes);
      for (const result of quoteResults) {
        if (result.status !== "fulfilled" || !result.value) continue;
        const nextQuote = result.value;
        const prevQuote = updatedQuotes.get(nextQuote.symbol);
        if (hasQuoteChanged(prevQuote, nextQuote)) {
          updatedQuotes.set(nextQuote.symbol, nextQuote);
          quotesChanged = true;
        }
      }

      const patch: Partial<MarketDataState> = {};

      if (quotesChanged) {
        patch.stockQuotes = updatedQuotes;
      }

      if (shouldFetchFearGreed) {
        nextFearGreedFetchAt = now + FEAR_GREED_REFRESH_MS;
        if (fgResult.status === "fulfilled" && fgResult.value) {
          const prev = currentState.fearGreed;
          if (!prev || prev.value !== fgResult.value.value || prev.classification !== fgResult.value.classification) {
            patch.fearGreed = fgResult.value;
          }
        } else if (fgResult.status === "rejected") {
          warnOnce("fear-greed", "[MarketData] Fear & Greed stream degraded; retrying with backoff.");
        }
      }

      if (shouldFetchNews) {
        nextNewsFetchAt = now + NEWS_REFRESH_MS;
        if (newsResult.status === "fulfilled" && Array.isArray(newsResult.value) && hasNewsChanged(currentState.marketNews, newsResult.value)) {
          patch.marketNews = newsResult.value;
        } else if (newsResult.status === "rejected") {
          warnOnce("market-news", "[MarketData] News stream degraded; retrying with backoff.");
        }
      }

      if (Object.keys(patch).length > 0) {
        patch.lastUpdate = Date.now();
        set(patch);
      }
    } finally {
      marketFetchInFlight = false;
    }
  },

  startAutoRefresh: () => {
    marketRefreshSubscribers += 1;

    if (!marketRefreshInterval) {
      void get().fetchAllMarketData();
      marketRefreshInterval = setInterval(() => {
        if (typeof document !== "undefined" && document.visibilityState !== "visible") return;
        void get().fetchAllMarketData();
      }, MARKET_REFRESH_MS);

      if (typeof document !== "undefined") {
        document.addEventListener("visibilitychange", handleVisibilityChange);
      }
    }

    return () => {
      marketRefreshSubscribers = Math.max(0, marketRefreshSubscribers - 1);

      if (marketRefreshSubscribers === 0) {
        if (marketRefreshInterval) {
          clearInterval(marketRefreshInterval);
          marketRefreshInterval = null;
        }
        if (typeof document !== "undefined") {
          document.removeEventListener("visibilitychange", handleVisibilityChange);
        }
      }
    };
  },
}));
