"use client";

import HeatmapEmbed from "@/app/components/visualizations/tradingview-widgets/HeatmapEmbed";
import NewsFeedWidget from "@/app/components/visualizations/tradingview-widgets/NewsFeedWidget";
import TechnicalGaugeWidget from "@/app/components/visualizations/tradingview-widgets/TechnicalGaugeWidget";
import TradingViewWidget from "@/app/components/visualizations/tradingview-widgets/TradingViewWidget";
import Big8Panel from "@/components/big8/Big8Panel";
import ConfidenceBadge from "@/components/metrics/ConfidenceBadge";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

// Dynamic import for Lightweight Charts (client-side only)
const ZLFullScreenChart = dynamic(() => import("@/app/components/charts/ZLFullScreenChart"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-[#070a12]">
      <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
    </div>
  ),
});

type ZLRow = {
  as_of_date?: string | { value?: string };
  date?: string | { value?: string };
  close: number;
};

type ZLQuote = {
  success: boolean;
  price?: number;
  changePercent?: number | null;
  ts?: string;
  source?: string;
  warning?: string;
};

export default function ZLChart() {
  const [latestPrice, setLatestPrice] = useState<number | null>(null); // null = not yet loaded
  const [priceChange, setPriceChange] = useState<number | null>(null);
  const [confScore, setConfScore] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    async function fetchQuote() {
      try {
        const qRes = await fetch("/api/live/zl/quote", { cache: "no-store" });
        const qJson: unknown = await qRes.json();
        const q = qJson as ZLQuote;
        if (q?.success && typeof q.price === "number" && Number.isFinite(q.price) && q.price > 0) {
          setLatestPrice(q.price);
          if (typeof q.changePercent === "number" && Number.isFinite(q.changePercent)) {
            setPriceChange(q.changePercent);
          }
          setLastUpdate(new Date(q.ts || Date.now()));
        }
      } catch (error) {
        console.error("Error fetching ZL quote:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchQuote();
    const interval = setInterval(fetchQuote, 15000); // 15s quote refresh
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch("/api/live/zl");
        const json: unknown = await res.json();
        const j = json as { data?: unknown[]; error?: string };

        if (Array.isArray(j.data) && j.data.length > 0) {
          const rows = j.data as ZLRow[];

          // If quote API isn't available, fall back to last close from OHLCV table
          const latest = Number(rows[rows.length - 1].close);
          const prev = rows.length > 1 ? Number(rows[rows.length - 2].close) : latest;
          if (latestPrice === null && Number.isFinite(latest) && latest > 0) {
            setLatestPrice(latest);
            setPriceChange(prev > 0 ? ((latest - prev) / prev) * 100 : null);
            setLastUpdate(new Date());
          }
        }

        // Fetch training metrics as proxy for confidence (later wire real metric)
        const mRes = await fetch("/api/training/metrics");
        if (mRes.ok) {
          setConfScore(0.7);
        }
      } catch (error) {
        console.error("Error fetching ZL data:", error);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 300000); // 5 minutes
    return () => clearInterval(interval);
  }, [latestPrice]);

  return (
    <main className="min-h-screen w-screen bg-[#070a12] flex flex-col">
      {/* Background glow */}
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute -top-40 left-1/2 h-[700px] w-[1100px] -translate-x-1/2 rounded-full bg-gradient-to-b from-cyan-500/10 via-indigo-500/5 to-transparent blur-3xl" />
        <div className="absolute bottom-[-280px] right-[-220px] h-[700px] w-[700px] rounded-full bg-gradient-to-b from-emerald-500/8 via-emerald-500/0 to-transparent blur-3xl" />
      </div>

      {/* Header */}
      <div className="flex items-center justify-between px-10 py-6 border-b border-white/5">
        <div className="flex items-center gap-4">
          <div className="h-9 w-9 rounded-xl bg-white/5 border border-white/10" />
          <div>
            <h1 className="text-2xl md:text-3xl font-thin text-white tracking-wide">ZL Soybean Oil Futures</h1>
            <p className="text-zinc-500 text-xs font-light">
              Live • Databento • Quant Dashboard
              {lastUpdate ? ` • ${lastUpdate.toLocaleTimeString()}` : ""}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-5">
          <div className="hidden md:flex items-center gap-2 mr-2">
            <a
              href="/quant-fox"
              className="px-3 py-1.5 rounded-lg text-xs font-extralight bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors border border-white/10"
            >
              Quant Fox
            </a>
            <a
              href="/quant-admin"
              className="px-3 py-1.5 rounded-lg text-xs font-extralight bg-black/20 hover:bg-black/10 text-zinc-300 transition-colors border border-white/10"
            >
              Quant Admin
            </a>
          </div>
          {confScore !== null && <ConfidenceBadge value={confScore} />}
          {/* Always show TradingView widget for reliable live quote - our API is a bonus layer */}
          <div className="flex items-center gap-4">
            {latestPrice !== null && latestPrice > 0 ? (
              <div className="flex items-baseline gap-4">
                <span className="text-4xl md:text-6xl font-extralight text-white tabular-nums">
                  ${latestPrice.toFixed(2)}
                </span>
                <span
                  className={`text-lg md:text-2xl font-light tabular-nums ${
                    (priceChange ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"
                  }`}
                >
                  {(priceChange ?? 0) >= 0 ? "+" : ""}
                  {(priceChange ?? 0).toFixed(2)}%
                </span>
                <span className="text-xs font-medium text-emerald-400/90 bg-emerald-500/10 border border-emerald-500/20 px-2 py-1 rounded-full">
                  LIVE
                </span>
              </div>
            ) : (
              <div className="w-[360px] rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-md overflow-hidden">
                <div className="px-4 py-2 border-b border-white/10 flex items-center justify-between">
                  <div className="text-[11px] text-zinc-500 font-extralight">
                    {loading ? "Loading…" : "ZL Quote"}
                  </div>
                  <span className="text-[10px] font-medium text-emerald-400/90 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full">
                    LIVE
                  </span>
                </div>
                <div className="p-2">
                  <TradingViewWidget
                    scriptSrc="https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js"
                    config={{
                      symbol: "CBOT:ZL1!",
                      locale: "en",
                      colorTheme: "dark",
                      isTransparent: true,
                    }}
                    height={84}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Full-screen Lightweight Chart with real data from /api/live/zl */}
      <div className="flex-1 w-full" style={{ height: "75vh", minHeight: "500px" }}>
        <ZLFullScreenChart height="100%" />
      </div>

      {/* TradingView widget rail (fast, CDN-served embeds) */}
      <section className="px-10 py-8 border-t border-white/5 bg-[#070a12]">
        <div className="max-w-screen-2xl mx-auto">
          <div className="flex items-end justify-between mb-4">
            <div>
              <h2 className="text-white text-2xl font-thin tracking-wide">Market Widgets</h2>
              <div className="text-xs text-zinc-500 font-extralight">
                TradingView embeds for quick context (TA, news, macro heatmap)
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            {/* Single quote (top strip style) */}
            <div className="lg:col-span-3 rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-md overflow-hidden">
              <div className="px-4 py-3 border-b border-white/10">
                <div className="text-xs text-zinc-400 font-extralight">Quote</div>
              </div>
              <div className="p-3">
                <TradingViewWidget
                  scriptSrc="https://s3.tradingview.com/external-embedding/embed-widget-single-quote.js"
                  config={{
                    symbol: "CBOT:ZL1!",
                    locale: "en",
                    isTransparent: true,
                  }}
                  height={170}
                />
              </div>
            </div>

            {/* Technical Analysis */}
            <div className="lg:col-span-3 rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-md overflow-hidden">
              <div className="px-4 py-3 border-b border-white/10">
                <div className="text-xs text-zinc-400 font-extralight">Technical Analysis</div>
              </div>
              <div className="p-3">
                <TechnicalGaugeWidget symbol="CBOT:ZL1!" height={220} displayMode="multiple" showIntervalTabs />
              </div>
            </div>

            {/* Top Stories */}
            <div className="lg:col-span-3 rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-md overflow-hidden">
              <div className="px-4 py-3 border-b border-white/10">
                <div className="text-xs text-zinc-400 font-extralight">Top Stories</div>
              </div>
              <div className="p-3">
                <NewsFeedWidget height={260} feedMode="all_symbols" />
              </div>
            </div>

            {/* Forex Heatmap */}
            <div className="lg:col-span-3 rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-md overflow-hidden">
              <div className="px-4 py-3 border-b border-white/10">
                <div className="text-xs text-zinc-400 font-extralight">FX Heatmap</div>
              </div>
              <div className="p-3">
                <HeatmapEmbed height={260} market="forex" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {lastUpdate && (
        <div className="px-10 py-3 text-right border-t border-white/5">
          <span className="text-xs text-zinc-600">Last updated: {lastUpdate.toLocaleTimeString()}</span>
        </div>
      )}

      {/* Big 8 panel (scroll below the fold) */}
      <Big8Panel />
    </main>
  );
}
