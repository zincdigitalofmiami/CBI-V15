"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";

const HeatmapEmbed = dynamic(
  () => import("@/app/components/visualizations/tradingview-widgets/HeatmapEmbed"),
  { ssr: false, loading: () => <WidgetLoader height={300} /> },
);

const TechnicalGaugeWidget = dynamic(
  () => import("@/app/components/visualizations/tradingview-widgets/TechnicalGaugeWidget"),
  { ssr: false, loading: () => <WidgetLoader height={300} /> },
);

const TradingViewWidget = dynamic(
  () => import("@/app/components/visualizations/tradingview-widgets/TradingViewWidget"),
  { ssr: false, loading: () => <WidgetLoader height={520} /> },
);

function WidgetLoader({ height = 300 }: { height?: number }) {
  return (
    <div
      style={{ height }}
      className="bg-zinc-900 rounded-lg border border-zinc-800 animate-pulse flex items-center justify-center"
    >
      <span className="text-zinc-600 text-sm font-extralight">Loading...</span>
    </div>
  );
}

export default function SentimentPage() {
  const [macroTab, setMacroTab] = useState<"global" | "growth" | "trade">("global");

  const macroSymbols = useMemo(() => {
    const global: [string, string][] = [
      ["FRED:UNRATE", "US Unemployment"],
      ["FRED:CPIAUCSL", "US CPI"],
      ["FRED:DGS10", "US 10Y"],
      ["FRED:DFF", "Fed Funds"],
      ["FRED:VIXCLS", "VIX"],
      ["FRED:DTWEXBGS", "USD Broad"],
    ];

    // “Growth” proxy panel using TradingView ECONOMICS symbols shown on the GDP growth heatmap list.
    // Source: https://www.tradingview.com/markets/world-economy/charts-growth/
    const growth: [string, string][] = [
      ["ECONOMICS:USGDPYY", "USA GDP YoY"],
      ["ECONOMICS:CNGDPYY", "China GDP YoY"],
      ["ECONOMICS:DEGDPYY", "Germany GDP YoY"],
      ["ECONOMICS:JPGDPYY", "Japan GDP YoY"],
      ["ECONOMICS:INGDPYY", "India GDP YoY"],
      ["ECONOMICS:GBGDPYY", "UK GDP YoY"],
      ["ECONOMICS:FRGDPYY", "France GDP YoY"],
      ["ECONOMICS:ITGDPYY", "Italy GDP YoY"],
      ["ECONOMICS:CAGDPYY", "Canada GDP YoY"],
      ["ECONOMICS:BRGDPYY", "Brazil GDP YoY"],
    ];

    // “Trade & capital” proxy panel (using series we already track + FX macro)
    const trade: [string, string][] = [
      ["FRED:DTWEXBGS", "USD Trade-Weighted"],
      ["FRED:DEXCHUS", "CNY/USD"],
      ["FRED:DEXBZUS", "BRL/USD"],
      ["FRED:DCOILWTICO", "WTI"],
      ["CBOT:ZL1!", "ZL"],
      ["NYMEX:CL1!", "CL"],
    ];

    return macroTab === "global" ? global : macroTab === "growth" ? growth : trade;
  }, [macroTab]);

  const buckets = [
    { id: "crush", name: "Crush", desc: "ZL/ZS/ZM spreads", color: "bg-amber-500" },
    { id: "china", name: "China", desc: "Demand proxy", color: "bg-red-500" },
    { id: "fx", name: "FX", desc: "Currency effects", color: "bg-blue-500" },
    { id: "fed", name: "Fed", desc: "Monetary policy", color: "bg-purple-500" },
    { id: "tariff", name: "Tariff", desc: "Trade policy", color: "bg-orange-500" },
    { id: "biofuel", name: "Biofuel", desc: "RIN/RFS", color: "bg-green-500" },
    { id: "energy", name: "Energy", desc: "Crude/HO/RB", color: "bg-yellow-500" },
    { id: "volatility", name: "Volatility", desc: "VIX/stress", color: "bg-pink-500" },
  ];

  const regimes = [
    { label: "Risk-On", color: "bg-green-600" },
    { label: "Risk-Off", color: "bg-red-600" },
    { label: "Range-Bound", color: "bg-zinc-600" },
    { label: "Trending", color: "bg-blue-600" },
  ];

  return (
    <main className="min-h-screen bg-[#070a12] px-10 py-10">
      <div className="max-w-screen-2xl mx-auto">
        {/* World economy hero (full width) */}
        <section className="mb-10 rounded-3xl border border-white/10 bg-white/[0.03] backdrop-blur-md overflow-hidden">
          <div className="px-6 py-5 border-b border-white/10 flex items-center justify-between gap-4">
            <div>
              <div className="text-white text-2xl font-thin tracking-wide">World Economy</div>
              <div className="text-xs text-zinc-500 font-extralight mt-1">
                Mirrors TradingView “Global trends” + “Trade & capital” views.
              </div>
            </div>

            <div className="flex items-center gap-2">
              <a
                href="https://www.tradingview.com/markets/world-economy/charts-global-trends/"
                target="_blank"
                rel="noreferrer"
                className="px-3 py-1.5 rounded-xl text-sm font-extralight border border-white/10 bg-black/20 text-zinc-300 hover:text-white hover:border-white/20 transition-colors"
              >
                Global trends
              </a>
              <a
                href="https://www.tradingview.com/markets/world-economy/charts-growth/"
                target="_blank"
                rel="noreferrer"
                className="px-3 py-1.5 rounded-xl text-sm font-extralight border border-white/10 bg-black/20 text-zinc-300 hover:text-white hover:border-white/20 transition-colors"
              >
                Growth
              </a>
              <a
                href="https://www.tradingview.com/markets/world-economy/charts-trade-and-capital/"
                target="_blank"
                rel="noreferrer"
                className="px-3 py-1.5 rounded-xl text-sm font-extralight border border-white/10 bg-black/20 text-zinc-300 hover:text-white hover:border-white/20 transition-colors"
              >
                Trade & capital
              </a>
              <div className="w-px h-6 bg-white/10 mx-1" />
              <button
                onClick={() => setMacroTab("global")}
                className={
                  "px-3 py-1.5 rounded-xl text-sm font-extralight transition-colors " +
                  (macroTab === "global" ? "bg-white/10 text-white" : "text-zinc-400 hover:text-zinc-200")
                }
              >
                Global
              </button>
              <button
                onClick={() => setMacroTab("growth")}
                className={
                  "px-3 py-1.5 rounded-xl text-sm font-extralight transition-colors " +
                  (macroTab === "growth" ? "bg-white/10 text-white" : "text-zinc-400 hover:text-zinc-200")
                }
              >
                Growth
              </button>
              <button
                onClick={() => setMacroTab("trade")}
                className={
                  "px-3 py-1.5 rounded-xl text-sm font-extralight transition-colors " +
                  (macroTab === "trade" ? "bg-white/10 text-white" : "text-zinc-400 hover:text-zinc-200")
                }
              >
                Trade & capital
              </button>
            </div>
          </div>

          <div className="p-4">
            <TradingViewWidget
              scriptSrc="https://s3.tradingview.com/external-embedding/embed-widget-symbol-overview.js"
              config={{
                symbols: macroSymbols,
                chartOnly: true,
                width: "100%",
                height: "100%",
                locale: "en",
                colorTheme: "dark",
                autosize: true,
                showVolume: false,
                showMA: false,
                hideDateRanges: false,
                hideMarketStatus: false,
                hideSymbolLogo: false,
                scalePosition: "right",
                scaleMode: "Normal",
                fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
                fontSize: "11",
                noTimeScale: false,
                valuesTracking: "1",
                changeMode: "price-and-percent",
                chartType: "area",
                lineWidth: 2,
              }}
              height={560}
            />
          </div>
        </section>

        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-3xl">🎯</span>
            <h1 className="text-3xl font-thin text-white tracking-wide">
              Sentiment & Regime Monitor
            </h1>
          </div>
          <p className="text-zinc-400 font-extralight">
            Big-8 bucket sentiment scores and market regime evolution
          </p>
        </header>

        <div className="grid lg:grid-cols-4 gap-6">
          {/* Big-8 Heatmap */}
          <section className="lg:col-span-4 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Big-8 Sentiment Heatmap</h2>
            <div className="grid grid-cols-4 md:grid-cols-8 gap-3">
              {buckets.map((bucket) => (
                <div
                  key={bucket.id}
                  className="bg-black border border-zinc-800 rounded-lg p-4 text-center hover:border-zinc-700 transition-colors"
                >
                  <div className={`w-3 h-3 ${bucket.color} rounded-full mx-auto mb-2`}></div>
                  <div className="text-white font-extralight text-sm">{bucket.name}</div>
                  <div className="text-3xl font-thin text-white my-2">—</div>
                  <div className="text-zinc-600 text-xs font-extralight">{bucket.desc}</div>
                </div>
              ))}
            </div>
            <div className="mt-4 flex items-center justify-center gap-6 text-xs text-zinc-500">
              <span className="flex items-center gap-1">
                <span className="w-3 h-1 bg-red-600 rounded"></span> Bearish
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-1 bg-zinc-600 rounded"></span> Neutral
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-1 bg-green-600 rounded"></span> Bullish
              </span>
            </div>
          </section>

          {/* Regime Timeline */}
          <section className="lg:col-span-3 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Regime Timeline</h2>
            <div className="h-16 bg-black border border-zinc-800 rounded-lg flex items-center px-4 gap-1 overflow-hidden">
              {/* Timeline bars placeholder */}
              {Array.from({ length: 30 }).map((_, i) => (
                <div
                  key={i}
                  className={`flex-1 h-8 ${regimes[i % 4].color} opacity-60 rounded-sm`}
                ></div>
              ))}
            </div>
            <div className="mt-4 flex items-center justify-between">
              <span className="text-zinc-600 text-xs font-extralight">30 days ago</span>
              <div className="flex items-center gap-4">
                {regimes.map((r) => (
                  <span key={r.label} className="flex items-center gap-1 text-xs text-zinc-500">
                    <span className={`w-2 h-2 ${r.color} rounded-sm`}></span>
                    {r.label}
                  </span>
                ))}
              </div>
              <span className="text-zinc-600 text-xs font-extralight">Today</span>
            </div>
          </section>

          {/* Current Regime */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Current Regime</h2>
            <div className="text-center">
              <div className="text-4xl font-thin text-white mb-2">—</div>
              <div className="text-zinc-500 text-sm font-extralight">Since: —</div>
              <div className="mt-4 space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-zinc-500">Confidence</span>
                  <span className="text-white">—%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-zinc-500">Duration</span>
                  <span className="text-white">— days</span>
                </div>
              </div>
            </div>
          </section>

          {/* Bucket Detail */}
          <section className="lg:col-span-2 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-thin text-white">Bucket Detail</h2>
              <select className="bg-black border border-zinc-800 rounded px-3 py-1 text-zinc-300 text-sm">
                {buckets.map((b) => (
                  <option key={b.id} value={b.id}>
                    {b.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="bg-black border border-zinc-800 rounded-lg p-3 text-center">
                <div className="text-zinc-500 text-xs font-extralight mb-1">Current Score</div>
                <div className="text-2xl font-thin text-white">—</div>
              </div>
              <div className="bg-black border border-zinc-800 rounded-lg p-3 text-center">
                <div className="text-zinc-500 text-xs font-extralight mb-1">7d Change</div>
                <div className="text-2xl font-thin text-zinc-400">—</div>
              </div>
            </div>
            <div className="h-32 bg-black border border-zinc-800 rounded-lg flex items-center justify-center">
              <span className="text-zinc-600 text-sm font-extralight">Time series chart</span>
            </div>
          </section>

          {/* Contributing Factors */}
          <section className="lg:col-span-2 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Contributing Factors</h2>
            <div className="space-y-3">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="flex items-center gap-3">
                  <div className="flex-1">
                    <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-zinc-600 to-zinc-500 rounded-full"
                        style={{ width: "0%" }}
                      ></div>
                    </div>
                  </div>
                  <span className="text-zinc-500 text-sm font-extralight w-20">—</span>
                  <span className="text-white text-sm font-extralight w-12 text-right">—%</span>
                </div>
              ))}
            </div>
          </section>

          {/* TradingView Widgets */}
          <section className="lg:col-span-2">
            <h2 className="text-lg font-thin text-white mb-3 flex items-center gap-2">
              <span className="text-orange-500">●</span> FX Heatmap
            </h2>
            <HeatmapEmbed market="forex" height={350} />
          </section>

          <section className="lg:col-span-2">
            <h2 className="text-lg font-thin text-white mb-3 flex items-center gap-2">
              <span className="text-purple-500">●</span> Technical Sentiment
            </h2>
            <TechnicalGaugeWidget symbol="CBOT:ZL1!" height={350} displayMode="multiple" />
          </section>
        </div>

        {/* Data Sources Footer */}
        <div className="mt-6 flex flex-wrap gap-4 text-xs text-zinc-600 font-extralight">
          <span>features.daily_ml_matrix_zl</span>
          <span>•</span>
          <span>reference.regime_calendar</span>
          <span>•</span>
          <span>staging.sentiment_buckets</span>
        </div>
      </div>
    </main>
  );
}
