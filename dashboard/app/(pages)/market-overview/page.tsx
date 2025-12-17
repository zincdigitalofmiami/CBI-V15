"use client";

import dynamic from "next/dynamic";

// Dynamic imports for TradingView widgets (client-side only)
const SymbolOverviewCard = dynamic(
  () => import("@/app/components/visualizations/tradingview-widgets/SymbolOverviewCard"),
  { ssr: false, loading: () => <WidgetLoader /> },
);

const TechnicalGaugeWidget = dynamic(
  () => import("@/app/components/visualizations/tradingview-widgets/TechnicalGaugeWidget"),
  { ssr: false, loading: () => <WidgetLoader /> },
);

const HeatmapEmbed = dynamic(
  () => import("@/app/components/visualizations/tradingview-widgets/HeatmapEmbed"),
  { ssr: false, loading: () => <WidgetLoader /> },
);

function WidgetLoader() {
  return (
    <div className="h-[250px] bg-[#131722] rounded-lg border border-[#2a2f3e] animate-pulse flex items-center justify-center">
      <span className="text-gray-500 text-sm">Loading widget...</span>
    </div>
  );
}

// Mini chart symbols for the grid
const MINI_SYMBOLS = [
  { symbol: "CBOT:ZL1!", name: "Soybean Oil" },
  { symbol: "CBOT:ZS1!", name: "Soybeans" },
  { symbol: "CBOT:ZM1!", name: "Soybean Meal" },
  { symbol: "NYMEX:CL1!", name: "Crude Oil" },
  { symbol: "TVC:DXY", name: "Dollar Index" },
  { symbol: "FOREXCOM:USDBRL", name: "USD/BRL" },
];

export default function MarketOverviewPage() {
  return (
    <main className="min-h-screen bg-[#0a0e1a] p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-3xl">📊</span>
            <h1 className="text-3xl font-light text-white">Market Overview</h1>
          </div>
          <p className="text-gray-400">
            Technical dashboard — Live charts, gauges, and market heatmaps
          </p>
        </header>

        {/* Main Grid */}
        <div className="grid lg:grid-cols-4 gap-6">
          {/* Main Chart - Full width on mobile, 3 cols on desktop */}
          <section className="lg:col-span-3">
            <h2 className="text-lg font-light text-white mb-3 flex items-center gap-2">
              <span className="text-[#22c55e]">●</span> ZL Soybean Oil Overview
            </h2>
            <SymbolOverviewCard symbol="CBOT:ZL1!" symbolName="Soybean Oil" height={350} />
          </section>

          {/* Technical Gauge - Sidebar */}
          <section className="lg:col-span-1">
            <h2 className="text-lg font-light text-white mb-3 flex items-center gap-2">
              <span className="text-[#a855f7]">●</span> Technical Sentiment
            </h2>
            <TechnicalGaugeWidget symbol="CBOT:ZL1!" height={350} displayMode="single" />
          </section>
        </div>

        {/* Mini Charts Grid */}
        <section className="mt-8">
          <h2 className="text-lg font-light text-white mb-4 flex items-center gap-2">
            <span className="text-[#3b82f6]">●</span> Related Markets
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {MINI_SYMBOLS.map((item) => (
              <div
                key={item.symbol}
                className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-3"
              >
                <div className="text-xs text-gray-400 mb-2 truncate">{item.name}</div>
                <SymbolOverviewCard symbol={item.symbol} symbolName={item.name} height={120} />
              </div>
            ))}
          </div>
        </section>

        {/* Forex Heatmap */}
        <section className="mt-8">
          <h2 className="text-lg font-light text-white mb-4 flex items-center gap-2">
            <span className="text-[#f97316]">●</span> FX Heatmap
          </h2>
          <HeatmapEmbed market="forex" height={400} />
        </section>
      </div>
    </main>
  );
}
