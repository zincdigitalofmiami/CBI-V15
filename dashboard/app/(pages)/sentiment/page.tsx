'use client';

import dynamic from 'next/dynamic';

// Dynamic imports for TradingView widgets (client-side only)
const HeatmapEmbed = dynamic(
  () => import('@/app/components/visualizations/tradingview-widgets/HeatmapEmbed'),
  { ssr: false, loading: () => <WidgetLoader height={300} /> }
);

const TechnicalGaugeWidget = dynamic(
  () => import('@/app/components/visualizations/tradingview-widgets/TechnicalGaugeWidget'),
  { ssr: false, loading: () => <WidgetLoader height={300} /> }
);

function WidgetLoader({ height = 300 }: { height?: number }) {
  return (
    <div
      style={{ height }}
      className="bg-[#131722] rounded-lg border border-[#2a2f3e] animate-pulse flex items-center justify-center"
    >
      <span className="text-gray-500 text-sm">Loading widget...</span>
    </div>
  );
}

export default function SentimentPage() {
  const buckets = ['crush', 'china', 'fx', 'fed', 'tariff', 'biofuel', 'energy', 'vol'];

  return (
    <main className="min-h-screen bg-[#0a0e1a] p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-light text-white mb-2">Sentiment & Regime Monitor</h1>
          <p className="text-gray-400">Big-8 bucket sentiment and regime evolution</p>
        </header>

        <div className="grid gap-6">
          {/* Big-8 Heatmap */}
          <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
            <h2 className="text-xl font-light text-white mb-4">Big-8 Sentiment Heatmap</h2>
            <div className="grid grid-cols-4 gap-2 mb-4">
              {buckets.map((bucket) => (
                <div
                  key={bucket}
                  className="bg-[#0a0e1a] border border-[#2a2f3e] p-3 rounded text-center"
                >
                  <div className="text-sm text-gray-400 capitalize">{bucket}</div>
                  <div className="text-lg text-white font-light">--</div>
                </div>
              ))}
            </div>
            <div className="text-sm text-gray-600">
              Data source: <code className="text-gray-400">features.daily_ml_matrix_zl_v15</code>
            </div>
          </section>

          {/* Regime Timeline */}
          <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
            <h2 className="text-xl font-light text-white mb-4">Regime Timeline</h2>
            <p className="text-gray-500">Horizontal strip showing regime labels by date</p>
            <div className="mt-4 text-sm text-gray-600">
              Data source: <code className="text-gray-400">reference.regime_calendar</code>
            </div>
          </section>

          {/* Bucket Detail */}
          <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
            <h2 className="text-xl font-light text-white mb-4">Bucket Detail</h2>
            <p className="text-gray-500">Time series, forecast contribution, related news</p>
            <div className="mt-4 text-sm text-gray-600">
              Data source: <code className="text-gray-400">staging.news_signals_daily</code>
            </div>
          </section>

          {/* TradingView Widgets Section */}
          <div className="grid lg:grid-cols-2 gap-6">
            {/* FX Heatmap */}
            <section>
              <h2 className="text-lg font-light text-white mb-3 flex items-center gap-2">
                <span className="text-[#f97316]">●</span> FX Heatmap
              </h2>
              <HeatmapEmbed market="forex" height={350} />
            </section>

            {/* Technical Sentiment */}
            <section>
              <h2 className="text-lg font-light text-white mb-3 flex items-center gap-2">
                <span className="text-[#a855f7]">●</span> Technical Sentiment
              </h2>
              <TechnicalGaugeWidget symbol="CBOT:ZL1!" height={350} displayMode="multiple" />
            </section>
          </div>
        </div>

        <div className="mt-8 p-4 bg-yellow-900/20 border border-yellow-700/30 rounded-lg">
          <p className="text-yellow-200 text-sm">
            ⚠️ <strong>Placeholder:</strong> Big-8 data requires implementation. TradingView widgets are live.
          </p>
        </div>
      </div>
    </main>
  );
}
