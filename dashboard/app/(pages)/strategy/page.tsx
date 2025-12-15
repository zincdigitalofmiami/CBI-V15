'use client';

export default function StrategyPage() {
  const horizons = [
    { label: '1W', days: 5, desc: 'Immediate needs' },
    { label: '1M', days: 20, desc: 'Near-term procurement' },
    { label: '3M', days: 60, desc: 'Quarterly planning' },
    { label: '6M', days: 126, desc: 'Strategic positioning' },
  ];

  const scenarios = [
    { id: 'china-soft', name: 'China Soft', impact: 'bearish', desc: 'Reduced China demand, export slowdown' },
    { id: 'tariffs-escalate', name: 'Tariffs Escalate', impact: 'bullish', desc: 'Section 301 expansion, retaliatory measures' },
    { id: 'weather-shock', name: 'Weather Shock', impact: 'bullish', desc: 'La Niña, drought conditions in key regions' },
    { id: 'fed-pivot', name: 'Fed Pivot', impact: 'mixed', desc: 'Rate cuts, dollar weakness' },
  ];

  return (
    <main className="min-h-screen bg-black p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-3xl">📈</span>
            <h1 className="text-3xl font-thin text-white tracking-wide">Strategy & Procurement Plan</h1>
          </div>
          <p className="text-zinc-400 font-extralight">
            Hedge ladder, scenario analysis, and P&L distribution for ZL procurement
          </p>
        </header>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Strategy Summary - Full Width */}
          <section className="lg:col-span-3 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-thin text-white">Strategy Summary</h2>
              <span className="px-3 py-1 bg-green-900/30 border border-green-700/50 text-green-400 text-sm rounded-full">
                BUY
              </span>
            </div>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="space-y-2">
                <div className="text-zinc-500 text-sm font-extralight">Recommendation</div>
                <div className="text-white text-lg font-extralight">—</div>
              </div>
              <div className="space-y-2">
                <div className="text-zinc-500 text-sm font-extralight">Confidence</div>
                <div className="text-white text-lg font-extralight">—%</div>
              </div>
              <div className="space-y-2">
                <div className="text-zinc-500 text-sm font-extralight">Last Updated</div>
                <div className="text-white text-lg font-extralight">—</div>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-zinc-800">
              <div className="text-zinc-500 text-sm font-extralight">Rationale</div>
              <p className="text-zinc-300 font-extralight mt-1">—</p>
            </div>
          </section>

          {/* Hedge Ladder */}
          <section className="lg:col-span-2 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Hedge Ladder</h2>
            <div className="grid grid-cols-4 gap-3">
              {horizons.map((h) => (
                <div
                  key={h.label}
                  className="bg-black border border-zinc-800 p-4 rounded-lg text-center hover:border-zinc-700 transition-colors"
                >
                  <div className="text-zinc-400 text-sm font-extralight mb-1">{h.label}</div>
                  <div className="text-3xl text-white font-thin mb-1">—</div>
                  <div className="text-zinc-600 text-xs font-extralight">lbs</div>
                  <div className="text-zinc-500 text-xs font-extralight mt-2">{h.desc}</div>
                </div>
              ))}
            </div>
            <div className="mt-4 flex items-center gap-4 text-xs text-zinc-600">
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 bg-green-600 rounded-sm"></span> Low urgency
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 bg-yellow-600 rounded-sm"></span> Medium
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 bg-red-600 rounded-sm"></span> High urgency
              </span>
            </div>
          </section>

          {/* Current Position */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Current Position</h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-zinc-400 font-extralight">Open Contracts</span>
                <span className="text-white font-extralight">—</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-zinc-400 font-extralight">Avg Entry Price</span>
                <span className="text-white font-extralight">$—</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-zinc-400 font-extralight">Current Value</span>
                <span className="text-white font-extralight">$—</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-zinc-400 font-extralight">Unrealized P&L</span>
                <span className="text-zinc-400 font-extralight">—</span>
              </div>
            </div>
          </section>

          {/* Scenario Analysis */}
          <section className="lg:col-span-2 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Scenario Analysis</h2>
            <div className="space-y-3">
              {scenarios.map((s) => (
                <div
                  key={s.id}
                  className="flex items-center gap-4 bg-black border border-zinc-800 p-4 rounded-lg hover:border-zinc-700 transition-colors"
                >
                  <input
                    type="checkbox"
                    className="w-4 h-4 bg-black border-zinc-600 rounded"
                  />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-white font-extralight">{s.name}</span>
                      <span
                        className={`text-xs px-2 py-0.5 rounded ${
                          s.impact === 'bullish'
                            ? 'bg-green-900/30 text-green-400'
                            : s.impact === 'bearish'
                            ? 'bg-red-900/30 text-red-400'
                            : 'bg-zinc-800 text-zinc-400'
                        }`}
                      >
                        {s.impact}
                      </span>
                    </div>
                    <div className="text-zinc-500 text-sm font-extralight">{s.desc}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-white font-extralight">—</div>
                    <div className="text-zinc-500 text-xs">impact</div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* P&L Distribution */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">P&L Distribution</h2>
            <div className="space-y-3">
              <div className="flex justify-between items-center text-sm">
                <span className="text-zinc-400 font-extralight">P10 (Bear)</span>
                <span className="text-red-400 font-extralight">$—</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-zinc-400 font-extralight">P25</span>
                <span className="text-zinc-300 font-extralight">$—</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-zinc-400 font-extralight">P50 (Expected)</span>
                <span className="text-white font-extralight text-lg">$—</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-zinc-400 font-extralight">P75</span>
                <span className="text-zinc-300 font-extralight">$—</span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-zinc-400 font-extralight">P90 (Bull)</span>
                <span className="text-green-400 font-extralight">$—</span>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-zinc-800">
              <div className="h-24 bg-black border border-zinc-800 rounded flex items-center justify-center">
                <span className="text-zinc-600 text-sm font-extralight">Distribution chart</span>
              </div>
            </div>
          </section>
        </div>

        {/* Data Sources Footer */}
        <div className="mt-6 flex flex-wrap gap-4 text-xs text-zinc-600 font-extralight">
          <span>forecasts.zl_predictions</span>
          <span>•</span>
          <span>training.ensemble_weights</span>
          <span>•</span>
          <span>forecasts.monte_carlo_scenarios</span>
        </div>
      </div>
    </main>
  );
}
