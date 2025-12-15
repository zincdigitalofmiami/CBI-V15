'use client';

export default function VegasIntelPage() {
  const priorityLevels = {
    high: 'bg-red-900/30 border-red-700/50 text-red-400',
    medium: 'bg-yellow-900/30 border-yellow-700/50 text-yellow-400',
    low: 'bg-zinc-800 border-zinc-700 text-zinc-400',
  };

  const customerStatuses = {
    active: 'bg-green-500',
    atRisk: 'bg-yellow-500',
    winBack: 'bg-red-500',
  };

  return (
    <main className="min-h-screen bg-black p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="mb-6">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-3xl">🎰</span>
            <h1 className="text-3xl font-thin text-white tracking-wide">Vegas Intel</h1>
          </div>
          <p className="text-zinc-400 font-extralight">
            Event-driven sales intelligence for Kevin
          </p>
        </header>

        {/* KPI Bar */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-zinc-950/40 border border-zinc-800 rounded-lg p-4 text-center">
            <div className="text-3xl font-thin text-red-400">—</div>
            <div className="text-zinc-500 text-sm font-extralight">High-Priority Calls</div>
          </div>
          <div className="bg-zinc-950/40 border border-zinc-800 rounded-lg p-4 text-center">
            <div className="text-3xl font-thin text-yellow-400">—</div>
            <div className="text-zinc-500 text-sm font-extralight">At-Risk Accounts</div>
          </div>
          <div className="bg-zinc-950/40 border border-zinc-800 rounded-lg p-4 text-center">
            <div className="text-3xl font-thin text-white">—</div>
            <div className="text-zinc-500 text-sm font-extralight">Events This Week</div>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Upsell Targets */}
          <section className="lg:col-span-2 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-thin text-white">Upsell Targets</h2>
              <span className="text-zinc-500 text-sm font-extralight">Ranked by opportunity</span>
            </div>
            <div className="space-y-3">
              {/* Customer Card Template */}
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="bg-black border border-zinc-800 rounded-lg p-4 hover:border-zinc-700 transition-colors"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <span className={`w-2 h-2 rounded-full ${customerStatuses.active}`}></span>
                      <div>
                        <div className="text-white font-extralight">—</div>
                        <div className="text-zinc-500 text-sm font-extralight">Tier — | Last order: —</div>
                      </div>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded border ${priorityLevels.high}`}>
                      HIGH
                    </span>
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <div>
                      <span className="text-zinc-500">Event:</span>
                      <span className="text-zinc-300 ml-1">—</span>
                    </div>
                    <div>
                      <span className="text-zinc-500">Attendees:</span>
                      <span className="text-zinc-300 ml-1">—</span>
                    </div>
                    <div>
                      <span className="text-zinc-500">Days out:</span>
                      <span className="text-zinc-300 ml-1">—</span>
                    </div>
                  </div>
                  <div className="mt-3 pt-3 border-t border-zinc-800 flex items-center justify-between">
                    <span className="text-zinc-400 text-sm font-extralight">
                      <strong className="text-white">Action:</strong> —
                    </span>
                    <div className="flex gap-2">
                      <button className="px-3 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs rounded transition-colors">
                        Generate Outreach
                      </button>
                      <button className="px-3 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs rounded transition-colors">
                        View Profile
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Margin Alerts */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Margin Alerts</h2>
            <div className="space-y-3">
              <div className="bg-red-900/20 border border-red-800/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-red-400 font-medium">⚠️ LOCK IN EARLY</span>
                </div>
                <p className="text-zinc-300 text-sm font-extralight">—</p>
              </div>
              <div className="bg-green-900/20 border border-green-800/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-green-400 font-medium">✓ DELAY OK</span>
                </div>
                <p className="text-zinc-300 text-sm font-extralight">—</p>
              </div>
            </div>
            <div className="mt-4 text-xs text-zinc-600 font-extralight">
              Context: forecasts.zl_predictions
            </div>
          </section>

          {/* Event Calendar */}
          <section className="lg:col-span-2 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Upcoming Events</h2>
            <div className="grid md:grid-cols-2 gap-4">
              {[1, 2, 3, 4].map((i) => (
                <div
                  key={i}
                  className="bg-black border border-zinc-800 rounded-lg overflow-hidden hover:border-zinc-700 transition-colors"
                >
                  <div className="h-24 bg-zinc-900 flex items-center justify-center">
                    <span className="text-zinc-700 text-sm">Event Image</span>
                  </div>
                  <div className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="text-white font-extralight">—</div>
                      <span className="text-xs px-2 py-1 bg-zinc-800 text-zinc-400 rounded">
                        — days
                      </span>
                    </div>
                    <div className="flex items-center gap-4 text-sm text-zinc-500">
                      <span>— attendees</span>
                      <span>—x multiplier</span>
                    </div>
                    <button className="mt-3 w-full py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs rounded transition-colors">
                      View Affected Customers
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* At-Risk Customers */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">At-Risk / Win-Back</h2>
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="bg-black border border-zinc-800 rounded-lg p-3 hover:border-zinc-700 transition-colors"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span className={`w-2 h-2 rounded-full ${customerStatuses.winBack}`}></span>
                    <span className="text-white font-extralight text-sm">—</span>
                  </div>
                  <div className="text-zinc-500 text-xs font-extralight">
                    Last order: — days ago
                  </div>
                  <div className="text-zinc-400 text-xs mt-1">
                    Suggest: —
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Outreach Generator */}
          <section className="lg:col-span-3 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Outreach Generator</h2>
            <div className="grid md:grid-cols-2 gap-6">
              {/* Inputs */}
              <div className="space-y-4">
                <div>
                  <label className="block text-zinc-400 text-sm font-extralight mb-2">Event</label>
                  <div className="bg-black border border-zinc-800 rounded px-3 py-2 text-zinc-500">—</div>
                </div>
                <div>
                  <label className="block text-zinc-400 text-sm font-extralight mb-2">Customer</label>
                  <div className="bg-black border border-zinc-800 rounded px-3 py-2 text-zinc-500">—</div>
                </div>
                <div>
                  <label className="block text-zinc-400 text-sm font-extralight mb-2">Fryer Setup</label>
                  <div className="bg-black border border-zinc-800 rounded px-3 py-2 text-zinc-500">—</div>
                </div>
                <div>
                  <label className="block text-zinc-400 text-sm font-extralight mb-2">Notes</label>
                  <div className="bg-black border border-zinc-800 rounded px-3 py-2 text-zinc-500 h-20">—</div>
                </div>
              </div>
              {/* Email Draft */}
              <div>
                <label className="block text-zinc-400 text-sm font-extralight mb-2">Email Draft</label>
                <div className="bg-black border border-zinc-800 rounded-lg p-4 h-64 font-mono text-sm">
                  <div className="text-zinc-500 mb-2">Subject: —</div>
                  <div className="text-zinc-400 font-extralight">
                    <p className="mb-2">Hey team —</p>
                    <p className="mb-2">—</p>
                    <p>—</p>
                  </div>
                </div>
                <div className="flex gap-2 mt-3">
                  <button className="px-4 py-2 bg-green-700 hover:bg-green-600 text-white text-sm rounded transition-colors">
                    Copy Email
                  </button>
                  <button className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-sm rounded transition-colors">
                    Regenerate
                  </button>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Data Sources Footer */}
        <div className="mt-6 flex flex-wrap gap-4 text-xs text-zinc-600 font-extralight">
          <span>raw.glide_customers</span>
          <span>•</span>
          <span>raw.events</span>
          <span>•</span>
          <span>raw.customer_volume_history</span>
        </div>
      </div>
    </main>
  );
}
