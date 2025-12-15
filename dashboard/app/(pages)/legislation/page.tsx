'use client';

export default function LegislationPage() {
  const usPolicies = [
    { id: 'rfs', name: 'RFS/RVO', agency: 'EPA', status: 'active', impact: 'bullish' },
    { id: 'farm-bill', name: 'Farm Bill 2024', agency: 'USDA', status: 'pending', impact: 'neutral' },
    { id: 'biodiesel-blenders', name: 'Biodiesel Blender Credit', agency: 'Treasury', status: 'expired', impact: 'bearish' },
    { id: 'saf-credit', name: 'SAF Tax Credit', agency: 'Treasury', status: 'active', impact: 'bullish' },
  ];

  const tradePolicies = [
    { id: 'section-301', name: 'Section 301 Tariffs', target: 'China', status: 'active', rate: '25%' },
    { id: 'wto-dispute', name: 'WTO Ag Subsidy Dispute', target: 'Brazil', status: 'pending', rate: '—' },
    { id: 'usmca', name: 'USMCA Review', target: 'Mexico/Canada', status: 'scheduled', rate: '—' },
  ];

  const timeline = [
    { date: 'Dec 15', event: 'EPA RVO Final Rule', impact: 'bullish', days: 1 },
    { date: 'Jan 3', event: 'New Congress Convenes', impact: 'neutral', days: 20 },
    { date: 'Jan 20', event: 'Administration Transition', impact: 'mixed', days: 37 },
    { date: 'Feb 1', event: 'USDA Outlook Forum', impact: 'neutral', days: 49 },
    { date: 'Mar 15', event: 'Farm Bill Deadline', impact: 'bullish', days: 91 },
  ];

  return (
    <main className="min-h-screen bg-black p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-3xl">⚖️</span>
            <h1 className="text-3xl font-thin text-white tracking-wide">Legislation & Policy Intel</h1>
          </div>
          <p className="text-zinc-400 font-extralight">
            Track RFS, tariffs, Farm Bill, and trade regulations impacting ZL
          </p>
        </header>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* US Policy Stream */}
          <section className="lg:col-span-2 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">US Policy Stream</h2>
            <div className="space-y-3">
              {usPolicies.map((policy) => (
                <div
                  key={policy.id}
                  className="flex items-center gap-4 bg-black border border-zinc-800 p-4 rounded-lg hover:border-zinc-700 transition-colors"
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-white font-extralight">{policy.name}</span>
                      <span className="text-xs px-2 py-0.5 bg-zinc-800 text-zinc-400 rounded">
                        {policy.agency}
                      </span>
                    </div>
                  </div>
                  <span
                    className={`text-xs px-2 py-1 rounded ${
                      policy.status === 'active'
                        ? 'bg-green-900/30 text-green-400'
                        : policy.status === 'pending'
                        ? 'bg-yellow-900/30 text-yellow-400'
                        : 'bg-zinc-800 text-zinc-500'
                    }`}
                  >
                    {policy.status}
                  </span>
                  <span
                    className={`text-xs px-2 py-1 rounded ${
                      policy.impact === 'bullish'
                        ? 'bg-green-900/20 text-green-400 border border-green-800/50'
                        : policy.impact === 'bearish'
                        ? 'bg-red-900/20 text-red-400 border border-red-800/50'
                        : 'bg-zinc-900 text-zinc-400 border border-zinc-700'
                    }`}
                  >
                    {policy.impact}
                  </span>
                </div>
              ))}
            </div>
          </section>

          {/* Impact Timeline */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Impact Timeline</h2>
            <div className="space-y-4">
              {timeline.map((item, idx) => (
                <div key={idx} className="flex items-start gap-3">
                  <div className="w-16 flex-shrink-0">
                    <div className="text-white font-extralight text-sm">{item.date}</div>
                    <div className="text-zinc-600 text-xs">{item.days}d</div>
                  </div>
                  <div
                    className={`w-2 h-2 rounded-full mt-1.5 ${
                      item.impact === 'bullish'
                        ? 'bg-green-500'
                        : item.impact === 'bearish'
                        ? 'bg-red-500'
                        : 'bg-zinc-500'
                    }`}
                  />
                  <div className="flex-1">
                    <div className="text-zinc-300 font-extralight text-sm">{item.event}</div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* China / Trade Policy */}
          <section className="lg:col-span-2 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Trade Policy</h2>
            <div className="space-y-3">
              {tradePolicies.map((policy) => (
                <div
                  key={policy.id}
                  className="flex items-center gap-4 bg-black border border-zinc-800 p-4 rounded-lg hover:border-zinc-700 transition-colors"
                >
                  <div className="flex-1">
                    <div className="text-white font-extralight">{policy.name}</div>
                    <div className="text-zinc-500 text-sm font-extralight">Target: {policy.target}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-white font-extralight">{policy.rate}</div>
                    <div className="text-zinc-500 text-xs">rate</div>
                  </div>
                  <span
                    className={`text-xs px-2 py-1 rounded ${
                      policy.status === 'active'
                        ? 'bg-red-900/30 text-red-400'
                        : policy.status === 'pending'
                        ? 'bg-yellow-900/30 text-yellow-400'
                        : 'bg-zinc-800 text-zinc-400'
                    }`}
                  >
                    {policy.status}
                  </span>
                </div>
              ))}
            </div>
          </section>

          {/* Quick Stats */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Policy Summary</h2>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-zinc-400 font-extralight">Active Policies</span>
                <span className="text-white font-extralight">—</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-zinc-400 font-extralight">Pending Reviews</span>
                <span className="text-yellow-400 font-extralight">—</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-zinc-400 font-extralight">Net Policy Bias</span>
                <span className="text-zinc-400 font-extralight">—</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-zinc-400 font-extralight">Next Major Event</span>
                <span className="text-white font-extralight">—</span>
              </div>
            </div>
          </section>

          {/* News Feed Placeholder */}
          <section className="lg:col-span-3 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Recent Policy News</h2>
            <div className="grid md:grid-cols-3 gap-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="bg-black border border-zinc-800 p-4 rounded-lg">
                  <div className="h-4 bg-zinc-800 rounded w-3/4 mb-2"></div>
                  <div className="h-3 bg-zinc-900 rounded w-full mb-1"></div>
                  <div className="h-3 bg-zinc-900 rounded w-2/3"></div>
                  <div className="mt-3 flex items-center gap-2">
                    <span className="text-zinc-600 text-xs">—</span>
                    <span className="text-zinc-700">•</span>
                    <span className="text-zinc-600 text-xs">—</span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* Data Sources Footer */}
        <div className="mt-6 flex flex-wrap gap-4 text-xs text-zinc-600 font-extralight">
          <span>raw.news_articles</span>
          <span>•</span>
          <span>staging.news_topic_signals</span>
          <span>•</span>
          <span>reference.policy_calendar</span>
        </div>
      </div>
    </main>
  );
}
