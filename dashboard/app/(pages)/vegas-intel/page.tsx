'use client';

export default function VegasIntelPage() {
    return (
        <main className="min-h-screen bg-[#0a0e1a] p-8">
            <div className="max-w-7xl mx-auto">
                <header className="mb-8">
                    <h1 className="text-3xl font-light text-white mb-2">Vegas Intel / Sales Intelligence</h1>
                    <p className="text-gray-400">Event-driven upsell targets and margin protection for Kevin</p>
                </header>

                <div className="grid gap-6">
                    {/* Kevin's Upsell Targets */}
                    <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
                        <h2 className="text-xl font-light text-white mb-4">Kevin's Upsell Targets</h2>
                        <div className="space-y-3">
                            <div className="bg-[#0a0e1a] border border-green-700/30 p-4 rounded">
                                <div className="flex justify-between items-start mb-2">
                                    <div>
                                        <div className="text-white font-medium">MGM Grand</div>
                                        <div className="text-sm text-gray-400">Tier: Premium | Last order: 3 days ago</div>
                                    </div>
                                    <span className="px-2 py-1 bg-green-900/30 text-green-400 text-xs rounded">HIGH</span>
                                </div>
                                <div className="text-sm text-gray-300">
                                    <strong>Action:</strong> Call now – F1 weekend + prices rising
                                </div>
                            </div>
                        </div>
                        <div className="mt-4 text-sm text-gray-600">
                            Data source: <code className="text-gray-400">raw.glide_customers</code>
                        </div>
                    </section>

                    {/* Event Calendar */}
                    <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
                        <h2 className="text-xl font-light text-white mb-4">Upcoming Events</h2>
                        <p className="text-gray-500">F1, CES, fights, conventions with volume multipliers</p>
                    </section>

                    {/* Margin Protection */}
                    <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
                        <h2 className="text-xl font-light text-white mb-4">Margin Protection Alerts</h2>
                        <div className="p-4 bg-red-900/20 border border-red-700/30 rounded">
                            <p className="text-red-200">
                                <strong>LOCK IN EARLY:</strong> ZL forecast spike before CES (Jan 7-10)
                            </p>
                        </div>
                        <div className="mt-4 text-sm text-gray-600">
                            Context: <code className="text-gray-400">forecasts.zl_predictions</code>
                        </div>
                    </section>

                    {/* At-Risk Customers */}
                    <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
                        <h2 className="text-xl font-light text-white mb-4">At-Risk / Win-Back</h2>
                        <p className="text-gray-500">Customers &gt;14 days since last order</p>
                    </section>
                </div>

                <div className="mt-8 p-4 bg-yellow-900/20 border border-yellow-700/30 rounded-lg">
                    <p className="text-yellow-200 text-sm">
                        ⚠️ <strong>Placeholder:</strong> This page requires implementation. See{' '}
                        <a href="/dashboard/app/vegas-intel/README.md" className="underline">README.md</a> for requirements.
                    </p>
                </div>
            </div>
        </main>
    );
}
