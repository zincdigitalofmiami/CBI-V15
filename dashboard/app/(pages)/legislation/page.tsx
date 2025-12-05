'use client';

export default function LegislationPage() {
    return (
        <main className="min-h-screen bg-[#0a0e1a] p-8">
            <div className="max-w-7xl mx-auto">
                <header className="mb-8">
                    <h1 className="text-3xl font-light text-white mb-2">Legislation & Policy Intel</h1>
                    <p className="text-gray-400">Track RFS, tariffs, Farm Bill, and logistics regulations impacting ZL</p>
                </header>

                <div className="grid gap-6">
                    {/* US Policy Stream */}
                    <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
                        <h2 className="text-xl font-light text-white mb-4">US Policy Stream</h2>
                        <p className="text-gray-500">RFS, Farm Bill, EPA, USDA, CBP, DHS events</p>
                        <div className="mt-4 text-sm text-gray-600">
                            Data source: <code className="text-gray-400">raw.news_articles</code>
                        </div>
                    </section>

                    {/* China / Trade Policy */}
                    <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
                        <h2 className="text-xl font-light text-white mb-4">China / Trade Policy</h2>
                        <p className="text-gray-500">Section 301, WTO disputes, tariff moves</p>
                        <div className="mt-4 text-sm text-gray-600">
                            Data source: <code className="text-gray-400">raw.gdelt_events</code>
                        </div>
                    </section>

                    {/* Impact Timeline */}
                    <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
                        <h2 className="text-xl font-light text-white mb-4">Impact Timeline (30-90 days)</h2>
                        <p className="text-gray-500">Major events with bullish/bearish/neutral tags</p>
                        <div className="mt-4 text-sm text-gray-600">
                            Data source: <code className="text-gray-400">staging.news_topic_signals</code>
                        </div>
                    </section>
                </div>

                <div className="mt-8 p-4 bg-yellow-900/20 border border-yellow-700/30 rounded-lg">
                    <p className="text-yellow-200 text-sm">
                        ⚠️ <strong>Placeholder:</strong> This page requires implementation. See{' '}
                        <a href="/dashboard/app/legislation/README.md" className="underline">README.md</a> for requirements.
                    </p>
                </div>
            </div>
        </main>
    );
}
