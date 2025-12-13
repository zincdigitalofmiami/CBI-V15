'use client';

export default function StrategyPage() {
    const horizons = ['1W', '1M', '3M', '6M'];

    return (
        <main className="min-h-screen bg-[#0a0e1a] p-8">
            <div className="max-w-7xl mx-auto">
                <header className="mb-8">
                    <h1 className="text-3xl font-light text-white mb-2">ZL Strategy & Procurement Plan</h1>
                    <p className="text-gray-400">Hedge ladder, scenarios, and P&L distributions</p>
                </header>

                <div className="grid gap-6">
                    {/* Strategy Summary */}
                    <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
                        <h2 className="text-xl font-light text-white mb-4">Strategy Summary</h2>
                        <p className="text-gray-300 italic">AutoGluon + SQL strategy summary will appear here</p>
                        <div className="mt-4 text-sm text-gray-600">
                            Data source: <code className="text-gray-400">forecasts.zl_predictions, reference.model_registry</code>
                        </div>
                    </section>

                    {/* Hedge Ladder */}
                    <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
                        <h2 className="text-xl font-light text-white mb-4">Hedge Ladder</h2>
                        <div className="grid grid-cols-4 gap-4">
                            {horizons.map((h) => (
                                <div key={h} className="bg-[#0a0e1a] border border-[#2a2f3e] p-4 rounded text-center">
                                    <div className="text-sm text-gray-400">{h}</div>
                                    <div className="text-2xl text-white font-light">--</div>
                                    <div className="text-xs text-gray-500">lbs</div>
                                </div>
                            ))}
                        </div>
                        <div className="mt-4 text-sm text-gray-600">
                            Data source: <code className="text-gray-400">forecasts.zl_predictions</code>
                        </div>
                    </section>

                    {/* Scenario Panel */}
                    <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
                        <h2 className="text-xl font-light text-white mb-4">Scenario Analysis</h2>
                        <div className="space-y-2">
                            <div className="flex items-center gap-3">
                                <input type="checkbox" className="w-4 h-4" />
                                <span className="text-gray-300">China Soft</span>
                            </div>
                            <div className="flex items-center gap-3">
                                <input type="checkbox" className="w-4 h-4" />
                                <span className="text-gray-300">Tariffs Escalate</span>
                            </div>
                            <div className="flex items-center gap-3">
                                <input type="checkbox" className="w-4 h-4" />
                                <span className="text-gray-300">Weather Shock</span>
                            </div>
                        </div>
                    </section>

                    {/* P&L Distribution */}
                    <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
                        <h2 className="text-xl font-light text-white mb-4">P&L Distribution</h2>
                        <p className="text-gray-500">Compare current strategy vs benchmarks</p>
                        <div className="mt-4 text-sm text-gray-600">
                            Data source: <code className="text-gray-400">training.daily_ml_matrix_zl</code>
                        </div>
                    </section>
                </div>

                <div className="mt-8 p-4 bg-yellow-900/20 border border-yellow-700/30 rounded-lg">
                    <p className="text-yellow-200 text-sm">
                        ⚠️ <strong>Placeholder:</strong> This page requires implementation. See{' '}
                        <a href="/dashboard/app/strategy/README.md" className="underline">README.md</a> for requirements.
                    </p>
                </div>
            </div>
        </main>
    );
}
