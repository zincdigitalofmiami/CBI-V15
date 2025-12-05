'use client';

export default function AdminPage() {
    return (
        <main className="min-h-screen bg-[#0a0e1a] p-8">
            <div className="max-w-4xl mx-auto">
                <header className="mb-8">
                    <h1 className="text-3xl font-light text-white mb-2">Admin / Business Configuration</h1>
                    <p className="text-gray-400">Configure volumes, thresholds, and visibility settings</p>
                </header>

                <div className="space-y-6">
                    {/* App Settings */}
                    <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
                        <h2 className="text-xl font-light text-white mb-4">App Settings</h2>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm text-gray-400 mb-2">Base Volume (lbs)</label>
                                <input
                                    type="number"
                                    defaultValue="1000000"
                                    className="w-full bg-[#0a0e1a] border border-[#2a2f3e] rounded px-3 py-2 text-white"
                                />
                            </div>
                            <div>
                                <label className="block text-sm text-gray-400 mb-2">Currency</label>
                                <select className="w-full bg-[#0a0e1a] border border-[#2a2f3e] rounded px-3 py-2 text-white">
                                    <option>USD</option>
                                </select>
                            </div>
                        </div>
                    </section>

                    {/* Risk Thresholds */}
                    <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
                        <h2 className="text-xl font-light text-white mb-4">Risk Thresholds</h2>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm text-gray-400 mb-2">China Demand - Green Threshold</label>
                                <input
                                    type="number"
                                    defaultValue="70"
                                    className="w-full bg-[#0a0e1a] border border-[#2a2f3e] rounded px-3 py-2 text-white"
                                />
                            </div>
                            <div>
                                <label className="block text-sm text-gray-400 mb-2">Tariff Risk - Red Threshold</label>
                                <input
                                    type="number"
                                    defaultValue="60"
                                    className="w-full bg-[#0a0e1a] border border-[#2a2f3e] rounded px-3 py-2 text-white"
                                />
                            </div>
                        </div>
                    </section>

                    {/* Visibility Toggles */}
                    <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-6">
                        <h2 className="text-xl font-light text-white mb-4">Visibility Toggles</h2>
                        <div className="space-y-3">
                            <div className="flex items-center gap-3">
                                <input type="checkbox" defaultChecked className="w-4 h-4" />
                                <span className="text-gray-300">Show Chris's Four Factors</span>
                            </div>
                            <div className="flex items-center gap-3">
                                <input type="checkbox" defaultChecked className="w-4 h-4" />
                                <span className="text-gray-300">Show Big-8 Heatmap</span>
                            </div>
                            <div className="flex items-center gap-3">
                                <input type="checkbox" defaultChecked className="w-4 h-4" />
                                <span className="text-gray-300">Show Vegas Intel in Nav</span>
                            </div>
                        </div>
                    </section>

                    <div className="flex gap-4">
                        <button className="px-6 py-2 bg-green-700 hover:bg-green-600 text-white rounded">
                            Save Changes
                        </button>
                        <button className="px-6 py-2 bg-[#2a2f3e] hover:bg-[#3a3f4e] text-white rounded">
                            Reset to Defaults
                        </button>
                    </div>
                </div>

                <div className="mt-8 p-4 bg-yellow-900/20 border border-yellow-700/30 rounded-lg">
                    <p className="text-yellow-200 text-sm">
                        ⚠️ <strong>Placeholder:</strong> This page requires API implementation. See{' '}
                        <a href="/dashboard/app/admin/README.md" className="underline">README.md</a> for requirements.
                    </p>
                </div>
            </div>
        </main>
    );
}
