"use client";

export default function AdminPage() {
  return (
    <main className="min-h-screen bg-black p-6 lg:p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-3xl">⚙️</span>
            <h1 className="text-3xl font-thin text-white tracking-wide">Admin Configuration</h1>
          </div>
          <p className="text-zinc-400 font-extralight">
            Business settings, thresholds, and visibility controls
          </p>
        </header>

        <div className="space-y-6">
          {/* App Settings */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-6">App Settings</h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-zinc-400 text-sm font-extralight mb-2">
                  Base Volume (lbs)
                </label>
                <input
                  type="number"
                  placeholder="1000000"
                  className="w-full bg-black border border-zinc-800 rounded-lg px-4 py-3 text-white font-extralight focus:border-zinc-600 focus:outline-none transition-colors"
                />
              </div>
              <div>
                <label className="block text-zinc-400 text-sm font-extralight mb-2">
                  Default Currency
                </label>
                <select className="w-full bg-black border border-zinc-800 rounded-lg px-4 py-3 text-white font-extralight focus:border-zinc-600 focus:outline-none transition-colors">
                  <option value="USD">USD - US Dollar</option>
                  <option value="EUR">EUR - Euro</option>
                  <option value="BRL">BRL - Brazilian Real</option>
                </select>
              </div>
              <div>
                <label className="block text-zinc-400 text-sm font-extralight mb-2">
                  Contract Size
                </label>
                <input
                  type="number"
                  placeholder="60000"
                  className="w-full bg-black border border-zinc-800 rounded-lg px-4 py-3 text-white font-extralight focus:border-zinc-600 focus:outline-none transition-colors"
                />
                <span className="text-zinc-600 text-xs font-extralight">lbs per contract</span>
              </div>
              <div>
                <label className="block text-zinc-400 text-sm font-extralight mb-2">
                  Refresh Interval
                </label>
                <select className="w-full bg-black border border-zinc-800 rounded-lg px-4 py-3 text-white font-extralight focus:border-zinc-600 focus:outline-none transition-colors">
                  <option value="1">1 minute</option>
                  <option value="5">5 minutes</option>
                  <option value="15">15 minutes</option>
                  <option value="30">30 minutes</option>
                </select>
              </div>
            </div>
          </section>

          {/* Risk Thresholds */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-6">Risk Thresholds</h2>
            <p className="text-zinc-500 text-sm font-extralight mb-4">
              Set threshold values for color-coded risk indicators
            </p>
            <div className="space-y-4">
              {["China Demand", "Tariff Risk", "Biofuel Policy", "FX Impact"].map((metric) => (
                <div key={metric} className="grid grid-cols-4 gap-4 items-center">
                  <div className="text-zinc-300 font-extralight">{metric}</div>
                  <div>
                    <label className="block text-green-500 text-xs font-extralight mb-1">
                      Green
                    </label>
                    <input
                      type="number"
                      placeholder="70"
                      className="w-full bg-black border border-zinc-800 rounded px-3 py-2 text-white text-sm font-extralight focus:border-green-800 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-yellow-500 text-xs font-extralight mb-1">
                      Yellow
                    </label>
                    <input
                      type="number"
                      placeholder="50"
                      className="w-full bg-black border border-zinc-800 rounded px-3 py-2 text-white text-sm font-extralight focus:border-yellow-800 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-red-500 text-xs font-extralight mb-1">Red</label>
                    <input
                      type="number"
                      placeholder="30"
                      className="w-full bg-black border border-zinc-800 rounded px-3 py-2 text-white text-sm font-extralight focus:border-red-800 focus:outline-none"
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Visibility Toggles */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-6">Visibility Toggles</h2>
            <p className="text-zinc-500 text-sm font-extralight mb-4">
              Control which components appear on dashboard pages
            </p>
            <div className="space-y-4">
              {[
                { label: "Show Chris's Four Factors", default: true },
                { label: "Show Big-8 Heatmap", default: true },
                { label: "Show Forward Curve", default: true },
                { label: "Show Vegas Intel in Nav", default: true },
                { label: "Show Quant Admin (internal)", default: false },
                { label: "Enable Scenario Analysis", default: true },
              ].map((toggle, idx) => (
                <label
                  key={idx}
                  className="flex items-center gap-4 cursor-pointer hover:bg-zinc-900/30 p-2 -mx-2 rounded transition-colors"
                >
                  <input
                    type="checkbox"
                    defaultChecked={toggle.default}
                    className="w-5 h-5 bg-black border-zinc-700 rounded focus:ring-0 focus:ring-offset-0"
                  />
                  <span className="text-zinc-300 font-extralight">{toggle.label}</span>
                </label>
              ))}
            </div>
          </section>

          {/* Notification Settings */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-6">Notifications</h2>
            <div className="space-y-4">
              {[
                { label: "Price alerts", desc: "Notify when ZL crosses threshold" },
                { label: "Regime changes", desc: "Notify on market regime shifts" },
                { label: "Policy events", desc: "Upcoming legislation deadlines" },
                { label: "Customer alerts", desc: "At-risk customer notifications" },
              ].map((item, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between py-2 border-b border-zinc-800 last:border-0"
                >
                  <div>
                    <div className="text-zinc-300 font-extralight">{item.label}</div>
                    <div className="text-zinc-600 text-sm font-extralight">{item.desc}</div>
                  </div>
                  <input
                    type="checkbox"
                    defaultChecked
                    className="w-5 h-5 bg-black border-zinc-700 rounded"
                  />
                </div>
              ))}
            </div>
          </section>

          {/* Action Buttons */}
          <div className="flex gap-4">
            <button className="px-6 py-3 bg-green-700 hover:bg-green-600 text-white font-extralight rounded-lg transition-colors">
              Save Changes
            </button>
            <button className="px-6 py-3 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 font-extralight rounded-lg transition-colors">
              Reset to Defaults
            </button>
          </div>
        </div>

        {/* Data Sources Footer */}
        <div className="mt-8 text-xs text-zinc-600 font-extralight">
          Settings stored in: reference.app_config
        </div>
      </div>
    </main>
  );
}
