"use client";

export default function QuantAdminPage() {
  const pipelineStages = [
    { name: "Databento", status: "ok", lastRun: "—", rows: "—" },
    { name: "FRED", status: "ok", lastRun: "—", rows: "—" },
    { name: "CFTC COT", status: "ok", lastRun: "—", rows: "—" },
    { name: "USDA", status: "pending", lastRun: "—", rows: "—" },
    { name: "News/Sentiment", status: "pending", lastRun: "—", rows: "—" },
    { name: "EPA RIN", status: "pending", lastRun: "—", rows: "—" },
  ];

  const schemas = [
    { name: "raw", tables: "—", rows: "—", coverage: "—" },
    { name: "staging", tables: "—", rows: "—", coverage: "—" },
    { name: "features", tables: "—", rows: "—", coverage: "—" },
    { name: "training", tables: "—", rows: "—", coverage: "—" },
    { name: "forecasts", tables: "—", rows: "—", coverage: "—" },
  ];

  const models = [
    { name: "Crush Specialist", type: "TabularPredictor", status: "pending" },
    { name: "China Specialist", type: "TabularPredictor", status: "pending" },
    { name: "Core ZL Forecaster", type: "TimeSeriesPredictor", status: "pending" },
    { name: "Meta Ensemble", type: "GreedyEnsemble", status: "pending" },
  ];

  return (
    <main className="min-h-screen bg-black p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-3xl">🔬</span>
            <h1 className="text-3xl font-thin text-white tracking-wide">Quant Admin</h1>
          </div>
          <p className="text-zinc-400 font-extralight">
            Pipeline health, data coverage, and model registry
          </p>
        </header>

        {/* Status Overview */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-zinc-950/40 border border-zinc-800 rounded-lg p-4 text-center">
            <div className="text-3xl font-thin text-green-400">—</div>
            <div className="text-zinc-500 text-sm font-extralight">Pipelines OK</div>
          </div>
          <div className="bg-zinc-950/40 border border-zinc-800 rounded-lg p-4 text-center">
            <div className="text-3xl font-thin text-yellow-400">—</div>
            <div className="text-zinc-500 text-sm font-extralight">Pending</div>
          </div>
          <div className="bg-zinc-950/40 border border-zinc-800 rounded-lg p-4 text-center">
            <div className="text-3xl font-thin text-white">—</div>
            <div className="text-zinc-500 text-sm font-extralight">Total Features</div>
          </div>
          <div className="bg-zinc-950/40 border border-zinc-800 rounded-lg p-4 text-center">
            <div className="text-3xl font-thin text-white">—</div>
            <div className="text-zinc-500 text-sm font-extralight">Last Training</div>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Pipeline Health */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Pipeline Health</h2>
            <div className="space-y-2">
              {pipelineStages.map((stage) => (
                <div
                  key={stage.name}
                  className="flex items-center justify-between bg-black border border-zinc-800 rounded-lg p-3 hover:border-zinc-700 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span
                      className={`w-2 h-2 rounded-full ${
                        stage.status === "ok"
                          ? "bg-green-500"
                          : stage.status === "error"
                            ? "bg-red-500"
                            : "bg-yellow-500"
                      }`}
                    ></span>
                    <span className="text-zinc-300 font-extralight">{stage.name}</span>
                  </div>
                  <div className="flex items-center gap-6 text-sm">
                    <span className="text-zinc-500">{stage.lastRun}</span>
                    <span className="text-zinc-400">{stage.rows} rows</span>
                  </div>
                </div>
              ))}
            </div>
            <button className="mt-4 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-sm font-extralight rounded transition-colors">
              Refresh Status
            </button>
          </section>

          {/* Schema Coverage */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Schema Coverage</h2>
            <div className="space-y-2">
              {schemas.map((schema) => (
                <div
                  key={schema.name}
                  className="flex items-center justify-between bg-black border border-zinc-800 rounded-lg p-3"
                >
                  <span className="text-zinc-300 font-extralight font-mono">{schema.name}</span>
                  <div className="flex items-center gap-6 text-sm">
                    <span className="text-zinc-500">{schema.tables} tables</span>
                    <span className="text-zinc-400">{schema.rows} rows</span>
                    <span className="text-zinc-500">{schema.coverage}</span>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4 text-xs text-zinc-600 font-extralight">
              Source: MotherDuck cbi_v15
            </div>
          </section>

          {/* Model Registry */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Model Registry</h2>
            <div className="space-y-2">
              {models.map((model) => (
                <div
                  key={model.name}
                  className="flex items-center justify-between bg-black border border-zinc-800 rounded-lg p-3"
                >
                  <div>
                    <div className="text-zinc-300 font-extralight">{model.name}</div>
                    <div className="text-zinc-600 text-xs font-extralight">{model.type}</div>
                  </div>
                  <span
                    className={`text-xs px-2 py-1 rounded ${
                      model.status === "active"
                        ? "bg-green-900/30 text-green-400"
                        : model.status === "training"
                          ? "bg-blue-900/30 text-blue-400"
                          : "bg-zinc-800 text-zinc-500"
                    }`}
                  >
                    {model.status}
                  </span>
                </div>
              ))}
            </div>
            <button className="mt-4 px-4 py-2 bg-green-700 hover:bg-green-600 text-white text-sm font-extralight rounded transition-colors">
              Trigger Training
            </button>
          </section>

          {/* Recent Runs */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Recent Runs</h2>
            <div className="space-y-2">
              {[1, 2, 3, 4].map((i) => (
                <div
                  key={i}
                  className="flex items-center justify-between bg-black border border-zinc-800 rounded-lg p-3"
                >
                  <div>
                    <div className="text-zinc-300 font-extralight text-sm">—</div>
                    <div className="text-zinc-600 text-xs font-extralight">—</div>
                  </div>
                  <div className="text-right">
                    <div className="text-zinc-400 text-sm">—</div>
                    <div className="text-zinc-600 text-xs">duration</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4 text-xs text-zinc-600 font-extralight">
              Source: ops.training_runs
            </div>
          </section>

          {/* Feature Matrix Status */}
          <section className="lg:col-span-2 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Feature Matrix Status</h2>
            <div className="grid grid-cols-4 gap-4 mb-4">
              <div className="bg-black border border-zinc-800 rounded-lg p-4 text-center">
                <div className="text-2xl font-thin text-white">—</div>
                <div className="text-zinc-500 text-xs font-extralight">Total Features</div>
              </div>
              <div className="bg-black border border-zinc-800 rounded-lg p-4 text-center">
                <div className="text-2xl font-thin text-white">—</div>
                <div className="text-zinc-500 text-xs font-extralight">Date Range</div>
              </div>
              <div className="bg-black border border-zinc-800 rounded-lg p-4 text-center">
                <div className="text-2xl font-thin text-green-400">—</div>
                <div className="text-zinc-500 text-xs font-extralight">Complete</div>
              </div>
              <div className="bg-black border border-zinc-800 rounded-lg p-4 text-center">
                <div className="text-2xl font-thin text-yellow-400">—</div>
                <div className="text-zinc-500 text-xs font-extralight">Missing</div>
              </div>
            </div>
            <div className="h-32 bg-black border border-zinc-800 rounded-lg flex items-center justify-center">
              <span className="text-zinc-600 text-sm font-extralight">
                Feature coverage heatmap
              </span>
            </div>
            <div className="mt-4 text-xs text-zinc-600 font-extralight">
              Source: features.daily_ml_matrix_zl
            </div>
          </section>
        </div>

        {/* Data Sources Footer */}
        <div className="mt-6 flex flex-wrap gap-4 text-xs text-zinc-600 font-extralight">
          <span>ops.ingestion_completion</span>
          <span>•</span>
          <span>ops.training_runs</span>
          <span>•</span>
          <span>reference.model_registry</span>
        </div>
      </div>
    </main>
  );
}
