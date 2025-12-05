"use client";

export default function StrategyPage() {
  return (
    <div className="min-h-screen bg-[#020617] text-slate-100">
      <div className="max-w-6xl mx-auto px-6 py-8">
        <h2 className="text-2xl font-thin tracking-wide mb-2">
          Strategy & Model Stack
        </h2>
        <p className="text-sm text-slate-400 mb-6">
          How CBI‑V15 turns Databento + FRED + fundamentals into ZL forecasts.
        </p>

        <div className="space-y-6 text-[13px] text-slate-300">
          <section className="bg-[#020617] border border-slate-800 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-200 mb-2">
              Pipeline
            </h3>
            <p>
              Databento futures + FRED macro + regimes flow into a single
              canonical matrix <code className="text-xs">training.daily_ml_matrix</code>.
              No joins at train time; all features and targets are pre‑materialized.
            </p>
          </section>

          <section className="bg-[#020617] border border-slate-800 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-200 mb-2">
              Baseline Models
            </h3>
            <ul className="space-y-1">
              <li>• LightGBM baselines at 1w / 1m / 3m / 6m horizons.</li>
              <li>• Targets in price space, trained on return space.</li>
              <li>• Metrics logged: MAE, RMSE, R², MAPE, Sharpe, Sortino.</li>
              <li>• SHAP‑pruned 1m feature set for production‑grade models.</li>
            </ul>
          </section>

          <section className="bg-[#020617] border border-slate-800 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-200 mb-2">
              Next Phases
            </h3>
            <ul className="space-y-1">
              <li>• Probabilistic forecasts (quantile / pinball loss + Monte Carlo bands).</li>
              <li>• GARCH / regime‑switching vol layers on top of ZL returns.</li>
              <li>• Vegas Intel overlay: options skew, CVOL, and dealer positioning.</li>
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
}

