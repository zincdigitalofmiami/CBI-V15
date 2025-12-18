"use client";

import { useEffect, useState } from "react";

const MLFLOW_UI_URL = process.env.NEXT_PUBLIC_MLFLOW_UI_URL || "";

type DqOhlcvDailyRow = {
  report_date?: unknown;
  total_rows?: number;
  null_ratio?: number;
  duplicate_count?: number;
  max_date?: unknown;
  days_stale?: number;
  gap_count?: number;
  max_gap_days?: number;
  status?: string;
  created_at?: unknown;
};

type DuckDbValueWrapper = { value: unknown };

function isDuckDbValueWrapper(v: unknown): v is DuckDbValueWrapper {
  return typeof v === "object" && v !== null && "value" in v;
}

function asIsoString(v: unknown): string {
  if (isDuckDbValueWrapper(v)) return String(v.value ?? "");
  return String(v ?? "");
}

type MetricsResponse = {
  success?: boolean;
  data?: {
    v15_matrix_stats?: Record<string, unknown> | null;
    preconditioning_params?: Record<string, unknown> | null;
    dq_ohlcv_daily?: Record<string, unknown> | null;
  };
  explanation?: string | null;
  explanation_error?: string | null;
};

export default function QuantFoxPage() {
  const [v15Meta, setV15Meta] = useState<{ row_count?: number; min_date?: string; max_date?: string; last_updated_at?: string }>(
    {},
  );
  const [precondParams, setPrecondParams] = useState<{ param_count?: number; last_updated_at?: string }>({});
  const [dqOhlcvDaily, setDqOhlcvDaily] = useState<DqOhlcvDailyRow | null>(null);
  const [aiExplainer, setAiExplainer] = useState<string>("");
  const [aiExplainerError, setAiExplainerError] = useState<string>("");
  const [aiLoading, setAiLoading] = useState(false);

  useEffect(() => {
    async function fetchAnoFox() {
      try {
        const res = await fetch("/api/training/metrics", { cache: "no-store" });
        const json: unknown = await res.json();
        const j = json as MetricsResponse;

        const v15Stats = (j?.data?.v15_matrix_stats ?? {}) as Record<string, unknown>;
        const pre = (j?.data?.preconditioning_params ?? {}) as Record<string, unknown>;
        const dq = (j?.data?.dq_ohlcv_daily ?? null) as Record<string, unknown> | null;

        const v15_row_count =
          typeof v15Stats?.row_count === "number" ? v15Stats.row_count : Number(v15Stats?.row_count);
        const v15_min_date = asIsoString(v15Stats?.min_date);
        const v15_max_date = asIsoString(v15Stats?.max_date);
        const v15_last_updated_at = asIsoString(v15Stats?.last_updated_at);
        setV15Meta({
          row_count: Number.isFinite(v15_row_count) ? v15_row_count : undefined,
          min_date: v15_min_date && v15_min_date !== "null" ? v15_min_date : undefined,
          max_date: v15_max_date && v15_max_date !== "null" ? v15_max_date : undefined,
          last_updated_at: v15_last_updated_at && v15_last_updated_at !== "null" ? v15_last_updated_at : undefined,
        });

        const param_count = typeof pre?.param_count === "number" ? pre.param_count : Number(pre?.param_count);
        const pre_last_updated_at = asIsoString(pre?.last_updated_at);
        setPrecondParams({
          param_count: Number.isFinite(param_count) ? param_count : undefined,
          last_updated_at: pre_last_updated_at && pre_last_updated_at !== "null" ? pre_last_updated_at : undefined,
        });

        const row = dq ? (dq as unknown as DqOhlcvDailyRow) : null;
        setDqOhlcvDaily(row);
      } catch {
        // ignore
      }
    }
    fetchAnoFox();
  }, []);

  async function runExplainer() {
    try {
      setAiLoading(true);
      setAiExplainerError("");
      const res = await fetch("/api/training/metrics?explain=1", { cache: "no-store" });
      const json: unknown = await res.json();
      const obj = typeof json === "object" && json !== null ? (json as Record<string, unknown>) : {};
      const text = typeof obj["explanation"] === "string" ? obj["explanation"] : "";
      const err = typeof obj["explanation_error"] === "string" ? obj["explanation_error"] : "";
      setAiExplainer(text);
      setAiExplainerError(err);
    } catch (e) {
      setAiExplainerError(e instanceof Error ? e.message : String(e));
    } finally {
      setAiLoading(false);
    }
  }

  const dqStatus = String(dqOhlcvDaily?.status || "—").toUpperCase();

  return (
    <main className="min-h-screen bg-[#070a12] px-10 py-10">
      <div className="max-w-screen-2xl mx-auto space-y-6">
        <header className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <div className="h-9 w-9 rounded-xl bg-white/5 border border-white/10" />
              <h1 className="text-3xl md:text-4xl font-thin text-white tracking-wide">Quant Fox</h1>
            </div>
            <div className="text-zinc-400 font-extralight mt-1">AnoFox-style data preparation + MLflow</div>
          </div>
          <div className="flex items-center gap-2">
            <a
              href="/quant-admin"
              className="px-3 py-1.5 rounded-lg text-sm font-extralight bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors"
            >
              Quant Admin
            </a>
            <a
              href="https://anofox.com/docs/forecast/guides/basic-workflow"
              target="_blank"
              rel="noreferrer"
              className="px-3 py-1.5 rounded-lg text-sm font-extralight bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors"
            >
              AnoFox Workflow
            </a>
            {MLFLOW_UI_URL ? (
              <a
                href={MLFLOW_UI_URL}
                target="_blank"
                rel="noreferrer"
                className="px-3 py-1.5 rounded-lg text-sm font-extralight bg-green-700 hover:bg-green-600 text-white transition-colors"
              >
                Open MLflow
              </a>
            ) : null}
          </div>
        </header>

        <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6 space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="bg-black border border-zinc-800 rounded-lg p-4">
              <div className="text-zinc-400 text-xs font-extralight mb-1">Training Matrix (V15 preconditioned)</div>
              <div className="text-2xl font-thin text-white">
                {typeof v15Meta.row_count === "number" ? v15Meta.row_count.toLocaleString() : "—"}
              </div>
              <div className="text-xs text-zinc-600 font-extralight">
                {v15Meta.min_date && v15Meta.max_date ? `${v15Meta.min_date} → ${v15Meta.max_date}` : "—"}
              </div>
              <div className="text-xs text-zinc-600 font-extralight mt-2">
                Updated: {v15Meta.last_updated_at ? v15Meta.last_updated_at.replace("T", " ").slice(0, 19) : "—"}
              </div>
            </div>

            <div className="bg-black border border-zinc-800 rounded-lg p-4">
              <div className="text-zinc-400 text-xs font-extralight mb-1">Preconditioning Params (train-only)</div>
              <div className="text-2xl font-thin text-white">
                {typeof precondParams.param_count === "number" ? precondParams.param_count.toLocaleString() : "—"}
              </div>
              <div className="text-xs text-zinc-600 font-extralight mt-2">
                Updated:{" "}
                {precondParams.last_updated_at ? precondParams.last_updated_at.replace("T", " ").slice(0, 19) : "—"}
              </div>
              <div className="text-xs text-zinc-600 font-extralight mt-2">Impute=median, clamp=±10×IQR, scale=(x−median)/IQR</div>
            </div>

            <div className="bg-black border border-zinc-800 rounded-lg p-4">
              <div className="text-zinc-400 text-xs font-extralight mb-1">Staging Quality (ohlcv_daily)</div>
              <div className="flex items-center gap-2">
                <div className="text-2xl font-thin text-white">
                  {typeof dqOhlcvDaily?.total_rows === "number" ? dqOhlcvDaily.total_rows.toLocaleString() : "—"}
                </div>
                <span
                  className={`text-xs px-2 py-1 rounded ${
                    dqStatus === "PASSED"
                      ? "bg-green-900/30 text-green-400"
                      : dqStatus === "FAILED"
                        ? "bg-red-900/30 text-red-400"
                        : "bg-yellow-900/30 text-yellow-300"
                  }`}
                >
                  {dqStatus}
                </span>
              </div>
              <div className="text-xs text-zinc-600 font-extralight mt-2">
                gaps filled: {typeof dqOhlcvDaily?.gap_count === "number" ? dqOhlcvDaily.gap_count.toLocaleString() : "—"} · days stale:{" "}
                {typeof dqOhlcvDaily?.days_stale === "number" ? dqOhlcvDaily.days_stale : "—"}
              </div>
              <div className="text-xs text-zinc-600 font-extralight">
                max gap days (raw): {typeof dqOhlcvDaily?.max_gap_days === "number" ? dqOhlcvDaily.max_gap_days : "—"}
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between gap-4">
            <div className="text-xs text-zinc-600 font-extralight">
              Local rebuild:
              <div className="mt-1 grid grid-cols-1 lg:grid-cols-2 gap-2">
                <pre className="text-[11px] bg-zinc-950 border border-zinc-800 rounded p-2 overflow-auto">
                  {"python src/engines/anofox/build_all_features.py"}
                </pre>
                <pre className="text-[11px] bg-zinc-950 border border-zinc-800 rounded p-2 overflow-auto">
                  {"python src/engines/anofox/build_training.py"}
                </pre>
              </div>
            </div>
            <button
              onClick={runExplainer}
              disabled={aiLoading}
              className={`px-3 py-1.5 rounded-lg text-sm font-extralight transition-colors ${
                aiLoading ? "bg-zinc-800 text-zinc-500" : "bg-zinc-800 hover:bg-zinc-700 text-zinc-200"
              }`}
            >
              {aiLoading ? "Explaining…" : "Run AI Explainer"}
            </button>
          </div>

          {aiExplainer ? (
            <pre className="text-xs bg-zinc-950 border border-zinc-800 rounded p-3 overflow-auto text-zinc-200 whitespace-pre-wrap">
              {aiExplainer}
            </pre>
          ) : null}
          {aiExplainerError ? <div className="text-xs text-red-400">{aiExplainerError}</div> : null}
        </section>

        <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
          <div className="flex items-center justify-between gap-4 mb-3">
            <div>
              <h2 className="text-xl font-thin text-white">MLflow</h2>
              <div className="text-xs text-zinc-600 font-extralight">
                Set <code className="text-zinc-400">NEXT_PUBLIC_MLFLOW_UI_URL</code> to embed your MLflow UI. If it blocks iframes, use “Open MLflow”.
              </div>
            </div>
            {MLFLOW_UI_URL ? (
              <a
                href={MLFLOW_UI_URL}
                target="_blank"
                rel="noreferrer"
                className="px-3 py-1.5 rounded-lg text-sm font-extralight bg-green-700 hover:bg-green-600 text-white transition-colors"
              >
                Open MLflow
              </a>
            ) : null}
          </div>

          {MLFLOW_UI_URL ? (
            <div className="border border-zinc-800 rounded-lg overflow-hidden bg-black">
              <iframe title="MLflow" src={MLFLOW_UI_URL} className="w-full h-[78vh]" />
            </div>
          ) : (
            <div className="text-sm text-zinc-400 font-extralight space-y-2">
              <div>Local (free) setup:</div>
              <pre className="text-xs bg-black border border-zinc-800 rounded p-3 overflow-auto">
                {"mlflow ui --backend-store-uri file:./mlruns --port 5000"}
              </pre>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

