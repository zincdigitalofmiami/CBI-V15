"use client";

import BandsChart from "@/components/charts/BandsChart";
import CompareMultipleSeries from "@/components/charts/CompareMultipleSeries";
import RangeSwitcherChart, { type RangeKey } from "@/components/charts/RangeSwitcherChart";
import StackedAreaChart from "@/components/charts/StackedAreaChart";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

const MLFLOW_UI_URL = process.env.NEXT_PUBLIC_MLFLOW_UI_URL || "";

type IngestionSourceRow = {
  source?: string;
  last_run?: unknown;
  last_status?: string;
  last_row_count?: number;
};

type TrainingRunRow = {
  run_id?: string;
  run_timestamp?: unknown;
  model_tier?: string;
  model_name?: string;
  bucket?: string;
  horizon_code?: string;
  status?: string;
  val_rmse?: number;
  val_directional_accuracy?: number;
  training_time_seconds?: number;
};

type ModelRegistryRow = {
  model_id?: string;
  model_tier?: string;
  model_name?: string;
  bucket?: string;
  horizon_code?: string;
  is_active?: boolean;
  version?: number;
  updated_at?: unknown;
};

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

export default function QuantAdminPage() {
  const router = useRouter();
  const [tab, setTab] = useState<"overview" | "anofox" | "mlflow">("overview");
  const [pipelineStages, setPipelineStages] = useState<
    { name: string; status: "ok" | "error" | "pending"; lastRun: string; rows: string }[]
  >([
    { name: "Databento", status: "pending", lastRun: "—", rows: "—" },
    { name: "FRED", status: "pending", lastRun: "—", rows: "—" },
    { name: "CFTC", status: "pending", lastRun: "—", rows: "—" },
    { name: "USDA", status: "pending", lastRun: "—", rows: "—" },
    { name: "News/Sentiment", status: "pending", lastRun: "—", rows: "—" },
    { name: "EPA RIN", status: "pending", lastRun: "—", rows: "—" },
  ]);

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

  const [multi, setMulti] = useState<Record<string, { time: number; value: number }[]>>({});
  const [loadingCharts, setLoadingCharts] = useState(true);
  const [big8Scores, setBig8Scores] = useState<Record<string, unknown>[]>([]);
  const [recentRuns, setRecentRuns] = useState<TrainingRunRow[]>([]);
  const [modelRegistry, setModelRegistry] = useState<ModelRegistryRow[]>([]);
  const [matrixMeta, setMatrixMeta] = useState<{ row_count?: number; min_date?: string; max_date?: string; column_count?: number }>(
    {},
  );
  const [v15Meta, setV15Meta] = useState<{ row_count?: number; min_date?: string; max_date?: string; last_updated_at?: string }>(
    {},
  );
  const [precondParams, setPrecondParams] = useState<{ param_count?: number; last_updated_at?: string }>({});
  const [dqOhlcvDaily, setDqOhlcvDaily] = useState<DqOhlcvDailyRow | null>(null);
  const [lastTraining, setLastTraining] = useState<string>("—");
  const [aiExplainer, setAiExplainer] = useState<string>("");
  const [aiExplainerError, setAiExplainerError] = useState<string>("");
  const [aiLoading, setAiLoading] = useState(false);

  useEffect(() => {
    const sp = new URLSearchParams(window.location.search);
    const t = String(sp.get("tab") || "").toLowerCase();
    if (t === "mlflow" || t === "anofox" || t === "overview") setTab(t as "overview" | "anofox" | "mlflow");
  }, []);

  function setTabAndUrl(nextTab: "overview" | "anofox" | "mlflow") {
    setTab(nextTab);
    const sp = new URLSearchParams(window.location.search);
    sp.set("tab", nextTab);
    router.push(`?${sp.toString()}`);
  }

  useEffect(() => {
    async function fetchCharts() {
      try {
        setLoadingCharts(true);
        const res = await fetch("/api/live/multi?symbols=ZL,ZS,ZM&days=365");
        const json = await res.json();
        if (json?.success && json?.data) setMulti(json.data);

        const bRes = await fetch("/api/big8/scores?days=365");
        const bJson = await bRes.json();
        if (bJson?.success && Array.isArray(bJson?.data)) setBig8Scores(bJson.data);
      } catch {
        // ignore; chart sections handle empties
      } finally {
        setLoadingCharts(false);
      }
    }
    fetchCharts();
  }, []);

  useEffect(() => {
    async function fetchOps() {
      try {
        const [healthRes, trainingRes] = await Promise.all([
          fetch("/api/health/sources").then((r) => r.json()).catch(() => null),
          fetch("/api/training/metrics").then((r) => r.json()).catch(() => null),
        ]);

        const healthRows: IngestionSourceRow[] = Array.isArray(healthRes?.data) ? healthRes.data : [];
        if (healthRows.length > 0) {
          const by = new Map(healthRows.map((r) => [String(r.source || ""), r]));
          const mapStage = (label: string, keys: string[]) => {
            const hit = keys.map((k) => by.get(k)).find(Boolean);
            const statusRaw = String(hit?.last_status || "").toLowerCase();
            const status: "ok" | "error" | "pending" =
              statusRaw === "success" ? "ok" : statusRaw === "failed" ? "error" : "pending";
            const lastRunIso = asIsoString(hit?.last_run);
            const lastRun = lastRunIso && lastRunIso !== "null" ? lastRunIso.replace("T", " ").slice(0, 19) : "—";
            const rows = typeof hit?.last_row_count === "number" ? hit!.last_row_count!.toLocaleString() : "—";
            return { name: label, status, lastRun, rows };
          };

          setPipelineStages([
            mapStage("Databento", ["databento", "Databento"]),
            mapStage("FRED", ["fred", "FRED"]),
            mapStage("CFTC", ["cftc", "CFTC", "cftc_cot"]),
            mapStage("USDA", ["usda", "USDA"]),
            mapStage("News/Sentiment", ["scrapecreators", "news", "sentiment"]),
            mapStage("EPA RIN", ["epa", "rin", "epa_rin"]),
          ]);
        }

        const data = trainingRes?.data;
        const runs: TrainingRunRow[] = Array.isArray(data?.recent_runs) ? data.recent_runs : [];
        const registry: ModelRegistryRow[] = Array.isArray(data?.model_registry) ? data.model_registry : [];
        setRecentRuns(runs);
        setModelRegistry(registry);

        const stats = data?.matrix_stats || {};
        const colCount = data?.matrix_column_count || {};
        const v15Stats = data?.v15_matrix_stats || {};
        const pre = data?.preconditioning_params || {};
        const dq = data?.dq_ohlcv_daily || null;
        setDqOhlcvDaily(dq);

        const row_count = typeof stats?.row_count === "number" ? stats.row_count : Number(stats?.row_count);
        const min_date = asIsoString(stats?.min_date);
        const max_date = asIsoString(stats?.max_date);
        const column_count =
          typeof colCount?.column_count === "number" ? colCount.column_count : Number(colCount?.column_count);
        setMatrixMeta({
          row_count: Number.isFinite(row_count) ? row_count : undefined,
          min_date: min_date && min_date !== "null" ? min_date : undefined,
          max_date: max_date && max_date !== "null" ? max_date : undefined,
          column_count: Number.isFinite(column_count) ? column_count : undefined,
        });

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

        const latest = runs[0]?.run_timestamp ?? registry[0]?.updated_at;
        const latestIso = asIsoString(latest);
        if (latestIso && latestIso !== "null") {
          setLastTraining(latestIso.replace("T", " ").slice(0, 19));
        }
      } catch {
        // ignore
      }
    }
    fetchOps();
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

  const compareSeries = useMemo(() => {
    const mk = (id: string, color?: string) => ({
      id,
      name: id,
      color,
      data: Array.isArray(multi?.[id]) ? multi[id] : [],
    });
    return [mk("ZL", "#2962FF"), mk("ZS", "rgb(225, 87, 90)"), mk("ZM", "rgb(242, 142, 44)")].filter(
      (s) => s.data.length > 0,
    );
  }, [multi]);

  const rangeSeries = useMemo(() => {
    const base = (Array.isArray(multi?.ZL) ? multi.ZL : []).slice();
    const now = base.length > 0 ? base[base.length - 1].time : Math.floor(Date.now() / 1000);
    const filterSince = (days: number) => base.filter((p) => p.time >= now - days * 86_400);
    const out: Record<RangeKey, { time: number; value: number }[]> = {
      "1D": filterSince(1),
      "1W": filterSince(7),
      "1M": filterSince(30),
      "1Y": filterSince(365),
    };
    return out;
  }, [multi]);

  const stackedPoints = useMemo(() => {
    // Primary: pull real bucket scores from features.big8_bucket_scores via /api/big8/scores
    if (big8Scores.length > 0) {
      return big8Scores
        .map((r) => {
          const rec = r as Record<string, unknown>;
          const asOf = rec["as_of_date"];
          const dt =
            typeof asOf === "object" && asOf !== null && "value" in asOf
              ? String((asOf as { value?: unknown }).value ?? "")
              : String(asOf ?? "");
          const time = Math.floor(new Date(dt).getTime() / 1000);
          if (!Number.isFinite(time)) return null;
          const v = (k: string) => {
            const n = Number(rec[k]);
            return Number.isFinite(n) ? Math.max(0, Math.min(100, n)) / 100 : 0;
          };
          return {
            time,
            values: {
              crush: v("crush_bucket_score"),
              china: v("china_bucket_score"),
              fx: v("fx_bucket_score"),
              fed: v("fed_bucket_score"),
              tariff: v("tariff_bucket_score"),
              biofuel: v("biofuel_bucket_score"),
              energy: v("energy_bucket_score"),
              volatility: v("volatility_bucket_score"),
            },
          };
        })
        .filter(Boolean) as { time: number; values: Record<string, number> }[];
    }

    // Fallback: normalize series into 0..1 contributions for display.
    const zl = Array.isArray(multi?.ZL) ? multi.ZL : [];
    const zs = Array.isArray(multi?.ZS) ? multi.ZS : [];
    const zm = Array.isArray(multi?.ZM) ? multi.ZM : [];
    const n = Math.min(zl.length, zs.length, zm.length);
    const pts = [];
    for (let i = 0; i < n; i++) {
      const a = zl[i]?.value ?? 0;
      const b = zs[i]?.value ?? 0;
      const c = zm[i]?.value ?? 0;
      const s = a + b + c || 1;
      pts.push({
        time: zl[i].time,
        values: { crush: a / s, china: b / s, fx: c / s },
      });
    }
    return pts;
  }, [multi, big8Scores]);

  return (
    <main className="min-h-screen bg-[#070a12] px-10 py-10">
      <div className="max-w-screen-2xl mx-auto">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center justify-between gap-3 mb-2">
            <div className="flex items-center gap-3">
              <div className="h-9 w-9 rounded-xl bg-white/5 border border-white/10" />
              <h1 className="text-3xl md:text-4xl font-thin text-white tracking-wide">Quant Admin</h1>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setTabAndUrl("overview")}
                className={`px-3 py-1.5 rounded-lg text-sm font-extralight border transition-colors ${
                  tab === "overview"
                    ? "bg-white/10 text-white border-white/20"
                    : "bg-black/20 text-zinc-400 border-white/10 hover:border-white/20"
                }`}
              >
                Overview
              </button>
              <button
                onClick={() => setTabAndUrl("anofox")}
                className={`px-3 py-1.5 rounded-lg text-sm font-extralight border transition-colors ${
                  tab === "anofox"
                    ? "bg-white/10 text-white border-white/20"
                    : "bg-black/20 text-zinc-400 border-white/10 hover:border-white/20"
                }`}
              >
                AnoFox
              </button>
              <button
                onClick={() => setTabAndUrl("mlflow")}
                className={`px-3 py-1.5 rounded-lg text-sm font-extralight border transition-colors ${
                  tab === "mlflow"
                    ? "bg-white/10 text-white border-white/20"
                    : "bg-black/20 text-zinc-400 border-white/10 hover:border-white/20"
                }`}
              >
                MLflow
              </button>
            </div>
          </div>
          <p className="text-zinc-400 font-extralight">
            Pipeline health, data coverage, and model registry
          </p>
        </header>

        {tab === "anofox" ? (
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6 space-y-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-xl font-thin text-white">AnoFox</h2>
                <div className="text-xs text-zinc-600 font-extralight">
                  SQL-first data preparation + guardrails (gap-fill, data-quality logging, robust preconditioning).
                </div>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href="https://anofox.com/docs/forecast/guides/basic-workflow"
                  target="_blank"
                  rel="noreferrer"
                  className="px-3 py-1.5 rounded-lg text-sm font-extralight bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors"
                >
                  AnoFox Workflow
                </a>
                <a
                  href="https://anofox.com/docs/forecast/guides/data-preparation"
                  target="_blank"
                  rel="noreferrer"
                  className="px-3 py-1.5 rounded-lg text-sm font-extralight bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors"
                >
                  Data Prep Macros
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
            </div>

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
                      String(dqOhlcvDaily?.status || "").toUpperCase() === "PASSED"
                        ? "bg-green-900/30 text-green-400"
                        : String(dqOhlcvDaily?.status || "").toUpperCase() === "FAILED"
                          ? "bg-red-900/30 text-red-400"
                          : "bg-yellow-900/30 text-yellow-300"
                    }`}
                  >
                    {String(dqOhlcvDaily?.status || "—").toUpperCase()}
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

            <div className="bg-black border border-zinc-800 rounded-lg p-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <div className="text-zinc-300 font-extralight">Controls</div>
                  <div className="text-xs text-zinc-600 font-extralight">Reads from MotherDuck; training/ingestion runs happen outside Vercel.</div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setTabAndUrl("overview")}
                    className="px-3 py-1.5 rounded-lg text-sm font-extralight bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors"
                  >
                    Go to Overview
                  </button>
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
              </div>
              {aiExplainer ? (
                <pre className="mt-3 text-xs bg-zinc-950 border border-zinc-800 rounded p-3 overflow-auto text-zinc-200 whitespace-pre-wrap">
                  {aiExplainer}
                </pre>
              ) : null}
              {aiExplainerError ? <div className="mt-2 text-xs text-red-400">{aiExplainerError}</div> : null}
              <div className="mt-3 text-xs text-zinc-600 font-extralight">
                Local commands:
                <div className="mt-1 grid grid-cols-1 lg:grid-cols-2 gap-2">
                  <pre className="text-[11px] bg-zinc-950 border border-zinc-800 rounded p-2 overflow-auto">
                    {"python src/engines/anofox/build_all_features.py"}
                  </pre>
                  <pre className="text-[11px] bg-zinc-950 border border-zinc-800 rounded p-2 overflow-auto">
                    {"python src/engines/anofox/build_training.py"}
                  </pre>
                </div>
              </div>
            </div>
          </section>
        ) : null}

        {tab === "mlflow" ? (
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <div className="flex items-center justify-between gap-4 mb-3">
              <div>
                <h2 className="text-xl font-thin text-white">MLflow</h2>
                <div className="text-xs text-zinc-600 font-extralight">
                  Set `NEXT_PUBLIC_MLFLOW_UI_URL` to embed a hosted MLflow UI. If it blocks iframes, use “Open MLflow”.
                </div>
              </div>
              {MLFLOW_UI_URL ? (
                <div className="flex items-center gap-2">
                  <a
                    href="https://github.com/mlflow/mlflow"
                    target="_blank"
                    rel="noreferrer"
                    className="px-3 py-1.5 rounded-lg text-sm font-extralight bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors"
                  >
                    MLflow Repo
                  </a>
                  <a
                    href={MLFLOW_UI_URL}
                    target="_blank"
                    rel="noreferrer"
                    className="px-3 py-1.5 rounded-lg text-sm font-extralight bg-green-700 hover:bg-green-600 text-white transition-colors"
                  >
                    Open MLflow
                  </a>
                </div>
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
                <div className="text-xs text-zinc-600">
                  Then set `NEXT_PUBLIC_MLFLOW_UI_URL=http://localhost:5000` for local dev, or point it at a hosted MLflow
                  server.
                </div>
              </div>
            )}
          </section>
        ) : null}

        {tab === "overview" ? (
          <>
            {/* AI Explainer */}
            <section className="mb-6 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <div className="text-zinc-300 font-extralight">AI Explainer</div>
                  <div className="text-xs text-zinc-600 font-extralight">
                    Summarizes ingestion + training state from MotherDuck (on-demand).
                  </div>
                </div>
                <button
                  onClick={runExplainer}
                  disabled={aiLoading}
                  className={`px-3 py-1.5 rounded-lg text-sm font-extralight transition-colors ${
                    aiLoading ? "bg-zinc-800 text-zinc-500" : "bg-zinc-800 hover:bg-zinc-700 text-zinc-200"
                  }`}
                >
                  {aiLoading ? "Explaining…" : "Explain"}
                </button>
              </div>
              {aiExplainerError ? (
                <div className="mt-3 text-xs text-red-400 font-extralight">{aiExplainerError}</div>
              ) : null}
              {aiExplainer ? (
                <pre className="mt-3 whitespace-pre-wrap text-xs text-zinc-300 font-extralight bg-black border border-zinc-800 rounded p-3 overflow-auto">
                  {aiExplainer}
                </pre>
              ) : null}
            </section>

            {/* Status Overview */}
            <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-5 text-center backdrop-blur-md">
            <div className="text-3xl font-thin text-green-400">
              {pipelineStages.filter((s) => s.status === "ok").length}
            </div>
            <div className="text-zinc-500 text-sm font-extralight">Pipelines OK</div>
          </div>
          <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-5 text-center backdrop-blur-md">
            <div className="text-3xl font-thin text-yellow-400">
              {pipelineStages.filter((s) => s.status !== "ok").length}
            </div>
            <div className="text-zinc-500 text-sm font-extralight">Pending</div>
          </div>
          <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-5 text-center backdrop-blur-md">
            <div className="text-3xl font-thin text-white">
              {typeof matrixMeta.column_count === "number" ? matrixMeta.column_count : "—"}
            </div>
            <div className="text-zinc-500 text-sm font-extralight">Total Features</div>
          </div>
          <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-5 text-center backdrop-blur-md">
            <div className="text-3xl font-thin text-white">{lastTraining}</div>
            <div className="text-zinc-500 text-sm font-extralight">Last Training</div>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Compare Multiple Series */}
          <section className="lg:col-span-2 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-thin text-white">Compare Multiple Series (ZL vs ZS vs ZM)</h2>
              <span className="text-xs text-zinc-600 font-extralight">
                {loadingCharts ? "Loading…" : "Last 1Y (daily)"}
              </span>
            </div>
            {compareSeries.length > 0 ? (
              <CompareMultipleSeries series={compareSeries} height={360} />
            ) : (
              <div className="h-[360px] flex items-center justify-center text-zinc-600 font-extralight">
                No series data available
              </div>
            )}
          </section>

          {/* Range Switcher */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Range Switcher (ZL)</h2>
            <RangeSwitcherChart seriesByRange={rangeSeries} height={320} defaultRange="1M" />
          </section>

          {/* Bands Indicator */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-2">Bands Indicator (ZL)</h2>
            <div className="text-xs text-zinc-600 font-extralight mb-4">
              UI bands (±2%) now; will swap to volatility-derived σ bands once available.
            </div>
            <BandsChart data={Array.isArray(multi?.ZL) ? multi.ZL : []} bandPct={0.02} height={320} />
          </section>

          {/* Stacked Area */}
          <section className="lg:col-span-2 bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-2">Stacked Area (Driver Composition)</h2>
            <div className="text-xs text-zinc-600 font-extralight mb-4">
              Uses `features.big8_bucket_scores` when available; falls back to a placeholder composition.
            </div>
            <StackedAreaChart
              points={stackedPoints}
              keysInOrder={[
                { id: "crush", name: "Crush", color: "#2962FF" },
                { id: "china", name: "China", color: "rgb(225, 87, 90)" },
                { id: "fx", name: "FX", color: "rgb(242, 142, 44)" },
                { id: "fed", name: "Fed", color: "rgb(164, 89, 209)" },
                { id: "tariff", name: "Tariff", color: "rgb(59, 130, 246)" },
                { id: "biofuel", name: "Biofuel", color: "rgb(34, 197, 94)" },
                { id: "energy", name: "Energy", color: "rgb(245, 158, 11)" },
                { id: "volatility", name: "Volatility", color: "rgb(239, 68, 68)" },
              ]}
              height={320}
            />
          </section>

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
              {(modelRegistry.length > 0
                ? modelRegistry.slice(0, 8).map((r) => ({
                    name: `${r.model_name || "model"}${r.horizon_code ? ` (${r.horizon_code})` : ""}`,
                    type: r.model_tier || "—",
                    status: r.is_active ? "active" : "pending",
                  }))
                : models
              ).map((model) => (
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
              Run Training
            </button>
          </section>

          {/* Recent Runs */}
          <section className="bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm rounded-lg p-6">
            <h2 className="text-xl font-thin text-white mb-4">Recent Runs</h2>
            <div className="space-y-2">
              {(recentRuns.length > 0 ? recentRuns.slice(0, 6) : [1, 2, 3, 4]).map((r, i) => (
                <div
                  key={typeof r === "number" ? r : r.run_id || String(i)}
                  className="flex items-center justify-between bg-black border border-zinc-800 rounded-lg p-3"
                >
                  <div>
                    <div className="text-zinc-300 font-extralight text-sm">
                      {typeof r === "number"
                        ? "—"
                        : `${r.model_name || "model"}${r.horizon_code ? ` (${r.horizon_code})` : ""}`}
                    </div>
                    <div className="text-zinc-600 text-xs font-extralight">
                      {typeof r === "number" ? "—" : r.status || "—"}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-zinc-400 text-sm">
                      {typeof r === "number"
                        ? "—"
                        : Number.isFinite(Number(r.val_rmse))
                          ? `RMSE ${Number(r.val_rmse).toFixed(4)}`
                          : "—"}
                    </div>
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
                <div className="text-2xl font-thin text-white">
                  {typeof matrixMeta.column_count === "number" ? matrixMeta.column_count : "—"}
                </div>
                <div className="text-zinc-500 text-xs font-extralight">Total Features</div>
              </div>
              <div className="bg-black border border-zinc-800 rounded-lg p-4 text-center">
                <div className="text-2xl font-thin text-white">
                  {matrixMeta.min_date && matrixMeta.max_date
                    ? `${matrixMeta.min_date} → ${matrixMeta.max_date}`
                    : "—"}
                </div>
                <div className="text-zinc-500 text-xs font-extralight">Date Range</div>
              </div>
              <div className="bg-black border border-zinc-800 rounded-lg p-4 text-center">
                <div className="text-2xl font-thin text-green-400">
                  {typeof matrixMeta.row_count === "number" ? matrixMeta.row_count.toLocaleString() : "—"}
                </div>
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
          </>
        ) : null}
      </div>
    </main>
  );
}
