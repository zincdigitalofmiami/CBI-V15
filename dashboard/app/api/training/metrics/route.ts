import { NextResponse } from "next/server";

export const runtime = "edge";
export const dynamic = "force-dynamic";

function hasMotherDuckToken(): boolean {
  return Boolean(
    process.env.MOTHERDUCK_TOKEN ||
      process.env.motherduck_storage_MOTHERDUCK_TOKEN ||
      process.env.MOTHERDUCK_STORAGE_TOKEN ||
      process.env.MOTHERDUCK_READ_SCALING_TOKEN ||
      process.env.motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN,
  );
}

async function safeQueryMotherDuck(sql: string): Promise<Record<string, unknown>[]> {
  try {
    const { queryMotherDuck } = await import("@/lib/md");
    return await queryMotherDuck(sql);
  } catch (error) {
    // MotherDuck WASM client may fail on Edge runtime (Worker not defined)
    console.error("MotherDuck query failed:", error instanceof Error ? error.message : error);
    return [];
  }
}

function asObject(v: unknown): Record<string, unknown> | null {
  return v && typeof v === "object" ? (v as Record<string, unknown>) : null;
}

function extractResponseText(payload: unknown): string {
  const obj = asObject(payload);
  if (!obj) return "";
  if (typeof obj["output_text"] === "string") return obj["output_text"];

  const output = obj["output"];
  if (Array.isArray(output)) {
    const texts: string[] = [];
    for (const item of output) {
      const itemObj = asObject(item);
      const content = itemObj ? itemObj["content"] : null;
      if (!Array.isArray(content)) continue;
      for (const c of content) {
        const cObj = asObject(c);
        if (!cObj) continue;
        if (cObj["type"] === "output_text" && typeof cObj["text"] === "string") texts.push(cObj["text"]);
        if (cObj["type"] === "text" && typeof cObj["text"] === "string") texts.push(cObj["text"]);
      }
    }
    if (texts.length) return texts.join("\n").trim();
  }
  return "";
}

async function generateExplainer(input: unknown): Promise<{ text: string; error?: string }> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return { text: "", error: "OPENAI_API_KEY not set" };

  const model = process.env.OPENAI_MODEL || "gpt-5.1-mini";
  const resp = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      max_output_tokens: 350,
      input: [
        {
          role: "system",
          content: [
            {
              type: "text",
              text: [
                "You are the internal Quant Admin explainer for CBI-V15.",
                "Given pipeline + training JSON, write:",
                "1) 5 bullets: current status",
                "2) 3 bullets: top risks",
                "3) 3 bullets: next actions",
                "Be terse. No hype. No hallucinations; if missing, say so.",
              ].join("\n"),
            },
          ],
        },
        {
          role: "user",
          content: [{ type: "text", text: JSON.stringify(input) }],
        },
      ],
    }),
  });

  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    return { text: "", error: `OpenAI error ${resp.status}: ${body.slice(0, 400)}` };
  }

  const payload = await resp.json();
  const text = extractResponseText(payload);
  if (!text) return { text: "", error: "Empty OpenAI response" };
  return { text };
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const explainRequested = searchParams.get("explain") === "1";

    if (!hasMotherDuckToken()) {
      return NextResponse.json(
        {
          success: false,
          error:
            "MotherDuck token is not defined in this Vercel environment. Set MOTHERDUCK_TOKEN (recommended) or motherduck_storage_MOTHERDUCK_TOKEN (MotherDuck integration), then redeploy.",
          has_motherduck_token: false,
          motherduck_db: process.env.MOTHERDUCK_DB || "cbi_v15",
          ts: new Date().toISOString(),
        },
        { status: 500 },
      );
    }

    const [matrixPreview, matrixStats, v15Stats, precondParams, dqOhlcvDaily, recentRuns, modelRegistry] =
      await Promise.all([
      safeQueryMotherDuck(`
        SELECT *
        FROM training.daily_ml_matrix_zl
        ORDER BY as_of_date DESC
        LIMIT 20
      `),
      safeQueryMotherDuck(`
        SELECT
          COUNT(*) AS row_count,
          MIN(as_of_date) AS min_date,
          MAX(as_of_date) AS max_date,
          COUNT(DISTINCT symbol) AS symbol_count
        FROM training.daily_ml_matrix_zl
      `),
      safeQueryMotherDuck(`
        SELECT
          COUNT(*) AS row_count,
          MIN(as_of_date) AS min_date,
          MAX(as_of_date) AS max_date,
          COUNT(DISTINCT symbol) AS symbol_count,
          MAX(updated_at) AS last_updated_at
        FROM training.daily_ml_matrix_zl_v15
      `),
      safeQueryMotherDuck(`
        SELECT COUNT(*) AS param_count, MAX(updated_at) AS last_updated_at
        FROM training.feature_preconditioning_params_zl
      `),
      safeQueryMotherDuck(`
        SELECT
          report_date,
          total_rows,
          null_ratio,
          duplicate_count,
          max_date,
          days_stale,
          gap_count,
          max_gap_days,
          mean_val,
          std_val,
          min_val,
          max_val,
          status,
          created_at
        FROM ops.data_quality_log
        WHERE schema_name = 'staging'
          AND table_name = 'ohlcv_daily'
        ORDER BY created_at DESC
        LIMIT 1
      `),
      safeQueryMotherDuck(`
        SELECT
          run_id,
          run_timestamp,
          model_tier,
          model_name,
          bucket,
          horizon_code,
          n_train_rows,
          n_val_rows,
          val_mape,
          val_rmse,
          val_directional_accuracy,
          training_time_seconds,
          status,
          error_message,
          artifact_uri
        FROM ops.training_runs
        ORDER BY run_timestamp DESC NULLS LAST, created_at DESC NULLS LAST
        LIMIT 20
      `),
      safeQueryMotherDuck(`
        SELECT
          model_id,
          model_tier,
          model_name,
          bucket,
          horizon_code,
          mape,
          directional_accuracy,
          coverage_90,
          ensemble_weight,
          is_active,
          version,
          artifact_uri,
          updated_at
        FROM reference.model_registry
        ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
        LIMIT 50
      `),
    ]);

    const ingestionSummary = await safeQueryMotherDuck(`
      SELECT source,
             max(completed_at) AS last_run,
             any_value(status) AS last_status,
             max(row_count) AS last_row_count
      FROM ops.ingestion_completion
      GROUP BY source
      ORDER BY source
    `);

    const columnCount = await safeQueryMotherDuck(`
      SELECT COUNT(*) AS column_count
      FROM information_schema.columns
      WHERE table_schema = 'training'
        AND table_name = 'daily_ml_matrix_zl'
    `);

    const base = {
      success: true,
      data: {
        matrix_preview: matrixPreview,
        matrix_stats: Array.isArray(matrixStats) ? matrixStats[0] : null,
        matrix_column_count: Array.isArray(columnCount) ? columnCount[0] : null,
        v15_matrix_stats: Array.isArray(v15Stats) ? v15Stats[0] : null,
        preconditioning_params: Array.isArray(precondParams) ? precondParams[0] : null,
        dq_ohlcv_daily: Array.isArray(dqOhlcvDaily) ? dqOhlcvDaily[0] : null,
        recent_runs: recentRuns,
        model_registry: modelRegistry,
        ingestion_summary: ingestionSummary,
      },
      timestamp: new Date().toISOString(),
    } as const;

    if (!explainRequested) {
      return NextResponse.json(base);
    }

    const explainerInput = {
      timestamp: base.timestamp,
      ingestion_summary: ingestionSummary,
      matrix_stats: base.data.matrix_stats,
      matrix_column_count: base.data.matrix_column_count,
      recent_runs: Array.isArray(recentRuns) ? recentRuns.slice(0, 5) : [],
      model_registry: Array.isArray(modelRegistry) ? modelRegistry.slice(0, 10) : [],
    };

    const explainer = await generateExplainer(explainerInput);
    return NextResponse.json({
      ...base,
      explanation: explainer.text || null,
      explanation_error: explainer.error || null,
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    console.error("Database Error:", message);
    return NextResponse.json(
      {
        success: false,
        error: message,
      },
      { status: 500 },
    );
  }
}
