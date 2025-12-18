import { queryMotherDuck } from "@/lib/md";
import { NextResponse } from "next/server";

export const runtime = "edge";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const perf = await queryMotherDuck(`
      SELECT as_of_date,
             model_name,
             horizon_code,
             rolling_mape_30d,
             rolling_rmse_30d,
             rolling_directional_accuracy_30d,
             is_degraded,
             degradation_reason
      FROM ops.model_performance
      WHERE as_of_date >= CURRENT_DATE - INTERVAL '30 days'
      ORDER BY as_of_date DESC, model_name, horizon_code
    `).catch(() => []);

    const registry = await queryMotherDuck(`
      SELECT model_id,
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
      LIMIT 20
    `).catch(() => []);

    return NextResponse.json({ success: true, perf, registry, ts: new Date().toISOString() });
  } catch (error: any) {
    return NextResponse.json({ success: false, error: error.message }, { status: 500 });
  }
}
