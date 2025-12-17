import { queryMotherDuck } from "@/lib/md";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const perf = await queryMotherDuck(`
      SELECT model_name,
             horizon,
             mae,
             mape,
             coverage_p90,
             as_of_date
      FROM ops.model_performance
      WHERE as_of_date >= CURRENT_DATE - INTERVAL '30 days'
      ORDER BY as_of_date DESC, model_name, horizon
    `).catch(() => []);

    const registry = await queryMotherDuck(`
      SELECT model_name, version, trained_at, notes
      FROM reference.model_registry
      ORDER BY trained_at DESC
      LIMIT 20
    `).catch(() => []);

    return NextResponse.json({ success: true, perf, registry, ts: new Date().toISOString() });
  } catch (error: any) {
    return NextResponse.json({ success: false, error: error.message }, { status: 500 });
  }
}
