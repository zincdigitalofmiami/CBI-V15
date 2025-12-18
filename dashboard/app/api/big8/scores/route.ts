import { queryMotherDuck } from "@/lib/md";
import { NextResponse } from "next/server";

export const runtime = "edge";
export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = Math.max(7, Math.min(3650, parseInt(searchParams.get("days") || "365")));

  try {
    const rows = await queryMotherDuck(`
      SELECT
        as_of_date,
        crush_bucket_score,
        china_bucket_score,
        fx_bucket_score,
        fed_bucket_score,
        tariff_bucket_score,
        biofuel_bucket_score,
        energy_bucket_score,
        volatility_bucket_score
      FROM features.big8_bucket_scores
      WHERE as_of_date >= CURRENT_DATE - INTERVAL '${days} days'
      ORDER BY as_of_date ASC
    `);

    return NextResponse.json({ success: true, days, data: rows, rows: rows.length, ts: new Date().toISOString() });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { success: false, error: message, days, data: [], rows: 0, ts: new Date().toISOString() },
      { status: 200 },
    );
  }
}

