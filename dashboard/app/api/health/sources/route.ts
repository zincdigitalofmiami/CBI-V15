import { queryMotherDuck } from "@/lib/md";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const rows = await queryMotherDuck(`
      SELECT source,
             max(completed_at) AS last_run,
             any_value(status) AS last_status,
             max(row_count) AS last_row_count
      FROM ops.ingestion_completion
      GROUP BY source
      ORDER BY source
    `).catch(() => []);

    return NextResponse.json({ success: true, data: rows, ts: new Date().toISOString() });
  } catch (error: any) {
    return NextResponse.json({ success: false, error: error.message }, { status: 500 });
  }
}
