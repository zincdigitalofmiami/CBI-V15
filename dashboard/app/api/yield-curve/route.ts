import { queryMotherDuck } from "@/lib/md";
import { NextResponse } from "next/server";

export const runtime = "edge";
export const dynamic = "force-dynamic";
export const revalidate = 0;

type CurvePoint = { maturityMonths: number; value: number };

const SERIES: { maturityMonths: number; series_id: string }[] = [
  { maturityMonths: 1, series_id: "DGS1MO" },
  { maturityMonths: 3, series_id: "DGS3MO" },
  { maturityMonths: 6, series_id: "DGS6MO" },
  { maturityMonths: 12, series_id: "DGS1" },
  { maturityMonths: 24, series_id: "DGS2" },
  { maturityMonths: 36, series_id: "DGS3" },
  { maturityMonths: 60, series_id: "DGS5" },
  { maturityMonths: 84, series_id: "DGS7" },
  { maturityMonths: 120, series_id: "DGS10" },
  { maturityMonths: 240, series_id: "DGS20" },
  { maturityMonths: 360, series_id: "DGS30" },
];

function demoCurves(): { current: CurvePoint[]; previous: CurvePoint[] } {
  // Minimal fallback resembling the TradingView demo
  const current: CurvePoint[] = [
    { maturityMonths: 1, value: 5.378 },
    { maturityMonths: 2, value: 5.372 },
    { maturityMonths: 3, value: 5.271 },
    { maturityMonths: 6, value: 5.094 },
    { maturityMonths: 12, value: 4.739 },
    { maturityMonths: 24, value: 4.237 },
    { maturityMonths: 36, value: 4.036 },
    { maturityMonths: 60, value: 3.887 },
    { maturityMonths: 84, value: 3.921 },
    { maturityMonths: 120, value: 4.007 },
    { maturityMonths: 240, value: 4.366 },
    { maturityMonths: 360, value: 4.29 },
  ];
  const previous: CurvePoint[] = current.map((p) => ({ ...p, value: p.value + (Math.random() - 0.5) * 0.1 }));
  return { current, previous };
}

function asDateString(v: unknown): string | null {
  if (typeof v === "string" && v) return v;
  if (typeof v === "object" && v !== null && "value" in v) {
    const vv = (v as { value?: unknown }).value;
    if (typeof vv === "string" && vv) return vv;
  }
  return null;
}

export async function GET() {
  try {
    const seriesIn = SERIES.map((s) => `'${s.series_id}'`).join(", ");

    // Use the latest available date for each series_id; then build curves for latest and previous date.
    const latestDateRows = await queryMotherDuck(`
      SELECT MAX(date) AS max_date
      FROM raw.fred_economic
      WHERE series_id IN (${seriesIn})
    `);

    const maxDate = asDateString((latestDateRows?.[0] as Record<string, unknown> | undefined)?.["max_date"]);
    if (!maxDate) {
      const curves = demoCurves();
      return NextResponse.json({
        success: true,
        source: "fallback",
        curves,
        ts: new Date().toISOString(),
      });
    }

    const rows = await queryMotherDuck(`
      WITH dates AS (
        SELECT
          DATE '${maxDate}' AS d0,
          DATE '${maxDate}' - INTERVAL '1 day' AS d1
      )
      SELECT
        e.series_id,
        e.date,
        e.value
      FROM raw.fred_economic e
      JOIN dates d ON e.date IN (d.d0, d.d1)
      WHERE e.series_id IN (${seriesIn})
    `);

    const byIdByDate: Record<string, Record<string, number>> = {};
    for (const r of rows as Record<string, unknown>[]) {
      const series_id = String(r["series_id"]);
      const dt = asDateString(r["date"]) || "";
      const value = Number(r["value"]);
      if (!Number.isFinite(value)) continue;
      if (!byIdByDate[series_id]) byIdByDate[series_id] = {};
      byIdByDate[series_id][dt] = value;
    }

    const d0 = maxDate;
    // best-effort previous date string: pick any other date in rows
    const d1 =
      Object.values(byIdByDate)
        .flatMap((m) => Object.keys(m))
        .filter((d) => d !== d0)
        .sort()
        .slice(-1)[0] || d0;

    const current: CurvePoint[] = [];
    const previous: CurvePoint[] = [];
    for (const s of SERIES) {
      const v0 = byIdByDate[s.series_id]?.[d0];
      const v1 = byIdByDate[s.series_id]?.[d1];
      if (typeof v0 === "number") current.push({ maturityMonths: s.maturityMonths, value: v0 });
      if (typeof v1 === "number") previous.push({ maturityMonths: s.maturityMonths, value: v1 });
    }

    if (current.length === 0) {
      const curves = demoCurves();
      return NextResponse.json({
        success: true,
        source: "fallback",
        curves,
        ts: new Date().toISOString(),
      });
    }

    return NextResponse.json({
      success: true,
      source: "motherduck",
      asOfDate: d0,
      previousDate: d1,
      curves: { current, previous },
      ts: new Date().toISOString(),
    });
  } catch (error: unknown) {
    const curves = demoCurves();
    const warning = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { success: true, source: "fallback", curves, warning, ts: new Date().toISOString() },
      { status: 200 },
    );
  }
}

