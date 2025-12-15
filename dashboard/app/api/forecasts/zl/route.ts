import { queryMotherDuck } from '@/lib/md';
import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET() {
  try {
    // Get latest as_of_date available
    const latestRows = await queryMotherDuck(`
      SELECT max(as_of_date) as latest_as_of
      FROM forecasts.zl_predictions
    `);

    const latest = Array.isArray(latestRows) && latestRows[0]?.latest_as_of;

    const rows = await queryMotherDuck(`
      SELECT as_of_date,
             horizon_code,
             target_date,
             q10, q25, q50, q75, q90,
             direction, direction_probability,
             prediction_confidence, model_agreement,
             regime, model_version
      FROM forecasts.zl_predictions
      ${latest ? `WHERE as_of_date = '${latest}'` : ''}
      AND horizon_code IN ('1w','1m','3m','6m')
      ORDER BY target_date ASC
    `);

    return NextResponse.json({
      success: true,
      as_of_date: latest ?? null,
      data: rows,
      count: Array.isArray(rows) ? rows.length : 0,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('Forecasts API Error:', error);
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}
