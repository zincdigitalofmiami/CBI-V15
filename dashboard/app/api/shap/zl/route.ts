import { queryMotherDuck } from '@/lib/md';
import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    // Placeholder query for SHAP values
    // Assuming a table exists or will exist
    const rows = await queryMotherDuck(`
      SELECT *
      FROM training.daily_ml_matrix_zl
      ORDER BY as_of_date DESC
      LIMIT 20
    `);

    return NextResponse.json({
      success: true,
      data: rows,
      count: Array.isArray(rows) ? rows.length : 0,
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    console.error('Database Error:', error);
    return NextResponse.json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
}
