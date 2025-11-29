import { BigQuery } from '@google-cloud/bigquery';
import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET() {
  try {
    const bigquery = new BigQuery({
      projectId: 'cbi-v15'
    });

    // REAL DATA ONLY - Last 90 days of ZL prices from Databento
    const query = `
      SELECT 
        date,
        open,
        high,
        low,
        close,
        volume,
        symbol
      FROM \`cbi-v15.raw.databento_futures_ohlcv_1d\`
      WHERE symbol = 'ZL'
      ORDER BY date DESC
      LIMIT 90
    `;

    const [rows] = await bigquery.query(query);

    return NextResponse.json({ 
      success: true,
      data: rows.reverse(), // Oldest to newest for chart
      count: rows.length,
      symbol: 'ZL',
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    console.error('BigQuery API error:', error);
    return NextResponse.json({ 
      success: false, 
      error: error.message,
      message: 'BigQuery credentials not configured in Vercel. Add GCP_PROJECT_ID and service account credentials in Vercel Environment Variables.'
    }, { status: 500 });
  }
}
