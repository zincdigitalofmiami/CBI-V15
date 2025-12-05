import { BigQuery } from '@google-cloud/bigquery';
import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET() {
  try {
    const bigquery = new BigQuery({
      projectId: 'cbi-v15',
    });

    // Full ZL OHLCV history (limited by table) from raw.databento_futures_ohlcv_1d.
    // We keep several years so 1W/1M/3M/6M/YTD/1Y stats are real, not placeholders.
    const query = `
      SELECT
        date,
        open,
        high,
        low,
        close,
        volume
      FROM \`cbi-v15.raw.databento_futures_ohlcv_1d\`
      WHERE symbol = 'ZL'
      ORDER BY date ASC
    `;

    const [rows] = await bigquery.query(query);

    // rows.date comes back as BigQueryDate { value: 'YYYY-MM-DD' }, which matches existing front-end expectations (d.date.value)
    const data = rows.map((row: any) => ({
      date: row.date,
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.close,
      volume: row.volume,
      symbol: 'ZL',
    }));

    return NextResponse.json({
      success: true,
      data,
      count: data.length,
      symbol: 'ZL',
      source: 'BigQuery raw.databento_futures_ohlcv_1d',
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('live/zl API error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 },
    );
  }
}
