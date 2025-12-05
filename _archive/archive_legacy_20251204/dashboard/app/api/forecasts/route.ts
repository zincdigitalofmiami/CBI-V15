import { BigQuery } from '@google-cloud/bigquery';
import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET() {
  try {
    const bigquery = new BigQuery({
      projectId: 'cbi-v15',
    });

    // Pull latest 180 days of ZL closes from raw OHLCV
    const historyQuery = `
      SELECT 
        date,
        close AS close_price
      FROM \`cbi-v15.raw.databento_futures_ohlcv_1d\`
      WHERE symbol = 'ZL'
      ORDER BY date DESC
      LIMIT 180
    `;

    // Pull most recent 1m horizon forecast + bands from predictions table
    const forecastQuery = `
      SELECT
        date,
        prediction,
        p10,
        p50,
        p90
      FROM \`cbi-v15.predictions.zl_predictions_1m\`
      WHERE horizon = '1m'
      ORDER BY as_of DESC, date ASC
      LIMIT 180
    `;

    const [[historyRows], [forecastRows]] = await Promise.all([
      bigquery.query(historyQuery),
      bigquery.query(forecastQuery),
    ]);

    return NextResponse.json({
      success: true,
      data: {
        history: historyRows,
        forecasts: forecastRows,
      },
      historyCount: historyRows.length,
      forecastCount: forecastRows.length,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('BigQuery forecasts API error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 },
    );
  }
}
