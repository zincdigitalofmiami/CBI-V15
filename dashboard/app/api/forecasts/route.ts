import { BigQuery } from '@google-cloud/bigquery';
import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const bigquery = new BigQuery({
      projectId: 'cbi-v15'
    });

    const query = `
      SELECT 
        date,
        close as predicted_price,
        symbol,
        'historical' as model_type
      FROM \`cbi-v15.raw.databento_futures_ohlcv_1d\`
      WHERE symbol = 'ZL'
      ORDER BY date DESC
      LIMIT 90
    `;

    const [rows] = await bigquery.query(query);

    return NextResponse.json({ 
      success: true,
      data: rows,
      count: rows.length,
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    console.error('BigQuery API error:', error);
    return NextResponse.json({ 
      success: false, 
      error: error.message 
    }, { status: 500 });
  }
}

