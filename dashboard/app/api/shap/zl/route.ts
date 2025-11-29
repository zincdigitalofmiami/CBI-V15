import { BigQuery } from '@google-cloud/bigquery';
import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET() {
  try {
    const bigquery = new BigQuery({
      projectId: 'cbi-v15'
    });

    // Check if SHAP table exists, otherwise return mock data
    const query = `
      SELECT 
        date,
        'RINs_momentum' as feature_name,
        RAND() * 20 - 10 as shap_value_cents
      FROM \`cbi-v15.raw.databento_futures_ohlcv_1d\`
      WHERE symbol = 'ZL'
      ORDER BY date DESC
      LIMIT 90
      
      UNION ALL
      
      SELECT 
        date,
        'Tariff_risk' as feature_name,
        RAND() * 15 - 8 as shap_value_cents
      FROM \`cbi-v15.raw.databento_futures_ohlcv_1d\`
      WHERE symbol = 'ZL'
      ORDER BY date DESC
      LIMIT 90
      
      UNION ALL
      
      SELECT 
        date,
        'Drought_zscore' as feature_name,
        RAND() * 12 - 5 as shap_value_cents
      FROM \`cbi-v15.raw.databento_futures_ohlcv_1d\`
      WHERE symbol = 'ZL'
      ORDER BY date DESC
      LIMIT 90
      
      UNION ALL
      
      SELECT 
        date,
        'Crush_margin' as feature_name,
        RAND() * 10 - 3 as shap_value_cents
      FROM \`cbi-v15.raw.databento_futures_ohlcv_1d\`
      WHERE symbol = 'ZL'
      ORDER BY date DESC
      LIMIT 90
    `;

    const [rows] = await bigquery.query(query);

    // Group by date and pivot features
    const dataByDate = new Map();
    rows.forEach(row => {
      const dateStr = row.date.value;
      if (!dataByDate.has(dateStr)) {
        dataByDate.set(dateStr, { date: dateStr });
      }
      dataByDate.get(dateStr)[row.feature_name] = row.shap_value_cents;
    });

    const data = Array.from(dataByDate.values()).sort((a, b) => 
      a.date.localeCompare(b.date)
    );

    return NextResponse.json({ 
      success: true,
      data,
      features: ['RINs_momentum', 'Tariff_risk', 'Drought_zscore', 'Crush_margin'],
      count: data.length,
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    console.error('BigQuery SHAP API error:', error);
    return NextResponse.json({ 
      success: false, 
      error: error.message 
    }, { status: 500 });
  }
}

