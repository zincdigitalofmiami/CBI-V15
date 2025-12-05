import { BigQuery } from '@google-cloud/bigquery';
import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET() {
  try {
    const bigquery = new BigQuery({ projectId: 'cbi-v15' });

    // Aggregate latest daily anomalies per region/metric from staging.weather_regions_aggregated
    const query = `
      WITH latest AS (
        SELECT
          MAX(date) AS max_date
        FROM \`cbi-v15.staging.weather_regions_aggregated\`
      )
      SELECT
        a.date,
        a.region,
        a.metric,
        a.value_mean,
        a.value_min,
        a.value_max,
        a.station_count,
        a.coverage_pct
      FROM \`cbi-v15.staging.weather_regions_aggregated\` a
      JOIN latest l
      ON a.date = l.max_date
    `;

    const [rows] = await bigquery.query(query);

    return NextResponse.json({
      success: true,
      data: rows,
      count: rows.length,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('weather/summary API error:', error);
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 },
    );
  }
}

