import { BigQuery } from '@google-cloud/bigquery';
import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const bigquery = new BigQuery({
      projectId: 'cbi-v15',
    });

    const query = `
      WITH metrics AS (
        SELECT
          '1w' AS horizon,
          COUNT(*) AS n,
          AVG(ABS(target - prediction)) AS mae,
          AVG(ABS((target - prediction) / NULLIF(ABS(target), 0))) * 100 AS mape
        FROM \`cbi-v15.predictions.zl_predictions_1w\`
        UNION ALL
        SELECT
          '1m' AS horizon,
          COUNT(*) AS n,
          AVG(ABS(target - prediction)) AS mae,
          AVG(ABS((target - prediction) / NULLIF(ABS(target), 0))) * 100 AS mape
        FROM \`cbi-v15.predictions.zl_predictions_1m\`
        UNION ALL
        SELECT
          '3m' AS horizon,
          COUNT(*) AS n,
          AVG(ABS(target - prediction)) AS mae,
          AVG(ABS((target - prediction) / NULLIF(ABS(target), 0))) * 100 AS mape
        FROM \`cbi-v15.predictions.zl_predictions_3m\`
        UNION ALL
        SELECT
          '6m' AS horizon,
          COUNT(*) AS n,
          AVG(ABS(target - prediction)) AS mae,
          AVG(ABS((target - prediction) / NULLIF(ABS(target), 0))) * 100 AS mape
        FROM \`cbi-v15.predictions.zl_predictions_6m\`
      )
      SELECT *
      FROM metrics
      ORDER BY
        CASE horizon
          WHEN '1w' THEN 1
          WHEN '1m' THEN 2
          WHEN '3m' THEN 3
          WHEN '6m' THEN 4
        END;
    `;

    const [rows] = await bigquery.query(query);

    return NextResponse.json({
      success: true,
      data: rows,
      count: rows.length,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('BigQuery training metrics error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 },
    );
  }
}

