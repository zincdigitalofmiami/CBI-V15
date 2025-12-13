import { queryMotherDuck } from '@/lib/md';
import { NextResponse } from 'next/server';

export const runtime = 'nodejs'; // Required for native DuckDB

export async function GET() {
  try {
    // Basic connectivity test
    const basicTest = await queryMotherDuck('SELECT 1 as test_value');

    // Table existence check
    const tableCheck = await queryMotherDuck(`
      SELECT
        'forecasts.zl_predictions' as table_name,
        COUNT(*) as row_count
      FROM forecasts.zl_predictions
      WHERE as_of_date >= CURRENT_DATE - INTERVAL '30 days'
    `);

    // Schema connectivity test
    const schemaTest = await queryMotherDuck(`
      SELECT
        table_schema,
        table_name,
        table_type
      FROM information_schema.tables
      WHERE table_schema IN ('forecasts', 'raw', 'features')
      LIMIT 5
    `);

    return NextResponse.json({
      success: true,
      status: 'healthy',
      timestamp: new Date().toISOString(),
      tests: {
        basic_connectivity: {
          passed: true,
          result: basicTest
        },
        table_access: {
          passed: true,
          result: tableCheck
        },
        schema_access: {
          passed: true,
          result: schemaTest
        }
      },
      environment: {
        node_version: process.version,
        has_motherduck_token: !!process.env.MOTHERDUCK_TOKEN,
        motherduck_db: process.env.MOTHERDUCK_DB || 'cbi_v15'
      }
    });
  } catch (error: any) {
    console.error('MotherDuck Health Check Failed:', error);
    return NextResponse.json({
      success: false,
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: {
        message: error.message,
        type: error.constructor.name,
        stack: error.stack?.split('\n')[0] // First line only for security
      },
      environment: {
        node_version: process.version,
        has_motherduck_token: !!process.env.MOTHERDUCK_TOKEN,
        motherduck_db: process.env.MOTHERDUCK_DB || 'cbi_v15'
      }
    }, { status: 500 });
  }
}