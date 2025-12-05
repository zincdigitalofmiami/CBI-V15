import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // TODO: Query DuckDB and system metrics
    // For now, return mock data
    const health = {
      duckdb: {
        status: 'healthy',
        databasePath: '/Volumes/Satechi Hub/ZL-Intelligence/duckdb/cbi-v15.duckdb',
        sizeMB: 125.5,
        lastBackup: '2024-12-03T00:00:00Z'
      },
      ingestion: {
        status: 'healthy',
        lastRun: '2024-12-03T13:00:00Z',
        nextRun: '2024-12-04T00:00:00Z',
        errors: 0
      },
      training: {
        status: 'idle',
        lastRun: '2024-12-02T10:00:00Z',
        nextRun: '2024-12-04T00:00:00Z',
        models: {
          '1w': { status: 'trained', mape: 8.09 },
          '1m': { status: 'trained', mape: 12.3 },
          '3m': { status: 'trained', mape: 15.7 },
          '6m': { status: 'trained', mape: 18.2 }
        }
      },
      dashboard: {
        status: 'healthy',
        uptime: '99.9%',
        lastDeploy: '2024-12-03T10:00:00Z'
      }
    };

    return NextResponse.json(health);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to fetch system health' },
      { status: 500 }
    );
  }
}

