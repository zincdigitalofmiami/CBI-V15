import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // TODO: Query DuckDB for ingestion status
    // For now, return mock data
    const status = {
      databento: {
        status: 'active',
        lastUpdate: '2024-12-03T13:00:00Z',
        rowCount: 3162810,
        tables: ['raw.databento_futures_ohlcv_1d', 'raw.databento_futures_ohlcv_1h']
      },
      fred: {
        status: 'active',
        lastUpdate: '2024-12-03T12:00:00Z',
        rowCount: 118102,
        tables: ['raw.fred_economic']
      },
      duckdb: {
        status: 'active',
        lastUpdate: '2024-12-03T13:12:00Z',
        totalRows: 3735413,
        tablesLoaded: 6
      }
    };

    return NextResponse.json(status);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to fetch ingestion status' },
      { status: 500 }
    );
  }
}

