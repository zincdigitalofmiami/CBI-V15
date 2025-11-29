import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET() {
  try {
    const DATABENTO_API_KEY = process.env.DATABENTO_API_KEY;
    
    if (!DATABENTO_API_KEY) {
      throw new Error('DATABENTO_API_KEY not configured');
    }

    // Get last 90 days of ZL OHLCV data from Databento API
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 90);

    const params = new URLSearchParams({
      dataset: 'GLBX.MDP3',
      symbols: 'ZL.FUT',
      schema: 'ohlcv-1d',
      start: startDate.toISOString().split('T')[0],
      end: endDate.toISOString().split('T')[0],
      stype_in: 'parent',
      encoding: 'json'
    });

    const response = await fetch(
      `https://hist.databento.com/v0/timeseries.get_range?${params}`,
      {
        headers: {
          'Authorization': `Bearer ${DATABENTO_API_KEY}`,
          'Accept': 'application/json'
        }
      }
    );

    if (!response.ok) {
      throw new Error(`Databento API error: ${response.status} ${response.statusText}`);
    }

    const text = await response.text();
    const lines = text.trim().split('\n');
    
    // Parse NDJSON response
    const data = lines.map(line => {
      const record = JSON.parse(line);
      return {
        date: { value: new Date(record.ts_event / 1000000).toISOString().split('T')[0] },
        open: record.open / 1e9,
        high: record.high / 1e9,
        low: record.low / 1e9,
        close: record.close / 1e9,
        volume: record.volume,
        symbol: 'ZL'
      };
    });

    return NextResponse.json({ 
      success: true,
      data,
      count: data.length,
      symbol: 'ZL',
      source: 'Databento Live API',
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    console.error('Databento API error:', error);
    return NextResponse.json({ 
      success: false, 
      error: error.message,
      message: 'Add DATABENTO_API_KEY to Vercel Environment Variables'
    }, { status: 500 });
  }
}
