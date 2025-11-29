import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

// Mock ZL data generator - replace with BigQuery once credentials are configured
function generateMockZLData() {
  const data = [];
  const basePrice = 45.50;
  const today = new Date();
  
  for (let i = 89; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    
    // Generate realistic price movement
    const trend = Math.sin(i / 10) * 2;
    const noise = (Math.random() - 0.5) * 1.5;
    const price = basePrice + trend + noise;
    
    data.push({
      date: { value: date.toISOString().split('T')[0] },
      open: price - 0.2,
      high: price + 0.3,
      low: price - 0.3,
      close: price,
      volume: Math.floor(15000 + Math.random() * 5000),
      symbol: 'ZL'
    });
  }
  
  return data;
}

export async function GET() {
  try {
    // TODO: Replace with actual BigQuery query once credentials are configured
    // For now, return mock data so chart renders immediately
    const rows = generateMockZLData();

    return NextResponse.json({ 
      success: true,
      data: rows,
      count: rows.length,
      symbol: 'ZL',
      note: 'Using mock data - configure BigQuery credentials in Vercel to load real data',
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    console.error('API error:', error);
    return NextResponse.json({ 
      success: false, 
      error: error.message 
    }, { status: 500 });
  }
}
