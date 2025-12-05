import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET() {
  try {
    return NextResponse.json({ 
      success: true,
      data: [],
      features: [],
      count: 0,
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




