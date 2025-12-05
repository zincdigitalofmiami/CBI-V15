import { NextResponse } from 'next/server';
import duckdb from 'duckdb';
import path from 'path';

export async function GET() {
  try {
    const dbPath = path.resolve('Data/db/cbi-v15.duckdb');
    const db = new duckdb.Database(dbPath);
    
    // Query the reporting view
    const result = await new Promise((resolve, reject) => {
        db.all("SELECT * FROM quant_report_view", (err, res) => {
            if (err) reject(err);
            else resolve(res);
        });
    });

    return NextResponse.json({ success: true, data: result });
  } catch (error) {
    console.error('Database Error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch quant report' },
      { status: 500 }
    );
  }
}

