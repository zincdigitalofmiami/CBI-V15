import { NextResponse } from "next/server";
import { queryMotherDuck } from "@/lib/md";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    // Query MotherDuck for ZL OHLCV data
    const sql = `
      SELECT 
        symbol,
        as_of_date::VARCHAR as date,
        open,
        high,
        low,
        close,
        volume
      FROM raw.databento_futures_ohlcv_1d 
      WHERE symbol = 'ZL' 
      ORDER BY as_of_date DESC 
      LIMIT 365
    `;

    const rows = await queryMotherDuck(sql);

    // Reverse to get chronological order
    const sortedRows = rows.reverse();

    return NextResponse.json({
      success: true,
      data: sortedRows,
      count: sortedRows.length,
      symbol: "ZL",
      source: "MotherDuck",
      timestamp: new Date().toISOString(),
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    console.error("Live ZL Error:", message);
    return NextResponse.json(
      {
        success: false,
        error: message,
      },
      { status: 500 },
    );
  }
}
