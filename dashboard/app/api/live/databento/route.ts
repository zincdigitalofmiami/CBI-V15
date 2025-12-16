import { queryMotherDuck } from "@/lib/md";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const symbol = searchParams.get("symbol") || "ZL";
  const days = parseInt(searchParams.get("days") || "90");

  try {
    const data = await queryMotherDuck(`
      SELECT 
        as_of_date as date,
        close,
        volume,
        open,
        high,
        low
      FROM raw.databento_futures_ohlcv_1d
      WHERE symbol = '${symbol}'
        AND as_of_date >= CURRENT_DATE - INTERVAL '${days} days'
      ORDER BY as_of_date ASC
    `);

    return NextResponse.json({
      success: true,
      symbol,
      days,
      rows: data.length,
      data
    });
  } catch (error) {
    console.error("Databento API error:", error);
    return NextResponse.json(
      { 
        success: false,
        error: String(error) 
      }, 
      { status: 500 }
    );
  }
}
