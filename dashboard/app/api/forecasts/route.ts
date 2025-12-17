import { queryMotherDuck } from "@/lib/md";
import { NextResponse } from "next/server";

export const runtime = "nodejs"; // Required for native DuckDB

export async function GET() {
  try {
    const rows = await queryMotherDuck(`
      SELECT *
      FROM forecasts.zl_predictions
      WHERE horizon = '1w'
      ORDER BY as_of_date DESC
      LIMIT 50
    `);

    return NextResponse.json({
      success: true,
      data: rows,
      count: Array.isArray(rows) ? rows.length : 0,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error("Database Error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 },
    );
  }
}
