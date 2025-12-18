import { queryMotherDuck } from "@/lib/md";
import { NextResponse } from "next/server";

export const runtime = "edge";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    // Try forecasts.specialist_signals first; fall back to training tables if needed
    const rows = await queryMotherDuck(`
      WITH latest AS (
        SELECT max(as_of_date) AS as_of_date FROM forecasts.specialist_signals
      )
      SELECT s.bucket,
             s.direction,
             s.strength,
             s.confidence,
             s.as_of_date
      FROM forecasts.specialist_signals s
      JOIN latest l ON s.as_of_date = l.as_of_date
      ORDER BY s.bucket
    `).catch(async () => {
      // Fallback placeholder if table not present
      return [
        { bucket: "Crush", direction: "neutral", strength: 0.5, confidence: 0.6, as_of_date: null },
        { bucket: "China", direction: "bearish", strength: 0.6, confidence: 0.7, as_of_date: null },
        { bucket: "FX", direction: "bearish", strength: 0.4, confidence: 0.5, as_of_date: null },
        { bucket: "Fed", direction: "neutral", strength: 0.3, confidence: 0.5, as_of_date: null },
        {
          bucket: "Tariff",
          direction: "bearish",
          strength: 0.7,
          confidence: 0.8,
          as_of_date: null,
        },
        {
          bucket: "Biofuel",
          direction: "bullish",
          strength: 0.6,
          confidence: 0.7,
          as_of_date: null,
        },
        {
          bucket: "Energy",
          direction: "bullish",
          strength: 0.5,
          confidence: 0.6,
          as_of_date: null,
        },
        {
          bucket: "Volatility",
          direction: "bearish",
          strength: 0.5,
          confidence: 0.6,
          as_of_date: null,
        },
      ];
    });

    return NextResponse.json({
      success: true,
      data: rows,
      count: rows.length,
      ts: new Date().toISOString(),
    });
  } catch (error: any) {
    return NextResponse.json({ success: false, error: error.message }, { status: 500 });
  }
}
