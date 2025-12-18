import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const revalidate = 0;

// Yahoo Finance symbol for Soybean Oil Futures
const YAHOO_SYMBOL = "ZL=F";

export async function GET() {
  try {
    // Yahoo Finance API - free, no auth needed
    const now = Math.floor(Date.now() / 1000);
    const oneYearAgo = now - 365 * 24 * 60 * 60;
    
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${YAHOO_SYMBOL}?period1=${oneYearAgo}&period2=${now}&interval=1d`;
    
    const resp = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (compatible; CBI-V15/1.0)",
      },
    });

    if (!resp.ok) {
      throw new Error(`Yahoo Finance error: ${resp.status}`);
    }

    const json = await resp.json();
    const result = json?.chart?.result?.[0];
    
    if (!result) {
      throw new Error("No data from Yahoo Finance");
    }

    const timestamps = result.timestamp || [];
    const quote = result.indicators?.quote?.[0] || {};
    const { open, high, low, close, volume } = quote;

    const rows = timestamps.map((ts: number, i: number) => ({
      date: new Date(ts * 1000).toISOString().slice(0, 10),
      open: open?.[i] ?? null,
      high: high?.[i] ?? null,
      low: low?.[i] ?? null,
      close: close?.[i] ?? null,
      volume: volume?.[i] ?? null,
    })).filter((r: { close: number | null }) => r.close !== null);

    return NextResponse.json({
      success: true,
      data: rows,
      count: rows.length,
      symbol: "ZL",
      source: "Yahoo Finance",
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
