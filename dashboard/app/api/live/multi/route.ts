import { queryMotherDuck } from "@/lib/md";
import { NextResponse } from "next/server";

export const runtime = "edge";
export const dynamic = "force-dynamic";
export const revalidate = 0;

function parseSymbols(raw: string | null): string[] {
  const symbols = (raw || "")
    .split(",")
    .map((s) => s.trim().toUpperCase())
    .filter(Boolean)
    .filter((s) => /^[A-Z0-9]{1,12}$/.test(s));
  return Array.from(new Set(symbols)).slice(0, 32);
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const symbols = parseSymbols(searchParams.get("symbols"));
  const days = Math.max(1, Math.min(3650, parseInt(searchParams.get("days") || "90")));

  if (symbols.length === 0) {
    return NextResponse.json(
      { success: false, error: "Missing symbols query param (e.g., ?symbols=ZL,ZS,ZM)" },
      { status: 400 },
    );
  }

  const inList = symbols.map((s) => `'${s}'`).join(", ");

  try {
    const rows = await queryMotherDuck(`
      SELECT
        symbol,
        as_of_date AS date,
        close
      FROM raw.databento_futures_ohlcv_1d
      WHERE symbol IN (${inList})
        AND as_of_date >= CURRENT_DATE - INTERVAL '${days} days'
      ORDER BY symbol, as_of_date ASC
    `);

    const bySymbol: Record<string, { time: number; value: number }[]> = {};
    for (const s of symbols) bySymbol[s] = [];

    for (const r of rows as Record<string, unknown>[]) {
      const symbol = String(r["symbol"]);
      const dateVal = r["date"];
      const dt =
        typeof dateVal === "object" && dateVal !== null && "value" in dateVal
          ? String((dateVal as { value?: unknown }).value ?? "")
          : String(dateVal ?? "");
      const close = Number(r["close"]);
      const time = Math.floor(new Date(dt).getTime() / 1000);
      if (!Number.isFinite(time) || !Number.isFinite(close)) continue;
      if (!bySymbol[symbol]) bySymbol[symbol] = [];
      bySymbol[symbol].push({ time, value: close });
    }

    return NextResponse.json({
      success: true,
      symbols,
      days,
      data: bySymbol,
      rows: rows.length,
      ts: new Date().toISOString(),
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ success: false, error: message }, { status: 500 });
  }
}

