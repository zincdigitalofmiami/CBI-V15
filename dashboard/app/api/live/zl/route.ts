import { NextResponse } from "next/server";

export const runtime = "edge";
export const dynamic = "force-dynamic";
export const revalidate = 0;

function isoDaysAgo(days: number): string {
  const d = new Date(Date.now() - days * 86_400_000);
  return d.toISOString().slice(0, 19);
}

function getDatabentoKey(): string | null {
  return (
    process.env.DATABENTO_API_KEY ||
    process.env.DATABENTO_KEY ||
    process.env.DATABENTO_TOKEN ||
    process.env.databento_api_key ||
    null
  );
}

function toContinuous(sym: string): string {
  return sym.includes(".c.") ? sym : `${sym}.c.0`;
}

function extractRecords(json: unknown): Record<string, unknown>[] {
  if (Array.isArray(json)) return json as Record<string, unknown>[];
  if (json && typeof json === "object") {
    const j = json as Record<string, unknown>;
    if (Array.isArray(j["data"])) return j["data"] as Record<string, unknown>[];
    if (Array.isArray(j["records"])) return j["records"] as Record<string, unknown>[];
  }
  return [];
}

function getNumber(rec: Record<string, unknown>, keys: string[]): number | null {
  for (const k of keys) {
    const v = rec[k];
    const n = typeof v === "string" ? Number(v) : Number(v);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

function asDateOnly(v: unknown): string | null {
  const s = typeof v === "string" ? v : v && typeof v === "object" && "value" in v ? String((v as any).value) : "";
  const d = s ? new Date(s) : null;
  if (!d || Number.isNaN(d.getTime())) return null;
  return d.toISOString().slice(0, 10);
}

export async function GET() {
  try {
    const key = getDatabentoKey();
    if (!key) throw new Error("DATABENTO_API_KEY not set");
    
    // Debug: show key prefix to verify it's being read
    const keyPreview = key.slice(0, 5) + "..." + key.slice(-3);

    // Databento Historical REST (daily OHLCV). Used on Vercel because MotherDuck WASM
    // client requires Worker APIs that are not available in Edge runtimes.
    const body = {
      dataset: "GLBX.MDP3",
      schema: "ohlcv-1d",
      symbols: [toContinuous("ZL")],
      stype_in: "continuous",
      start: isoDaysAgo(365),
      end: new Date().toISOString().slice(0, 19),
      limit: 220,
      encoding: "json",
      pretty_ts: true,
      pretty_px: true,
      map_symbols: true,
    };

    // Clean the key (remove any quotes/whitespace that might be in env var)
    const cleanKey = key.trim().replace(/^["']|["']$/g, "");
    
    const resp = await fetch("https://hist.databento.com/v0/timeseries.get_range", {
      method: "POST",
      headers: {
        Authorization: `Basic ${btoa(cleanKey + ":")}`,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const text = await resp.text();
      try {
        const j = JSON.parse(text) as Record<string, unknown>;
        const detail = typeof j["detail"] === "string" ? j["detail"] : JSON.stringify(j).slice(0, 500);
        throw new Error(`Databento error (${resp.status}) [key: ${keyPreview}]: ${detail}`);
      } catch {
        throw new Error(`Databento error (${resp.status}) [key: ${keyPreview}]: ${text.slice(0, 500)}`);
      }
    }

    const text = await resp.text();
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new Error(`Databento response was not JSON (status ${resp.status})`);
    }

    const recs = extractRecords(parsed)
      .slice()
      .sort((a, b) => String(a["ts_event"] ?? "").localeCompare(String(b["ts_event"] ?? "")));

    const rows = recs
      .map((r) => {
        const close = getNumber(r, ["close", "close_px", "close_price"]);
        const as_of_date = asDateOnly(r["ts_event"] ?? r["date"] ?? r["ts"]);
        if (close === null || !as_of_date) return null;
        return {
          symbol: "ZL",
          as_of_date,
          close,
        };
      })
      .filter(Boolean) as Record<string, unknown>[];

    return NextResponse.json({
      success: true,
      data: rows,
      count: Array.isArray(rows) ? rows.length : 0,
      symbol: "ZL",
      source: "Databento Hist",
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
