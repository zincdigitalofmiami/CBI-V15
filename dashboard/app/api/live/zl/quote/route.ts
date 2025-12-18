import { NextResponse } from "next/server";

export const runtime = "edge";
export const dynamic = "force-dynamic";
export const revalidate = 0;

type Quote = {
  symbol: string;
  price: number;
  prevPrice: number | null;
  change: number | null;
  changePercent: number | null;
  ts: string;
  source: "motherduck" | "databento_hist" | "unavailable";
  warning?: string;
};

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

export async function GET() {
  const ts = new Date().toISOString();

  // 1) Preferred: MotherDuck (fast, already normalized into `raw.databento_futures_ohlcv_1d`)
  try {
    const { queryMotherDuck } = await import("@/lib/md");
    const rows = await queryMotherDuck(`
      WITH ranked AS (
        SELECT
          as_of_date,
          close,
          ROW_NUMBER() OVER (ORDER BY as_of_date DESC) AS rn
        FROM raw.databento_futures_ohlcv_1d
        WHERE symbol = 'ZL'
      )
      SELECT
        MAX(CASE WHEN rn = 1 THEN close END) AS v0,
        MAX(CASE WHEN rn = 2 THEN close END) AS v1,
        MAX(CASE WHEN rn = 1 THEN as_of_date END) AS d0
      FROM ranked
      WHERE rn IN (1,2)
    `);

    const r0 = (rows?.[0] || {}) as Record<string, unknown>;
    const v0 = getNumber(r0, ["v0"]);
    const v1 = getNumber(r0, ["v1"]);

    if (v0 !== null) {
      const prev = v1 ?? null;
      const change = prev === null ? null : v0 - prev;
      const changePercent = prev === null || prev === 0 ? null : (change! / prev) * 100;
      const out: Quote = {
        symbol: "ZL",
        price: v0,
        prevPrice: prev,
        change,
        changePercent,
        ts,
        source: "motherduck",
      };
      return NextResponse.json({ success: true, ...out });
    }
  } catch {
    // fall through to Databento hist
  }

  // 2) Fallback: Databento Historical REST (still “live” enough for homepage)
  try {
    const key = getDatabentoKey();
    if (!key) throw new Error("DATABENTO_API_KEY not set");

    const body = {
      dataset: "GLBX.MDP3",
      schema: "ohlcv-1d",
      symbols: [toContinuous("ZL")],
      stype_in: "continuous",
      start: isoDaysAgo(7),
      end: new Date().toISOString().slice(0, 19),
      limit: 2,
      encoding: "json",
      pretty_ts: true,
      pretty_px: true,
      map_symbols: true,
    };

    const resp = await fetch("https://hist.databento.com/v0/timeseries.get_range", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${key}`,
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
        throw new Error(`Databento error (${resp.status}): ${detail}`);
      } catch {
        throw new Error(`Databento error (${resp.status}): ${text.slice(0, 500)}`);
      }
    }

    const text = await resp.text();
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new Error(`Databento response was not JSON (status ${resp.status})`);
    }

    const recs = extractRecords(parsed);
    // take last 2 by time if possible
    const sorted = recs
      .slice()
      .sort((a, b) => {
        const ta = String(a["ts_event"] ?? a["date"] ?? a["ts"] ?? "");
        const tb = String(b["ts_event"] ?? b["date"] ?? b["ts"] ?? "");
        return ta.localeCompare(tb);
      })
      .slice(-2);

    const last = sorted[sorted.length - 1] as Record<string, unknown> | undefined;
    const prev = sorted.length > 1 ? (sorted[sorted.length - 2] as Record<string, unknown>) : undefined;

    const v0 = last ? getNumber(last, ["close", "close_px", "px_close", "close_price", "c"]) : null;
    const v1 = prev ? getNumber(prev, ["close", "close_px", "px_close", "close_price", "c"]) : null;

    if (v0 !== null) {
      const p = v1 ?? null;
      const change = p === null ? null : v0 - p;
      const changePercent = p === null || p === 0 ? null : (change! / p) * 100;
      const out: Quote = {
        symbol: "ZL",
        price: v0,
        prevPrice: p,
        change,
        changePercent,
        ts,
        source: "databento_hist",
      };
      return NextResponse.json({ success: true, ...out });
    }

    const topKeys =
      parsed && typeof parsed === "object" ? Object.keys(parsed as Record<string, unknown>).join(",") : "(non-object)";
    const lastKeys = last ? Object.keys(last).join(",") : "(no last record)";
    throw new Error(`Could not parse close price from Databento response (top keys: ${topKeys}; last keys: ${lastKeys})`);
  } catch (e: unknown) {
    const warning = e instanceof Error ? e.message : String(e);
    const out: Quote = {
      symbol: "ZL",
      price: 0,
      prevPrice: null,
      change: null,
      changePercent: null,
      ts,
      source: "unavailable",
      warning,
    };
    return NextResponse.json({ success: false, ...out }, { status: 200 });
  }
}

