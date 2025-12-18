import { NextResponse } from "next/server";

export const runtime = "edge";
export const dynamic = "force-dynamic";
export const revalidate = 0;

type BucketLiveRow = {
  bucket: string;
  symbol: string;
  price: number;
  prevPrice: number | null;
  change: number | null;
  changePercent: number | null;
  asOfDate: string | null;
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

function asDateOnly(v: unknown): string | null {
  const s = typeof v === "string" ? v : "";
  const d = s ? new Date(s) : null;
  if (!d || Number.isNaN(d.getTime())) return null;
  return d.toISOString().slice(0, 10);
}

async function fetchDatabentoLastTwo(symbol: string): Promise<{
  price: number | null;
  prevPrice: number | null;
  asOfDate: string | null;
}> {
  const key = getDatabentoKey();
  if (!key) throw new Error("DATABENTO_API_KEY not set");

  const body = {
    dataset: "GLBX.MDP3",
    schema: "ohlcv-1d",
    symbols: [toContinuous(symbol)],
    stype_in: "continuous",
    start: isoDaysAgo(20),
    end: new Date().toISOString().slice(0, 19),
    limit: 4,
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

  const recs = extractRecords(parsed)
    .slice()
    .sort((a, b) => String(a["ts_event"] ?? "").localeCompare(String(b["ts_event"] ?? "")));

  const last = recs[recs.length - 1];
  const prev = recs.length > 1 ? recs[recs.length - 2] : undefined;

  const v0 = last ? getNumber(last, ["close", "close_px", "close_price"]) : null;
  const v1 = prev ? getNumber(prev, ["close", "close_px", "close_price"]) : null;
  const d0 = last ? asDateOnly(last["ts_event"] ?? "") : null;

  return { price: v0, prevPrice: v1, asOfDate: d0 };
}

const BUCKET_SYMBOLS: { bucket: string; symbol: string }[] = [
  { bucket: "Crush", symbol: "ZL" },
  { bucket: "China", symbol: "HG" },
  { bucket: "FX", symbol: "DX" },
  { bucket: "Fed", symbol: "ZN" },
  { bucket: "Tariff", symbol: "ZS" },
  { bucket: "Biofuel", symbol: "HO" },
  { bucket: "Energy", symbol: "CL" },
  { bucket: "Volatility", symbol: "VX" }, // VIX futures proxy in Databento
];

export async function GET() {
  try {
    const results = await Promise.all(
      BUCKET_SYMBOLS.map(async (b) => {
        try {
          const r = await fetchDatabentoLastTwo(b.symbol);
          const price = r.price;
          const prev = r.prevPrice;
          const change = price === null || prev === null ? null : price - prev;
          const changePercent = price === null || prev === null || prev === 0 ? null : (change! / prev) * 100;
          return {
            bucket: b.bucket,
            symbol: b.symbol,
            price: price ?? NaN,
            prevPrice: prev,
            change,
            changePercent,
            asOfDate: r.asOfDate,
          } satisfies BucketLiveRow;
        } catch {
          return {
            bucket: b.bucket,
            symbol: b.symbol,
            price: NaN,
            prevPrice: null,
            change: null,
            changePercent: null,
            asOfDate: null,
          } satisfies BucketLiveRow;
        }
      }),
    );

    return NextResponse.json({
      success: true,
      data: results,
      ts: new Date().toISOString(),
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ success: false, error: message }, { status: 500 });
  }
}

