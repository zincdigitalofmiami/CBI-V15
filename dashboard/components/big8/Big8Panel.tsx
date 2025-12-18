"use client";

import { useEffect, useMemo, useState } from "react";

type Big8Signal = {
  bucket: string; // Crush, China, FX, Fed, Tariff, Biofuel, Energy, Volatility
  direction: "bullish" | "bearish" | "neutral";
  strength: number; // 0..1
  confidence: number; // 0..1
};

type Big8LiveRow = {
  bucket: string;
  symbol: string;
  price: number;
  changePercent: number | null;
  asOfDate: string | null;
};

const DEFAULT_SIGNALS: Big8Signal[] = [
  { bucket: "Crush", direction: "neutral", strength: 0.5, confidence: 0.6 },
  { bucket: "China", direction: "bearish", strength: 0.6, confidence: 0.7 },
  { bucket: "FX", direction: "bearish", strength: 0.4, confidence: 0.5 },
  { bucket: "Fed", direction: "neutral", strength: 0.3, confidence: 0.5 },
  { bucket: "Tariff", direction: "bearish", strength: 0.7, confidence: 0.8 },
  { bucket: "Biofuel", direction: "bullish", strength: 0.6, confidence: 0.7 },
  { bucket: "Energy", direction: "bullish", strength: 0.5, confidence: 0.6 },
  { bucket: "Volatility", direction: "bearish", strength: 0.5, confidence: 0.6 },
];

export default function Big8Panel({ signals = DEFAULT_SIGNALS }: { signals?: Big8Signal[] }) {
  const [live, setLive] = useState<Record<string, Big8LiveRow> | null>(null);
  const [liveError, setLiveError] = useState<string | null>(null);
  const [liveTs, setLiveTs] = useState<string | null>(null);

  useEffect(() => {
    async function fetchLive() {
      try {
        setLiveError(null);
        const res = await fetch("/api/big8/live", { cache: "no-store" });
        const json: unknown = await res.json();
        const j = json as { success?: boolean; data?: unknown; ts?: unknown; error?: unknown };
        if (j?.success && Array.isArray(j?.data)) {
          const byBucket: Record<string, Big8LiveRow> = {};
          for (const r of j.data as Record<string, unknown>[]) {
            const bucket = String(r["bucket"]);
            byBucket[bucket] = {
              bucket,
              symbol: String(r["symbol"]),
              price: Number(r["price"]),
              changePercent:
                r["changePercent"] === null || r["changePercent"] === undefined
                  ? null
                  : Number(r["changePercent"]),
              asOfDate: typeof r["asOfDate"] === "string" ? r["asOfDate"] : null,
            };
          }
          setLive(byBucket);
          setLiveTs(typeof j.ts === "string" ? j.ts : new Date().toISOString());
        } else {
          setLiveError(typeof j?.error === "string" ? j.error : "Failed to load live data");
        }
      } catch (e: unknown) {
        setLiveError(e instanceof Error ? e.message : String(e));
      }
    }

    fetchLive();
    const interval = setInterval(fetchLive, 300000); // 5 min
    return () => clearInterval(interval);
  }, []);

  const merged = useMemo(() => {
    return signals.map((s) => ({
      ...s,
      live: live?.[s.bucket] || null,
    }));
  }, [signals, live]);

  return (
    <section className="w-full bg-[#070a12] border-t border-white/5 py-12">
      <div className="max-w-screen-2xl mx-auto px-10">
        <div className="flex items-end justify-between mb-4">
          <div>
            <h2 className="text-white text-2xl font-thin tracking-wide">Big 8 Driver Panel</h2>
            <div className="text-[11px] text-zinc-600 font-extralight mt-1">
              Live prices via `/api/big8/live` {liveTs ? `• ${new Date(liveTs).toLocaleTimeString()}` : ""}
              {liveError ? ` • ${liveError}` : ""}
            </div>
          </div>
          <span className="text-xs font-medium text-emerald-400/90 bg-emerald-500/10 border border-emerald-500/20 px-2 py-1 rounded-full">
            LIVE
          </span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-4">
          {merged.map((s) => (
            <div
              key={s.bucket}
              className="rounded-2xl border border-white/10 p-4 flex flex-col items-center gap-2 bg-white/[0.03] backdrop-blur-md"
              title={`${s.bucket}: ${s.direction} • strength ${Math.round(s.strength * 100)}% • confidence ${Math.round(
                s.confidence * 100,
              )}%`}
            >
              <span className="text-xs text-zinc-400 tracking-wide">{s.bucket}</span>
              {s.live?.price !== undefined && Number.isFinite(Number(s.live.price)) ? (
                <div className="text-[11px] text-zinc-300 font-extralight">
                  <span className="font-mono">{s.live.symbol}</span>{" "}
                  <span className="font-mono">{Number(s.live.price).toFixed(2)}</span>
                  {s.live.changePercent !== null ? (
                    <span
                      className={
                        "ml-1 " +
                        (Number(s.live.changePercent) >= 0 ? "text-emerald-400" : "text-red-400")
                      }
                    >
                      {Number(s.live.changePercent) >= 0 ? "+" : ""}
                      {Number(s.live.changePercent).toFixed(2)}%
                    </span>
                  ) : null}
                </div>
              ) : (
                <div className="text-[11px] text-zinc-600 font-extralight">—</div>
              )}
              <span
                className={
                  "text-sm font-medium " +
                  (s.direction === "bullish"
                    ? "text-emerald-400"
                    : s.direction === "bearish"
                      ? "text-red-400"
                      : "text-gray-400")
                }
              >
                {s.direction.toUpperCase()}
              </span>
              <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
                <div
                  className={
                    "h-2 rounded " +
                    (s.direction === "bullish"
                      ? "bg-emerald-500"
                      : s.direction === "bearish"
                        ? "bg-red-500"
                        : "bg-gray-500")
                  }
                  style={{ width: `${Math.round(s.strength * 100)}%` }}
                />
              </div>
              <span className="text-[10px] text-gray-500">
                Conf {Math.round(s.confidence * 100)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
