/**
 * Databento Futures Ingestion - Trigger.dev Job
 * 
 * Ingests futures prices for all 38 symbols:
 * - Agricultural: ZL, ZS, ZM, ZC, ZW, KE, MW, ZO, ZR, LE, GF
 * - Energy: CL, HO, RB, NG
 * - Metals: GC, SI, HG, PL, PA
 * - Treasuries: ZN, ZB, ZF
 * - FX: 6E, 6B, 6J, 6C, 6A, 6S, 6M, 6N, 6L, DX
 * - Palm Oil: FCPO
 * 
 * Uses Databento Historical API for backfills and Live API for real-time.
 */

import { task, schedules } from "@trigger.dev/sdk/v3";
import { MotherDuckClient } from "../src/shared/motherduck_client";

const DATABENTO_API_KEY = process.env.DATABENTO_API_KEY!;
const DATABENTO_BASE_URL = "https://hist.databento.com/v0";

// All 38 futures symbols
const FUTURES_SYMBOLS = {
  agricultural: ["ZL", "ZS", "ZM", "ZC", "ZW", "KE", "MW", "ZO", "ZR", "LE", "GF"],
  energy: ["CL", "HO", "RB", "NG"],
  metals: ["GC", "SI", "HG", "PL", "PA"],
  treasuries: ["ZN", "ZB", "ZF"],
  fx: ["6E", "6B", "6J", "6C", "6A", "6S", "6M", "6N", "6L", "DX"],
  palm_oil: ["FCPO"],
};

const ALL_SYMBOLS = Object.values(FUTURES_SYMBOLS).flat();

interface DatabentoBBO {
  ts_event: string;
  symbol: string;
  bid_px: number;
  ask_px: number;
  bid_sz: number;
  ask_sz: number;
}

/**
 * Fetch BBO (Best Bid/Offer) data from Databento
 */
async function fetchDatabentoBBO(
  symbols: string[],
  startDate: string,
  endDate: string
): Promise<DatabentoBBO[]> {
  const url = new URL(`${DATABENTO_BASE_URL}/timeseries.get_range`);
  
  const params = {
    dataset: "GLBX.MDP3", // CME Globex
    symbols: symbols.join(","),
    schema: "bbo-1s", // Best bid/offer at 1-second intervals
    start: startDate,
    end: endDate,
    stype_in: "continuous", // Continuous contracts
    encoding: "json",
  };

  Object.entries(params).forEach(([key, value]) => {
    url.searchParams.append(key, value);
  });

  const response = await fetch(url.toString(), {
    headers: {
      Authorization: `Bearer ${DATABENTO_API_KEY}`,
    },
  });

  if (!response.ok) {
    throw new Error(`Databento API error: ${response.statusText}`);
  }

  const data = await response.json();
  return data;
}

/**
 * Calculate OHLCV from BBO data
 */
function aggregateToOHLCV(bboData: DatabentoBBO[], interval: "1min" | "5min" | "1h" | "1d") {
  const intervalMs = {
    "1min": 60 * 1000,
    "5min": 5 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
  }[interval];

  const buckets = new Map<string, DatabentoBBO[]>();

  for (const tick of bboData) {
    const ts = new Date(tick.ts_event).getTime();
    const bucketKey = `${tick.symbol}_${Math.floor(ts / intervalMs) * intervalMs}`;
    
    if (!buckets.has(bucketKey)) {
      buckets.set(bucketKey, []);
    }
    buckets.get(bucketKey)!.push(tick);
  }

  const ohlcv = [];
  for (const [key, ticks] of buckets.entries()) {
    const [symbol, tsStr] = key.split("_");
    const midPrices = ticks.map((t) => (t.bid_px + t.ask_px) / 2);
    
    ohlcv.push({
      symbol,
      timestamp: new Date(parseInt(tsStr)),
      open: midPrices[0],
      high: Math.max(...midPrices),
      low: Math.min(...midPrices),
      close: midPrices[midPrices.length - 1],
      volume: ticks.reduce((sum, t) => sum + t.bid_sz + t.ask_sz, 0),
    });
  }

  return ohlcv;
}

/**
 * Main Trigger.dev task: Databento Daily Ingestion
 */
export const databentoIngestJob = task({
  id: "databento-ingest-job",
  retry: {
    maxAttempts: 3,
    factor: 2,
    minTimeoutInMs: 5000,
    maxTimeoutInMs: 60000,
  },
  run: async (payload: { symbols?: string[]; daysBack?: number }, { ctx }) => {
    const symbols = payload.symbols || ALL_SYMBOLS;
    const daysBack = payload.daysBack || 1;

    const endDate = new Date().toISOString().split("T")[0];
    const startDate = new Date(Date.now() - daysBack * 24 * 60 * 60 * 1000)
      .toISOString()
      .split("T")[0];

    console.log(`[Databento] Fetching ${symbols.length} symbols from ${startDate} to ${endDate}`);

    // Fetch in batches to avoid rate limits (1,500 req/min)
    const BATCH_SIZE = 10;
    const allOHLCV = [];

    for (let i = 0; i < symbols.length; i += BATCH_SIZE) {
      const batch = symbols.slice(i, i + BATCH_SIZE);
      
      try {
        console.log(`[Databento] Batch ${i / BATCH_SIZE + 1}: ${batch.join(", ")}`);
        
        const bboData = await fetchDatabentoBBO(batch, startDate, endDate);
        const ohlcv = aggregateToOHLCV(bboData, "1d");
        
        allOHLCV.push(...ohlcv);
        
        console.log(`[Databento] Fetched ${ohlcv.length} daily bars`);
        
        // Rate limit: 1,500 req/min = 1 req per 40ms
        await new Promise((resolve) => setTimeout(resolve, 100));
      } catch (error) {
        console.error(`[Databento] Error fetching batch ${batch.join(", ")}:`, error);
      }
    }

    // Load to MotherDuck
    if (allOHLCV.length > 0) {
      const motherduck = new MotherDuckClient();
      await motherduck.insertBatch("raw.databento_futures", allOHLCV);
      console.log(`[Databento] Loaded ${allOHLCV.length} records to MotherDuck`);
    }

    return {
      success: true,
      recordsIngested: allOHLCV.length,
      symbols: symbols.length,
      dateRange: { start: startDate, end: endDate },
      timestamp: new Date().toISOString(),
    };
  },
});

/**
 * Schedule: Run daily at 6 PM UTC (after market close)
 */
export const databentoSchedule = schedules.task({
  id: "databento-daily-schedule",
  cron: "0 18 * * *", // 6 PM UTC
  task: databentoIngestJob,
  payload: {
    symbols: ALL_SYMBOLS,
    daysBack: 1,
  },
});

