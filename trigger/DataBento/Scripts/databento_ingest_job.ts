/**
 * Databento Futures Ingestion - Trigger.dev Job
 *
 * Ingests futures prices for all 38 symbols using the official Databento SDK
 * Based on zl-intelligence working pattern: databento_to_motherduck.py
 *
 * Symbols by category:
 * - Agricultural: ZL, ZS, ZM, ZC, ZW, ZO, ZR, HE, LE, GF, FCPO
 * - Energy: CL, HO, RB, NG
 * - Metals: GC, SI, HG, PL, PA
 * - Treasuries: ZF, ZN, ZB
 * - FX: 6E, 6B, 6J, 6C, 6A, 6S, 6M, 6N, 6L, DX
 *
 * Uses Databento Historical API for backfills, schema: ohlcv-1d
 */

import { schedules, task } from "@trigger.dev/sdk/v3";
import { exec } from "child_process";
import path from "path";
import { promisify } from "util";

const execAsync = promisify(exec);

// All 38 futures symbols (matches zl-intelligence)
const FUTURES_SYMBOLS = {
  agricultural: ["ZL", "ZS", "ZM", "ZC", "ZW", "ZO", "ZR", "HE", "LE", "GF", "FCPO"],
  energy: ["CL", "HO", "RB", "NG"],
  metals: ["GC", "SI", "HG", "PL", "PA"],
  treasuries: ["ZF", "ZN", "ZB"],
  fx: ["6E", "6B", "6J", "6C", "6A", "6S", "6M", "6N", "6L", "DX"],
};

const ALL_SYMBOLS = Object.values(FUTURES_SYMBOLS).flat();

/**
 * Main Trigger.dev task: Databento Daily Ingestion
 * Uses the working Python SDK approach from zl-intelligence
 */
export const databentoIngestJob = task({
  id: "databento-ingest-job",
  retry: {
    maxAttempts: 3,
    factor: 2,
    minTimeoutInMs: 10000,
    maxTimeoutInMs: 120000, // 2 minutes for large data pulls
  },
  run: async (payload: { symbols?: string[]; daysBack?: number }, { ctx }) => {
    const symbols = payload.symbols || ["ZL", "ZS", "ZM"]; // Start with core symbols
    const daysBack = payload.daysBack || 1;

    const endDate = new Date().toISOString().split("T")[0];
    const startDate = new Date(Date.now() - daysBack * 24 * 60 * 60 * 1000)
      .toISOString()
      .split("T")[0];

    console.log(`[Databento] Starting ingestion for ${symbols.length} symbols: ${symbols.join(", ")}`);
    console.log(`[Databento] Date range: ${startDate} to ${endDate}`);

    // Path to the working Python script
    const scriptPath = path.join(process.cwd(), "trigger", "DataBento", "Scripts", "collect_daily.py");

    try {
      // Execute the Python script with proper environment
      const { stdout, stderr } = await execAsync(
        `cd ${process.cwd()} && python3 "${scriptPath}" --symbols ${symbols.join(" ")} --days ${daysBack}`,
        {
          env: {
            ...process.env,
            PYTHONPATH: process.cwd(),
          },
          timeout: 300000, // 5 minutes timeout
        }
      );

      console.log(`[Databento] Script output:`, stdout);

      if (stderr) {
        console.warn(`[Databento] Script warnings:`, stderr);
      }

      // Parse success from output (script should print success indicators)
      const success = stdout.includes("✅") || stdout.includes("SUCCESS") || !stdout.includes("ERROR");
      const recordsMatch = stdout.match(/(\d+)\s+records?|(\d+)\s+rows?/i);
      const recordsIngested = recordsMatch ? parseInt(recordsMatch[1] || recordsMatch[2]) : 0;

      return {
        success,
        recordsIngested,
        symbols: symbols.length,
        symbolsProcessed: symbols,
        dateRange: { start: startDate, end: endDate },
        scriptOutput: stdout.substring(0, 500), // Truncate for logging
        timestamp: new Date().toISOString(),
      };

    } catch (error) {
      console.error(`[Databento] Script execution failed:`, error);
      throw new Error(`Databento ingestion failed: ${error.message}`);
    }
  },
});

/**
 * Schedule: Run daily at 6 PM UTC (after market close)
 * Start conservative: core agricultural symbols only
 */
export const databentoSchedule = schedules.task({
  id: "databento-daily-schedule",
  cron: "0 18 * * *", // 6 PM UTC daily
  task: databentoIngestJob,
  payload: {
    symbols: ["ZL", "ZS", "ZM", "CL", "HO"], // Core symbols only
    daysBack: 1,
  },
});

/**
 * Weekly backfill schedule: Run Sundays at 2 AM UTC
 * Pulls larger historical batches for all symbols
 */
export const databentoWeeklyBackfill = schedules.task({
  id: "databento-weekly-backfill",
  cron: "0 2 * * 0", // Sundays at 2 AM UTC
  task: databentoIngestJob,
  payload: {
    symbols: ALL_SYMBOLS, // All 38 symbols
    daysBack: 7, // Weekly batch
  },
});

