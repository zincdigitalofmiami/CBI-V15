/**
 * CFTC COT Weekly Ingestion - Trigger.dev Cloud Job
 * 
 * Fetches CFTC Commitment of Traders reports (Disaggregated + TFF).
 * Schedule: Every Friday at 4 PM ET (after CFTC 3:30 PM release).
 * 
 * Target tables:
 * - raw.cftc_cot (Disaggregated Futures - agricultural/energy/metals)
 * - raw.cftc_cot_tff (Traders in Financial Futures - FX/treasuries)
 * 
 * CRITICAL: Runs in Trigger.dev cloud, writes to MotherDuck ONLY.
 */

import { logger, task } from "@trigger.dev/sdk/v3";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

export const cftcWeeklyUpdate = task({
  id: "cftc-weekly-update",
  // Run in Trigger.dev cloud
  machine: {
    preset: "small-1x",
  },
  retry: {
    maxAttempts: 3,
    minTimeoutInMs: 5000,
    maxTimeoutInMs: 30000,
  },
  run: async (payload: { startYear?: number; endYear?: number } = {}) => {
    const currentYear = new Date().getFullYear();
    const startYear = payload.startYear || currentYear;
    const endYear = payload.endYear || currentYear;

    logger.info("[CLOUD] Starting CFTC COT weekly update", { startYear, endYear });

    try {
      const { stdout, stderr } = await execAsync(
        `python3 trigger/CFTC/Scripts/ingest_cot.py --start-year ${startYear} --end-year ${endYear}`,
        { 
          cwd: process.cwd(),
          timeout: 180000, // 3 minutes for CFTC download
        }
      );

      logger.info("[CLOUD] CFTC update complete", {
        stdout: stdout.substring(0, 500),
      });

      // Parse row counts from output
      const disaggMatch = stdout.match(/Inserted (\d+,?\d*) rows/);
      const rowsInserted = disaggMatch ? parseInt(disaggMatch[1].replace(/,/g, '')) : 0;

      return {
        status: "success",
        timestamp: new Date().toISOString(),
        startYear,
        endYear,
        rowsInserted,
        output: stdout.substring(0, 1000),
      };
    } catch (error) {
      logger.error("[CLOUD] CFTC update failed", { error: String(error) });
      throw error;
    }
  },
});

