/**
 * USDA Weekly Updates - Trigger.dev Cloud Jobs
 * 
 * Schedules:
 * - Export Sales: Every Thursday 9 AM ET (after 8:30 AM release)
 * - WASDE: Monthly on 12th at 1 PM ET (after 12 PM release)
 * 
 * Target tables:
 * - raw.usda_export_sales (Soybeans, Soybean Oil, Soybean Meal)
 * - raw.usda_wasde (World supply/demand estimates)
 * 
 * CRITICAL: Runs in Trigger.dev cloud, writes to MotherDuck ONLY.
 */

import { logger, task } from "@trigger.dev/sdk/v3";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

// Helper task: Run Export Sales script
export const usdaExportSalesUpdate = task({
  id: "usda-export-sales-update",
  machine: {
    preset: "small-1x",
  },
  run: async (payload: { startDate?: string; backfill?: boolean } = {}) => {
    const startDate = payload.startDate || "2024-01-01";
    const args = payload.backfill ? "--backfill" : `--start-date ${startDate}`;

    logger.info("[CLOUD] Starting USDA Export Sales update", { startDate, backfill: payload.backfill });

    try {
      const { stdout, stderr } = await execAsync(
        `python3 trigger/USDA/Scripts/ingest_export_sales.py ${args}`,
        { cwd: process.cwd(), timeout: 120000 }
      );

      logger.info("[CLOUD] Export Sales update complete");

      const rowsMatch = stdout.match(/Loaded (\d+) rows/);
      const rowsLoaded = rowsMatch ? parseInt(rowsMatch[1]) : 0;

      return {
        status: "success",
        timestamp: new Date().toISOString(),
        rowsLoaded,
      };
    } catch (error) {
      logger.error("[CLOUD] Export Sales update failed", { error: String(error) });
      throw error;
    }
  },
});

// Helper task: Run WASDE script
export const usdaWasdeUpdate = task({
  id: "usda-wasde-update",
  machine: {
    preset: "small-1x",
  },
  run: async (payload: { startYear?: number; backfill?: boolean } = {}) => {
    const currentYear = new Date().getFullYear();
    const startYear = payload.startYear || 2024;
    const args = payload.backfill 
      ? "--backfill" 
      : `--start-year ${startYear} --end-year ${currentYear}`;

    logger.info("[CLOUD] Starting USDA WASDE update", { startYear, backfill: payload.backfill });

    try {
      const { stdout, stderr } = await execAsync(
        `python3 trigger/USDA/Scripts/ingest_wasde.py ${args}`,
        { cwd: process.cwd(), timeout: 120000 }
      );

      logger.info("[CLOUD] WASDE update complete");

      const rowsMatch = stdout.match(/Loaded (\d+) rows/);
      const rowsLoaded = rowsMatch ? parseInt(rowsMatch[1]) : 0;

      return {
        status: "success",
        timestamp: new Date().toISOString(),
        rowsLoaded,
      };
    } catch (error) {
      logger.error("[CLOUD] WASDE update failed", { error: String(error) });
      throw error;
    }
  },
});

