/**
 * EPA RIN Prices Weekly Update - Trigger.dev Cloud Job
 * 
 * Fetches weekly RIN prices (D3, D4, D5, D6) for biodiesel tracking.
 * 
 * Data sources (priority order):
 * 1. OPIS API (requires OPIS_API_KEY) - paid, ~$2000/year
 * 2. Historical estimates (fallback if no OPIS key)
 * 
 * Target: raw.epa_rin_prices
 * 
 * CRITICAL for ZL forecasting: D4 RINs drive soybean oil demand for biodiesel.
 * CRITICAL: Runs in Trigger.dev cloud, writes to MotherDuck ONLY.
 */

import { logger, task } from "@trigger.dev/sdk/v3";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

export const epaRinWeeklyUpdate = task({
  id: "epa-rin-weekly-update",
  machine: {
    preset: "small-1x",
  },
  run: async () => {
    logger.info("[CLOUD] Starting EPA RIN prices update");

    // Warn if OPIS key not set
    if (!process.env.OPIS_API_KEY) {
      logger.warn("[CLOUD] OPIS_API_KEY not set - using historical estimates");
      logger.warn("[CLOUD] For real RIN prices, set OPIS_API_KEY in Trigger.dev dashboard");
    }

    try {
      const { stdout, stderr } = await execAsync(
        "python3 trigger/EIA_EPA/Scripts/collect_epa_rin_prices.py",
        { cwd: process.cwd(), timeout: 60000 }
      );

      logger.info("[CLOUD] EPA RIN update complete");

      const rowsMatch = stdout.match(/Loaded (\d+) rows/);
      const rowsLoaded = rowsMatch ? parseInt(rowsMatch[1]) : 0;

      return {
        status: "success",
        timestamp: new Date().toISOString(),
        rowsLoaded,
        dataSource: process.env.OPIS_API_KEY ? "OPIS_API" : "HISTORICAL_ESTIMATES",
      };
    } catch (error) {
      logger.error("[CLOUD] EPA RIN update failed", { error: String(error) });
      throw error;
    }
  },
});

