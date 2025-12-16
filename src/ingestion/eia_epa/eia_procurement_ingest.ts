/**
 * EIA Biofuels Procurement Intelligence - Trigger.dev Job
 *
 * Ingests EIA data critical for soybean oil procurement decisions.
 *
 * BIOFUEL DEMAND (drives soybean oil demand):
 * - Soybean oil inputs to biodiesel/renewable diesel production
 * - Competing feedstocks (corn oil, yellow grease, tallow)
 * - Biodiesel production capacity
 *
 * NOTE:
 * - This job now **persists only biofuel/biodiesel-related series** into
 *   `raw.eia_biofuels`.
 * - Petroleum spot prices, refinery utilization, and diesel inventory endpoints
 *   are kept for potential future use, but are **not** written to any raw table
 *   until a dedicated energy table is defined.
 */

import { logger, task } from "@trigger.dev/sdk/v3";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

const EIA_API_KEY = process.env.EIA_API_KEY!;
const EIA_V2_BASE = "https://api.eia.gov/v2";

// Series configuration for soybean oil procurement intelligence
const EIA_SERIES_CONFIG = {
  // BIOFUEL FEEDSTOCKS (Monthly) - Critical for ZL demand
  biofuel_feedstocks: {
    endpoint: "/petroleum/pnp/feedbiofuel/data",
    frequency: "monthly",
    products: [
      "EPOOBDSOR", // Soybean Oil for Renewable Diesel
      "EPOOBDBOR", // Soybean Oil for Biodiesel
      "EPOOBDCNOR", // Corn Oil for RD (competitor)
      "EPOOBD4OR", // Yellow Grease (competitor)
      "EPOOBD5OR", // Tallow (competitor)
    ],
    description: "Feedstock inputs to biodiesel/renewable diesel production",
  },

};

export const eiaProcurementIngest = task({
  id: "eia-procurement-ingest",
  // Run in cloud
  machine: {
    preset: "small-1x",
  },
  run: async () => {
    logger.info("Starting EIA procurement data ingestion (CLOUD EXECUTION)");

    try {
      // Run Python script for EIA biofuels data
      const { stdout, stderr } = await execAsync(
        "python3 trigger/EIA_EPA/Scripts/collect_eia_biofuels.py",
        { cwd: process.cwd() }
      );

      logger.info("EIA biofuels ingestion complete", {
        stdout: stdout.substring(0, 500),
        stderr: stderr ? stderr.substring(0, 200) : null
      });

      return {
        status: "success",
        timestamp: new Date().toISOString(),
        output: stdout
      };
    } catch (error) {
      logger.error("EIA biofuels ingestion failed", { error: String(error) });
      throw error;
    }
  },
});

// NOTE: Scheduled via daily orchestrator, no individual schedule needed
