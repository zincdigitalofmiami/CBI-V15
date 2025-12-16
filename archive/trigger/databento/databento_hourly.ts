import { logger, task } from "@trigger.dev/sdk/v3";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

export const databentoDailyUpdate = task({
  id: "databento-hourly-update",
  // Run in Trigger.dev cloud - NOT local machine
  machine: {
    preset: "small-1x", // 0.5 vCPU, 1 GB RAM
  },
  queue: {
    name: "databento-ingestion",
    concurrencyLimit: 1, // One at a time for Databento API
  },
  run: async () => {
    logger.info("Starting Databento hourly update (CLOUD EXECUTION)");

    try {
      // Run existing Python script (incremental update - no args)
      const { stdout, stderr } = await execAsync(
        "python3 trigger/DataBento/Scripts/collect_daily.py",
        { cwd: process.cwd() }
      );

      logger.info("Databento update complete", { 
        stdout: stdout.substring(0, 500),
        stderr: stderr ? stderr.substring(0, 200) : null
      });

      return {
        status: "success",
        timestamp: new Date().toISOString(),
        output: stdout
      };
    } catch (error) {
      logger.error("Databento update failed", { error: String(error) });
      throw error;
    }
  },
});
