/**
 * News-to-Signals Processing - Trigger.dev Job
 *
 * Calls Python script for sentiment analysis and signal processing.
 */

import { logger, task } from "@trigger.dev/sdk/v3";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

export const newsToSignalsAgent = task({
  id: "news-to-signals-agent",
  // Run in cloud
  machine: {
    preset: "small-1x",
  },
  run: async () => {
    logger.info("Starting news-to-signals processing (CLOUD EXECUTION)");

    try {
      // Run Python script for sentiment calculation
      const { stdout, stderr } = await execAsync(
        "python3 trigger/ScrapeCreators/Scripts/sentiment_calculator.py",
        { cwd: process.cwd() }
      );

      logger.info("News sentiment processing complete", {
        stdout: stdout.substring(0, 500),
        stderr: stderr ? stderr.substring(0, 200) : null
      });

      return {
        status: "success",
        timestamp: new Date().toISOString(),
        output: stdout
      };
    } catch (error) {
      logger.error("News sentiment processing failed", { error: String(error) });
      throw error;
    }
  },
});
