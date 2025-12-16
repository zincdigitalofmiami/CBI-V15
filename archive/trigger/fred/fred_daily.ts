import { logger, runs, task } from "@trigger.dev/sdk/v3";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

// Helper task to run individual FRED scripts
export const runFredScript = task({
  id: "run-fred-script",
  // Run in cloud
  machine: {
    preset: "small-1x",
  },
  run: async (payload: { script: string }) => {
    logger.info(`Running FRED script: ${payload.script} (CLOUD)`);
    
    try {
      const { stdout, stderr } = await execAsync(
        `python3 trigger/FRED/Scripts/${payload.script}`,
        { cwd: process.cwd() }
      );

      return { 
        script: payload.script, 
        status: "success",
        output: stdout.substring(0, 500) 
      };
    } catch (error) {
      logger.error(`FRED script ${payload.script} failed`, { error: String(error) });
      throw error;
    }
  },
});

// Main FRED daily update task
export const fredDailyUpdate = task({
  id: "fred-daily-update",
  // Run in cloud
  machine: {
    preset: "small-1x",
  },
  queue: {
    name: "fred-ingestion",
    concurrencyLimit: 3, // Max 3 FRED scripts in parallel
  },
  run: async () => {
    logger.info("Starting FRED daily update (3 scripts in parallel, CLOUD)");

    // Trigger all 3 FRED scripts in parallel
    const handles = await runs.batch([
      {
        payload: { script: "collect_fred_fx.py" },
        task: runFredScript
      },
      {
        payload: { script: "collect_fred_rates_curve.py" },
        task: runFredScript
      },
      {
        payload: { script: "collect_fred_financial_conditions.py" },
        task: runFredScript
      }
    ]);

    const results = await Promise.all(
      handles.map(h => h.run.waitUntil())
    );

    logger.info("FRED daily update complete", {
      scriptsRun: results.length,
      allSuccess: results.every(r => r.output?.status === "success")
    });

    return {
      status: "success",
      scriptsRun: results.length,
      timestamp: new Date().toISOString()
    };
  },
});
