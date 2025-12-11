/**
 * Multi-Source ETL Pipeline - Trigger.dev Orchestrator
 * 
 * Coordinates all data ingestion jobs in proper sequence:
 * 1. FRED economic data (seed + daily updates)
 * 2. Databento futures prices
 * 3. EIA biofuels data
 * 4. ProFarmer news
 * 5. ScrapeCreators news buckets
 * 6. News-to-signals processing
 * 
 * Based on: https://trigger.dev/docs/guides/use-cases/data-processing-etl#multi-source-etl-pipeline
 */

import { task } from "@trigger.dev/sdk/v3";
import { fredSeedHarvest } from "./FRED/Scripts/fred_seed_harvest";
import { profarmerDailyIngest } from "./ProFarmer/Scripts/profarmer_ingest_job";

interface ETLResult {
  source: string;
  success: boolean;
  recordsProcessed: number;
  duration: number;
  error?: string;
}

/**
 * Main ETL Orchestrator
 * Runs all ingestion jobs in parallel where possible
 */
export const multiSourceETL = task({
  id: "multi-source-etl",
  retry: {
    maxAttempts: 2,
    factor: 2,
    minTimeoutInMs: 5000,
    maxTimeoutInMs: 30000,
  },
  run: async (payload: { sources?: string[] }, { ctx }) => {
    const startTime = Date.now();
    const sourcesToRun = payload.sources || ["fred", "profarmer", "databento", "eia"];
    
    console.log(`[ETL] Starting multi-source pipeline for: ${sourcesToRun.join(", ")}`);
    
    const results: ETLResult[] = [];
    
    // Phase 1: Core data sources (can run in parallel)
    const phase1Jobs = [];
    
    if (sourcesToRun.includes("fred")) {
      phase1Jobs.push(
        fredSeedHarvest.trigger({ categories: ["fx", "rates", "macro", "credit", "financial_conditions"] })
          .then(result => ({
            source: "fred",
            success: true,
            recordsProcessed: result.seriesDiscovered || 0,
            duration: Date.now() - startTime,
          }))
          .catch(error => ({
            source: "fred",
            success: false,
            recordsProcessed: 0,
            duration: Date.now() - startTime,
            error: error.message,
          }))
      );
    }
    
    if (sourcesToRun.includes("profarmer")) {
      phase1Jobs.push(
        profarmerDailyIngest.trigger({ daysBack: 1 })
          .then(result => ({
            source: "profarmer",
            success: true,
            recordsProcessed: result.articlesIngested || 0,
            duration: Date.now() - startTime,
          }))
          .catch(error => ({
            source: "profarmer",
            success: false,
            recordsProcessed: 0,
            duration: Date.now() - startTime,
            error: error.message,
          }))
      );
    }
    
    // TODO: Add databento, eia, scrapecreators jobs
    
    // Wait for Phase 1 to complete
    const phase1Results = await Promise.all(phase1Jobs);
    results.push(...phase1Results);
    
    console.log(`[ETL] Phase 1 complete: ${phase1Results.length} sources processed`);
    
    // Phase 2: Derived data (depends on Phase 1)
    // - News-to-signals processing
    // - Feature engineering
    // TODO: Implement phase 2 jobs
    
    // Summary
    const totalRecords = results.reduce((sum, r) => sum + r.recordsProcessed, 0);
    const successCount = results.filter(r => r.success).length;
    const failureCount = results.filter(r => !r.success).length;
    
    console.log(`[ETL] Pipeline complete:`);
    console.log(`  - Total records: ${totalRecords}`);
    console.log(`  - Successful sources: ${successCount}`);
    console.log(`  - Failed sources: ${failureCount}`);
    console.log(`  - Duration: ${Date.now() - startTime}ms`);
    
    return {
      success: failureCount === 0,
      totalRecords,
      successCount,
      failureCount,
      results,
      duration: Date.now() - startTime,
      timestamp: new Date().toISOString(),
    };
  },
});

/**
 * Daily ETL Schedule
 * Runs at 7 AM UTC (2 AM ET) to capture overnight data
 */
import { schedules } from "@trigger.dev/sdk/v3";

export const dailyETLSchedule = schedules.task({
  id: "daily-etl-schedule",
  cron: "0 7 * * *", // 7 AM UTC daily
  task: multiSourceETL,
  payload: {
    sources: ["fred", "profarmer", "databento", "eia"],
  },
});
