/**
 * Vegas Intel Ingestion - Trigger.dev Job
 * 
 * Scrapes Vegas-specific sources for demand signals:
 * - Eater Vegas: Restaurant developments
 * - LVCVA: Convention/event activity
 * - Nevada Tourism: Tourism arrivals, F&B demand
 * 
 * Why Vegas matters for ZL:
 * - Restaurant demand proxy for cooking oil consumption
 * - Convention/tourism activity correlates with F&B demand
 * - Leading indicator for broader consumer demand trends
 */

import { task, schedules } from "@trigger.dev/sdk/v3";
import { MotherDuckClient } from "../src/shared/motherduck_client";

interface VegasArticle {
  headline: string;
  content: string;
  url: string;
  source: string;
  source_trust_score: number;
  published_at: string;
  bucket_name: string;
}

/**
 * Run Python Vegas Intel collector
 */
async function runVegasIntelCollector(): Promise<VegasArticle[]> {
  const { spawn } = await import("child_process");
  
  return new Promise((resolve, reject) => {
    const python = spawn("python3", [
      "src/ingestion/buckets/vegas_intel/collect_vegas_intel.py",
    ]);
    
    let stdout = "";
    let stderr = "";
    
    python.stdout.on("data", (data) => {
      stdout += data.toString();
    });
    
    python.stderr.on("data", (data) => {
      stderr += data.toString();
    });
    
    python.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Vegas Intel collector failed: ${stderr}`));
        return;
      }
      
      try {
        const articles = JSON.parse(stdout);
        resolve(articles);
      } catch (error) {
        reject(new Error(`Failed to parse Vegas Intel output: ${error}`));
      }
    });
  });
}

/**
 * Main Trigger.dev task: Vegas Intel Ingestion
 */
export const vegasIntelJob = task({
  id: "vegas-intel-job",
  retry: {
    maxAttempts: 3,
    factor: 2,
    minTimeoutInMs: 5000,
    maxTimeoutInMs: 60000,
  },
  run: async (payload: {}, { ctx }) => {
    console.log("[Vegas Intel] Starting collection...");

    try {
      const articles = await runVegasIntelCollector();
      
      console.log(`[Vegas Intel] Collected ${articles.length} articles`);

      // Load to MotherDuck
      if (articles.length > 0) {
        const motherduck = new MotherDuckClient();
        await motherduck.insertBatch("raw.bucket_news", articles);
        console.log(`[Vegas Intel] Loaded ${articles.length} articles to MotherDuck`);
      }

      return {
        success: true,
        articlesCollected: articles.length,
        sources: {
          eater_vegas: articles.filter((a) => a.source === "Eater Vegas").length,
          lvcva: articles.filter((a) => a.source === "LVCVA").length,
          nevada_tourism: articles.filter((a) => a.source === "Nevada Tourism").length,
        },
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      console.error("[Vegas Intel] Error:", error);
      throw error;
    }
  },
});

/**
 * Schedule: Run weekly on Mondays at 8 AM UTC
 * (Vegas news doesn't change as frequently as commodity news)
 */
export const vegasIntelSchedule = schedules.task({
  id: "vegas-intel-schedule",
  cron: "0 8 * * 1", // Mondays at 8 AM UTC
  task: vegasIntelJob,
});

