/**
 * Intelligent News Pipeline - Trigger.dev Orchestrator
 * 
 * End-to-end news processing pipeline:
 * 1. Scrape news with Anchor browser automation
 * 2. Process with OpenAI Agents for signal extraction
 * 3. Store signals in MotherDuck
 * 4. Trigger downstream feature engineering
 * 
 * Combines:
 * - Anchor browser scraping
 * - OpenAI Agents with guardrails
 * - Multi-source ETL orchestration
 */

import { task, schedules } from "@trigger.dev/sdk/v3";
import { profarmerAnchorScraper } from "./profarmer_anchor_scraper";
import { newsToSignalsAgent } from "./news_to_signals_openai_agent";

/**
 * Main Intelligent News Pipeline
 */
export const intelligentNewsPipeline = task({
  id: "intelligent-news-pipeline",
  retry: {
    maxAttempts: 2,
    factor: 2,
    minTimeoutInMs: 10000,
    maxTimeoutInMs: 60000,
  },
  run: async (payload: { skipScraping?: boolean }, { ctx }) => {
    const startTime = Date.now();

    console.log("[Intelligent News] Starting pipeline...");

    // Phase 1: Scrape news with Anchor (if not skipped)
    let scrapingResult = null;
    if (!payload.skipScraping) {
      console.log("[Intelligent News] Phase 1: Scraping with Anchor...");
      
      try {
        scrapingResult = await profarmerAnchorScraper.trigger({
          sections: ["First Thing Today", "Ahead of the Open", "After the Bell"],
        });
        
        console.log(`[Intelligent News] Scraped ${scrapingResult.articlesScraped} articles`);
      } catch (error) {
        console.error("[Intelligent News] Scraping failed:", error);
        // Continue to processing phase even if scraping fails
      }
    }

    // Phase 2: Process news with OpenAI Agent
    console.log("[Intelligent News] Phase 2: Processing with OpenAI Agent...");
    
    let processingResult = null;
    try {
      processingResult = await newsToSignalsAgent.trigger({
        batchSize: 50,
        lookbackHours: 24,
      });
      
      console.log(`[Intelligent News] Processed ${processingResult.articlesProcessed} articles`);
    } catch (error) {
      console.error("[Intelligent News] Processing failed:", error);
    }

    // Phase 3: Trigger downstream jobs (TODO)
    // - Feature engineering
    // - Model retraining
    // - Dashboard updates

    const duration = Date.now() - startTime;

    return {
      success: true,
      scrapingResult,
      processingResult,
      duration,
      timestamp: new Date().toISOString(),
    };
  },
});

/**
 * Schedule: Run every 6 hours
 * Covers pre-open, intraday, and post-close news cycles
 */
export const intelligentNewsPipelineSchedule = schedules.task({
  id: "intelligent-news-pipeline-schedule",
  cron: "0 */6 * * *", // Every 6 hours
  task: intelligentNewsPipeline,
});

/**
 * Pre-Market News Pipeline
 * Runs at 6 AM UTC (1 AM ET) to capture overnight news
 */
export const preMarketNewsPipeline = schedules.task({
  id: "pre-market-news-pipeline",
  cron: "0 6 * * *", // 6 AM UTC daily
  task: intelligentNewsPipeline,
  payload: {
    skipScraping: false,
  },
});

/**
 * Post-Market News Pipeline
 * Runs at 9 PM UTC (4 PM ET) to capture closing news
 */
export const postMarketNewsPipeline = schedules.task({
  id: "post-market-news-pipeline",
  cron: "0 21 * * *", // 9 PM UTC daily
  task: intelligentNewsPipeline,
  payload: {
    skipScraping: false,
  },
});

