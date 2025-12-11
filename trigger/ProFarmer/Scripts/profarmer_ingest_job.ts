/**
 * ProFarmer News Ingestion - Trigger.dev Job
 * 
 * Scheduled job to scrape ProFarmer daily editions:
 * - First Thing Today (pre_open)
 * - Ahead of the Open (pre_open)
 * - After the Bell (post_close)
 * - Agriculture News (intraday)
 * - Newsletters (newsletter)
 * 
 * Uses Anchor browser automation for authenticated scraping.
 */

import path from "path";
import { task, schedules } from "@trigger.dev/sdk/v3";
import { MotherDuckClient } from "../../../src/shared/motherduck_client";

interface ProFarmerArticle {
  article_id: string;
  headline: string;
  content: string;
  author: string;
  source: string;
  source_trust_score: number;
  published_at: string;
  url: string;
  edition_type: string;
  bucket_name: string;
}

/**
 * Call Python ProFarmer scraper via subprocess
 */
async function runProFarmerScraper(): Promise<ProFarmerArticle[]> {
  const { spawn } = await import("child_process");
  
  return new Promise((resolve, reject) => {
    const python = spawn("python3", [
      path.join(__dirname, "profarmer_anchor.py"),
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
        reject(new Error(`ProFarmer scraper failed: ${stderr}`));
        return;
      }
      
      try {
        // Parse JSON output from Python script
        const articles = JSON.parse(stdout);
        resolve(articles);
      } catch (error) {
        reject(new Error(`Failed to parse ProFarmer output: ${error}`));
      }
    });
  });
}

/**
 * Main Trigger.dev task: ProFarmer Daily Ingestion
 */
export const profarmerDailyIngest = task({
  id: "profarmer-daily-ingest",
  retry: {
    maxAttempts: 3,
    factor: 2,
    minTimeoutInMs: 1000,
    maxTimeoutInMs: 30000,
  },
  run: async (payload: { daysBack?: number }, { ctx }) => {
    const daysBack = payload.daysBack || 1;
    
    console.log(`[ProFarmer] Starting daily ingestion (${daysBack} days back)...`);
    
    try {
      // Step 1: Scrape ProFarmer articles
      const articles = await runProFarmerScraper();
      
      console.log(`[ProFarmer] Scraped ${articles.length} articles`);
      
      if (articles.length === 0) {
        return {
          success: true,
          articlesIngested: 0,
          message: "No new articles found",
          timestamp: new Date().toISOString(),
        };
      }
      
      // Step 2: Load to MotherDuck
      const motherduck = new MotherDuckClient();
      
      const records = articles.map(a => ({
        date: new Date(a.published_at).toISOString().split('T')[0],
        article_id: a.article_id,
        headline: a.headline,
        content: a.content,
        author: a.author,
        source: a.source,
        source_trust_score: a.source_trust_score,
        url: a.url,
        bucket_name: a.bucket_name,
        edition_type: a.edition_type,
        published_at: a.published_at,
        created_at: new Date().toISOString(),
      }));
      
      await motherduck.insertBatch("raw.bucket_news", records);
      
      console.log(`[ProFarmer] Loaded ${records.length} articles to MotherDuck`);
      
      // Step 3: Log edition breakdown
      const editionCounts = records.reduce((acc, r) => {
        acc[r.edition_type] = (acc[r.edition_type] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);
      
      console.log(`[ProFarmer] Edition breakdown:`, editionCounts);
      
      return {
        success: true,
        articlesIngested: records.length,
        editionBreakdown: editionCounts,
        timestamp: new Date().toISOString(),
      };
      
    } catch (error) {
      console.error(`[ProFarmer] Error:`, error);
      throw error;
    }
  },
});

/**
 * Schedule: Run daily at 6 AM, 12 PM, 6 PM ET
 * (covers pre_open, intraday, post_close editions)
 */
export const profarmerSchedule = schedules.task({
  id: "profarmer-schedule",
  cron: "0 6,12,18 * * *", // 6 AM, 12 PM, 6 PM UTC (adjust for ET)
  task: profarmerDailyIngest,
});
