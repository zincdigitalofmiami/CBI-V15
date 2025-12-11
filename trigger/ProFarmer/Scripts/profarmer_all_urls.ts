/**
 * ProFarmer All URLs Scraper - Trigger.dev Job
 * 
 * PRIMARY ProFarmer ingestion job with comprehensive URL coverage:
 * - Daily Editions: First Thing Today, Ahead of the Open, After the Bell
 * - News Sections: Agriculture, Market, Policy, Weather
 * - Newsletters: Weekly Outlook
 * - Market Analysis: Grains, Livestock, Energy
 * - Commodity Reports: Soybeans, Soybean Oil, Soybean Meal, Corn, Wheat, Crude Oil
 * - Weather: Forecasts, Crop Conditions
 * 
 * Total: 22+ URLs scraped 3x daily (6 AM, 12 PM, 6 PM UTC)
 * 
 * Uses Anchor browser automation for authenticated access.
 * Writes to: raw.bucket_news
 */

import { schedules, task } from "@trigger.dev/sdk/v3";
import { MotherDuckClient } from "../../../src/shared/motherduck_client";

// ALL ProFarmer URLs to scrape (comprehensive coverage)
const PROFARMER_URLS = {
  // Daily Editions (CRITICAL - pre/post market)
  daily_editions: [
    { name: "First Thing Today", url: "/news/first-thing-today", edition_type: "pre_open", priority: "CRITICAL" },
    { name: "Ahead of the Open", url: "/news/ahead-of-the-open", edition_type: "pre_open", priority: "CRITICAL" },
    { name: "After the Bell", url: "/news/after-the-bell", edition_type: "post_close", priority: "CRITICAL" },
  ],
  
  // News Sections (HIGH - intraday updates)
  news: [
    { name: "Agriculture News", url: "/news/agriculture-news", edition_type: "intraday", priority: "HIGH" },
    { name: "Market News", url: "/news/markets", edition_type: "intraday", priority: "HIGH" },
    { name: "Policy News", url: "/news/policy", edition_type: "intraday", priority: "HIGH" },
    { name: "Weather News", url: "/news/weather", edition_type: "intraday", priority: "HIGH" },
  ],
  
  // Newsletters (HIGH - weekly analysis)
  newsletters: [
    { name: "Newsletters", url: "/newsletters", edition_type: "newsletter", priority: "HIGH" },
    { name: "Weekly Outlook", url: "/newsletters/weekly-outlook", edition_type: "newsletter", priority: "MEDIUM" },
  ],
  
  // Market Analysis (HIGH - commodity-specific)
  analysis: [
    { name: "Grain Analysis", url: "/analysis/grains", edition_type: "analysis", priority: "HIGH" },
    { name: "Livestock Analysis", url: "/analysis/livestock", edition_type: "analysis", priority: "MEDIUM" },
    { name: "Energy Analysis", url: "/analysis/energy", edition_type: "analysis", priority: "MEDIUM" },
  ],
  
  // Commodity Reports (CRITICAL - ZL/ZS/ZM focus)
  commodities: [
    { name: "Soybeans", url: "/markets/soybeans", edition_type: "commodity", priority: "CRITICAL" },
    { name: "Soybean Oil", url: "/markets/soybean-oil", edition_type: "commodity", priority: "CRITICAL" },
    { name: "Soybean Meal", url: "/markets/soybean-meal", edition_type: "commodity", priority: "CRITICAL" },
    { name: "Corn", url: "/markets/corn", edition_type: "commodity", priority: "HIGH" },
    { name: "Wheat", url: "/markets/wheat", edition_type: "commodity", priority: "MEDIUM" },
    { name: "Crude Oil", url: "/markets/crude-oil", edition_type: "commodity", priority: "HIGH" },
  ],
  
  // Weather (HIGH - crop conditions)
  weather: [
    { name: "Weather Forecast", url: "/weather/forecast", edition_type: "weather", priority: "HIGH" },
    { name: "Crop Conditions", url: "/weather/crop-conditions", edition_type: "weather", priority: "HIGH" },
  ],
};

const ALL_URLS = Object.values(PROFARMER_URLS).flat();

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
  priority: string;
  bucket_name: string;
}

/**
 * Scrape a single ProFarmer URL with Anchor
 */
async function scrapeProFarmerURL(
  urlConfig: { name: string; url: string; edition_type: string; priority: string }
): Promise<ProFarmerArticle[]> {
  const { Anchor } = await import("@ancorai/sdk");
  
  const anchor = new Anchor({
    apiKey: process.env.ANCHOR_API_KEY!,
  });

  try {
    // Login (if not already logged in)
    await anchor.goto("https://www.profarmer.com/login");
    await anchor.fill("#username", process.env.PROFARMER_USERNAME!);
    await anchor.fill("#password", process.env.PROFARMER_PASSWORD!);
    await anchor.click("button[type='submit']");
    await anchor.waitForNavigation();

    // Navigate to target URL
    await anchor.goto(`https://www.profarmer.com${urlConfig.url}`);

    // Extract articles
    const articles = await anchor.extract({
      schema: {
        articles: {
          type: "array",
          items: {
            type: "object",
            properties: {
              headline: { type: "string", selector: "h2, h3, .headline" },
              url: { type: "string", selector: "a[href]", attribute: "href" },
              author: { type: "string", selector: ".author, .byline" },
              date: { type: "string", selector: "time, .date" },
              summary: { type: "string", selector: "p, .excerpt, .summary" },
            },
          },
        },
      },
    });

    // Scrape full content (500 words max per article)
    const fullArticles: ProFarmerArticle[] = [];

    for (const article of articles.articles.slice(0, 20)) {
      try {
        await anchor.goto(article.url);

        const content = await anchor.extract({
          schema: {
            body: { type: "string", selector: ".article-body, article, .content" },
          },
        });

        // Limit to 500 words
        const words = content.body.split(/\s+/);
        const limitedContent = words.slice(0, 500).join(" ") + (words.length > 500 ? "..." : "");

        fullArticles.push({
          article_id: Buffer.from(article.url).toString("base64").slice(0, 32),
          headline: article.headline,
          content: limitedContent,
          author: article.author || "ProFarmer",
          source: "ProFarmer",
          source_trust_score: 0.95,
          published_at: article.date || new Date().toISOString(),
          url: article.url,
          edition_type: urlConfig.edition_type,
          priority: urlConfig.priority,
          bucket_name: "profarmer_all_urls",
        });
      } catch (error) {
        console.error(`[ProFarmer] Error scraping article ${article.url}:`, error);
      }
    }

    return fullArticles;
  } finally {
    await anchor.close();
  }
}

/**
 * Main Trigger.dev task: ProFarmer All URLs
 * 
 * Job Metadata:
 * - pipeline: "ingestion"
 * - domain: "media_ag_markets"
 * - source: "ProFarmer"
 * - buckets: ["Crush", "China", "Biofuel", "Weather", "Tariff"]
 * - method: "browser"
 * - frequency: "intraday" (3x daily)
 * - priority: "P1"
 */
export const profarmerAllURLs = task({
  id: "profarmer-all-urls",
  retry: {
    maxAttempts: 3,
    factor: 2,
    minTimeoutInMs: 10000,
    maxTimeoutInMs: 120000,
  },
  run: async (payload: { priorities?: string[] }, { ctx }) => {
    const priorities = payload.priorities || ["CRITICAL", "HIGH", "MEDIUM"];
    const urlsToScrape = ALL_URLS.filter((u) => priorities.includes(u.priority));

    console.log(`[ProFarmer All URLs] Scraping ${urlsToScrape.length} URLs (${ALL_URLS.length} total available)...`);

    const allArticles: ProFarmerArticle[] = [];
    const errors: string[] = [];

    for (const urlConfig of urlsToScrape) {
      try {
        console.log(`[ProFarmer] Scraping: ${urlConfig.name} (${urlConfig.priority})`);
        const articles = await scrapeProFarmerURL(urlConfig);
        allArticles.push(...articles);
        console.log(`[ProFarmer] Found ${articles.length} articles from ${urlConfig.name}`);
      } catch (error) {
        console.error(`[ProFarmer] Error scraping ${urlConfig.name}:`, error);
        errors.push(urlConfig.name);
      }
    }

    // Load to MotherDuck (raw.bucket_news)
    if (allArticles.length > 0) {
      const motherduck = new MotherDuckClient();
      await motherduck.insertBatch("raw.bucket_news", allArticles);
      console.log(`[ProFarmer] Loaded ${allArticles.length} articles to raw.bucket_news`);
    }

    return {
      success: errors.length === 0,
      articlesScraped: allArticles.length,
      urlsScraped: urlsToScrape.length - errors.length,
      totalURLs: ALL_URLS.length,
      errors: errors.length,
      errorURLs: errors,
      timestamp: new Date().toISOString(),
    };
  },
});

/**
 * Schedule: Run 3x daily (pre-market, intraday, post-market)
 * - 6 AM UTC: Pre-market (First Thing Today, Ahead of the Open)
 * - 12 PM UTC: Intraday (News, Analysis)
 * - 6 PM UTC: Post-market (After the Bell, Newsletters)
 */
export const profarmerAllURLsSchedule = schedules.task({
  id: "profarmer-all-urls-schedule",
  cron: "0 6,12,18 * * *", // 6 AM, 12 PM, 6 PM UTC
  task: profarmerAllURLs,
  payload: {
    priorities: ["CRITICAL", "HIGH"],
  },
});


