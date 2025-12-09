/**
 * TradingEconomics GOLDMINE Scraper - Trigger.dev Job
 * 
 * CRITICAL: This page is a GOLD MINE for commodity data and China trade balance.
 * 
 * Commodities (CRITICAL):
 * - Soybeans, Crude Oil, Copper, Palm Oil
 * - Natural Gas, Heating Oil, Canola, Wheat
 * - Rapeseed Oil, Ethanol, Methanol, Propane
 * 
 * Indices:
 * - Baltic Dry Index, CRB Index, GSCI Index
 * 
 * China Trade (CRITICAL FOR BIG 4):
 * - Balance of Trade (imports/exports)
 * 
 * Uses Anchor browser automation for authenticated access.
 */

import { task, schedules } from "@trigger.dev/sdk/v3";
import { MotherDuckClient } from "../src/shared/motherduck_client";

// ALL TradingEconomics URLs (GOLD MINE)
const TRADINGECONOMICS_URLS = {
  // Commodities (CRITICAL)
  commodities_critical: [
    { name: "Soybeans", url: "/commodity/soybeans", bucket: "crush", priority: "CRITICAL" },
    { name: "Crude Oil", url: "/commodity/crude-oil", bucket: "energy", priority: "CRITICAL" },
    { name: "Copper", url: "/commodity/copper", bucket: "china", priority: "CRITICAL" },
    { name: "Palm Oil", url: "/commodity/palm-oil", bucket: "crush", priority: "CRITICAL" },
  ],
  
  // Energy
  energy: [
    { name: "Natural Gas", url: "/commodity/natural-gas", bucket: "energy", priority: "HIGH" },
    { name: "Heating Oil", url: "/commodity/heating-oil", bucket: "energy", priority: "HIGH" },
  ],
  
  // Oilseeds & Grains
  oilseeds: [
    { name: "Canola", url: "/commodity/canola", bucket: "crush", priority: "HIGH" },
    { name: "Wheat", url: "/commodity/wheat", bucket: "crush", priority: "MEDIUM" },
    { name: "Rapeseed Oil", url: "/commodity/rapeseed-oil", bucket: "crush", priority: "HIGH" },
  ],
  
  // Biofuels
  biofuels: [
    { name: "Ethanol", url: "/commodity/ethanol", bucket: "biofuel", priority: "HIGH" },
    { name: "Methanol", url: "/commodity/methanol", bucket: "biofuel", priority: "MEDIUM" },
    { name: "Propane", url: "/commodity/propane", bucket: "biofuel", priority: "MEDIUM" },
  ],
  
  // Indices
  indices: [
    { name: "Baltic Dry Index", url: "/commodity/baltic", bucket: "china", priority: "HIGH" },
    { name: "CRB Index", url: "/commodity/crb", bucket: "volatility", priority: "MEDIUM" },
    { name: "GSCI Index", url: "/commodity/gsci", bucket: "volatility", priority: "MEDIUM" },
  ],
  
  // China Trade (CRITICAL FOR BIG 4)
  china_trade: [
    { 
      name: "China Balance of Trade", 
      url: "/china/balance-of-trade", 
      bucket: "china", 
      priority: "CRITICAL",
      note: "CRITICAL FOR CHRIS'S BIG 4 - China imports/exports"
    },
  ],
};

const ALL_URLS = Object.values(TRADINGECONOMICS_URLS).flat();

interface TradingEconomicsData {
  commodity: string;
  url: string;
  bucket_name: string;
  priority: string;
  
  // Price data
  current_price: number;
  price_change: number;
  price_change_pct: number;
  
  // Historical data
  high_52w: number;
  low_52w: number;
  
  // Forecast (if available)
  forecast_1m?: number;
  forecast_3m?: number;
  forecast_1y?: number;
  
  // News/Analysis
  latest_news: string;
  analysis: string;
  
  // Metadata
  scraped_at: string;
}

/**
 * Scrape a single TradingEconomics URL with Anchor
 */
async function scrapeTradingEconomicsURL(
  urlConfig: { name: string; url: string; bucket: string; priority: string }
): Promise<TradingEconomicsData> {
  const { Anchor } = await import("@ancorai/sdk");
  
  const anchor = new Anchor({
    apiKey: process.env.ANCHOR_API_KEY!,
  });

  try {
    // Navigate to commodity page
    await anchor.goto(`https://tradingeconomics.com${urlConfig.url}`);

    // Extract data with AI-powered selectors
    const data = await anchor.extract({
      schema: {
        current_price: { type: "number", selector: ".actual, .price-value" },
        price_change: { type: "number", selector: ".change, .price-change" },
        price_change_pct: { type: "number", selector: ".change-pct, .percentage" },
        high_52w: { type: "number", selector: ".high-52w, .year-high" },
        low_52w: { type: "number", selector: ".low-52w, .year-low" },
        forecast_1m: { type: "number", selector: ".forecast-1m" },
        forecast_3m: { type: "number", selector: ".forecast-3m" },
        forecast_1y: { type: "number", selector: ".forecast-1y" },
        latest_news: { type: "string", selector: ".news-item:first-child h3, .latest-news" },
        analysis: { type: "string", selector: ".analysis, .commentary" },
      },
    });

    return {
      commodity: urlConfig.name,
      url: `https://tradingeconomics.com${urlConfig.url}`,
      bucket_name: urlConfig.bucket,
      priority: urlConfig.priority,
      current_price: data.current_price || 0,
      price_change: data.price_change || 0,
      price_change_pct: data.price_change_pct || 0,
      high_52w: data.high_52w || 0,
      low_52w: data.low_52w || 0,
      forecast_1m: data.forecast_1m,
      forecast_3m: data.forecast_3m,
      forecast_1y: data.forecast_1y,
      latest_news: data.latest_news || "",
      analysis: data.analysis || "",
      scraped_at: new Date().toISOString(),
    };
  } finally {
    await anchor.close();
  }
}

/**
 * Main Trigger.dev task: TradingEconomics GOLDMINE
 */
export const tradingeconomicsGoldmine = task({
  id: "tradingeconomics-goldmine",
  retry: {
    maxAttempts: 3,
    factor: 2,
    minTimeoutInMs: 10000,
    maxTimeoutInMs: 120000,
  },
  run: async (payload: { priorities?: string[] }, { ctx }) => {
    const priorities = payload.priorities || ["CRITICAL", "HIGH", "MEDIUM"];
    const urlsToScrape = ALL_URLS.filter((u) => priorities.includes(u.priority));

    console.log(`[TradingEconomics GOLDMINE] Scraping ${urlsToScrape.length} URLs...`);

    const allData: TradingEconomicsData[] = [];
    const errors: string[] = [];

    for (const urlConfig of urlsToScrape) {
      try {
        console.log(`[TradingEconomics] Scraping: ${urlConfig.name} (${urlConfig.priority})`);
        const data = await scrapeTradingEconomicsURL(urlConfig);
        allData.push(data);
        console.log(`[TradingEconomics] ${urlConfig.name}: $${data.current_price} (${data.price_change_pct}%)`);
        
        // Rate limiting
        await new Promise((resolve) => setTimeout(resolve, 2000));
      } catch (error) {
        console.error(`[TradingEconomics] Error scraping ${urlConfig.name}:`, error);
        errors.push(urlConfig.name);
      }
    }

    // Load to MotherDuck
    if (allData.length > 0) {
      const motherduck = new MotherDuckClient();
      await motherduck.insertBatch("raw.tradingeconomics_commodities", allData);
      console.log(`[TradingEconomics] Loaded ${allData.length} commodities to MotherDuck`);
    }

    return {
      success: errors.length === 0,
      commoditiesScraped: allData.length,
      errors: errors.length,
      errorCommodities: errors,
      timestamp: new Date().toISOString(),
    };
  },
});

/**
 * Schedule: Run daily at 7 AM UTC (after Asian markets close, before US open)
 */
export const tradingeconomicsSchedule = schedules.task({
  id: "tradingeconomics-goldmine-schedule",
  cron: "0 7 * * *", // 7 AM UTC
  task: tradingeconomicsGoldmine,
  payload: {
    priorities: ["CRITICAL", "HIGH"],
  },
});

