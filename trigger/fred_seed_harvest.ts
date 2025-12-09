/**
 * FRED Seed Series Harvest - Trigger.dev Job
 * 
 * Discovers and ingests FRED economic series using search API.
 * Search categories: FX, Rates, Macro, Credit, Financial Conditions
 * 
 * Based on: https://trigger.dev/docs/guides/use-cases/data-processing-etl#multi-source-etl-pipeline
 */

import { task } from "@trigger.dev/sdk/v3";
import { MotherDuckClient } from "../src/shared/motherduck_client";

// FRED API configuration
const FRED_API_KEY = process.env.FRED_API_KEY!;
const FRED_BASE_URL = "https://api.stlouisfed.org/fred";

// Search terms by category
const SEARCH_CATEGORIES = {
  fx: ["trade weighted", "DEX", "DXY", "BRL", "ARS", "CNY", "MXN"],
  rates: ["treasury constant maturity", "federal funds", "sofr", "term premium", "yield curve"],
  macro: ["GDP", "CPI", "PCE", "employment", "industrial production", "M2"],
  credit: ["corporate bond spread", "high yield", "credit spread"],
  financial_conditions: ["stress index", "risk appetite", "volatility index", "NFCI", "STLFSI"],
};

interface FREDSeries {
  id: string;
  title: string;
  observation_start: string;
  observation_end: string;
  frequency: string;
  units: string;
  seasonal_adjustment: string;
  last_updated: string;
  popularity: number;
  notes: string;
}

interface FREDSearchResult {
  seriess: FREDSeries[];
  count: number;
}

/**
 * Search FRED for series matching a query
 */
async function searchFREDSeries(
  searchText: string,
  limit: number = 100
): Promise<FREDSeries[]> {
  const url = new URL(`${FRED_BASE_URL}/series/search`);
  url.searchParams.append("api_key", FRED_API_KEY);
  url.searchParams.append("search_text", searchText);
  url.searchParams.append("limit", limit.toString());
  url.searchParams.append("file_type", "json");
  url.searchParams.append("order_by", "popularity");
  url.searchParams.append("sort_order", "desc");

  const response = await fetch(url.toString());
  if (!response.ok) {
    throw new Error(`FRED API error: ${response.statusText}`);
  }

  const data: FREDSearchResult = await response.json();
  return data.seriess || [];
}

/**
 * Fetch observations for a FRED series
 */
async function fetchFREDObservations(
  seriesId: string,
  startDate?: string
): Promise<any[]> {
  const url = new URL(`${FRED_BASE_URL}/series/observations`);
  url.searchParams.append("api_key", FRED_API_KEY);
  url.searchParams.append("series_id", seriesId);
  url.searchParams.append("file_type", "json");
  
  if (startDate) {
    url.searchParams.append("observation_start", startDate);
  }

  const response = await fetch(url.toString());
  if (!response.ok) {
    throw new Error(`FRED API error for ${seriesId}: ${response.statusText}`);
  }

  const data = await response.json();
  return data.observations || [];
}

/**
 * Main Trigger.dev task: FRED Seed Series Harvest
 */
export const fredSeedHarvest = task({
  id: "fred-seed-harvest",
  retry: {
    maxAttempts: 3,
    factor: 2,
    minTimeoutInMs: 1000,
    maxTimeoutInMs: 10000,
  },
  run: async (payload: { categories?: string[] }, { ctx }) => {
    const categoriesToRun = payload.categories || Object.keys(SEARCH_CATEGORIES);
    
    console.log(`[FRED Harvest] Starting for categories: ${categoriesToRun.join(", ")}`);
    
    const allSeries: Map<string, FREDSeries> = new Map();
    
    // Step 1: Search and discover series
    for (const category of categoriesToRun) {
      const searchTerms = SEARCH_CATEGORIES[category as keyof typeof SEARCH_CATEGORIES];
      
      if (!searchTerms) {
        console.warn(`[FRED Harvest] Unknown category: ${category}`);
        continue;
      }
      
      console.log(`[FRED Harvest] Searching ${category} with ${searchTerms.length} terms...`);
      
      for (const term of searchTerms) {
        try {
          const series = await searchFREDSeries(term, 50);
          
          console.log(`[FRED Harvest] Found ${series.length} series for "${term}"`);
          
          // Deduplicate by series ID
          for (const s of series) {
            if (!allSeries.has(s.id)) {
              allSeries.set(s.id, { ...s, category });
            }
          }
          
          // Rate limit: 120 requests/minute for FRED API
          await new Promise(resolve => setTimeout(resolve, 500));
          
        } catch (error) {
          console.error(`[FRED Harvest] Error searching "${term}":`, error);
        }
      }
    }
    
    console.log(`[FRED Harvest] Discovered ${allSeries.size} unique series`);
    
    // Step 2: Store series metadata in MotherDuck
    const motherduck = new MotherDuckClient();
    
    const seriesMetadata = Array.from(allSeries.values()).map(s => ({
      series_id: s.id,
      title: s.title,
      category: s.category,
      frequency: s.frequency,
      units: s.units,
      seasonal_adjustment: s.seasonal_adjustment,
      observation_start: s.observation_start,
      observation_end: s.observation_end,
      last_updated: s.last_updated,
      popularity: s.popularity,
      notes: s.notes,
      discovered_at: new Date().toISOString(),
    }));
    
    await motherduck.insertBatch("raw.fred_series_metadata", seriesMetadata);
    
    console.log(`[FRED Harvest] Stored ${seriesMetadata.length} series metadata records`);
    
    return {
      success: true,
      seriesDiscovered: allSeries.size,
      categories: categoriesToRun,
      timestamp: new Date().toISOString(),
    };
  },
});

