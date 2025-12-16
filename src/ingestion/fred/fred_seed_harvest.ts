/**
 * FRED Seed Series Harvest - Trigger.dev Job
 *
 * Discovers and ingests FRED economic series using search API.
 * Search categories: FX, Rates, Macro, Credit, Financial Conditions
 *
 * Based on: https://trigger.dev/docs/guides/use-cases/data-processing-etl#multi-source-etl-pipeline
 */

import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

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
