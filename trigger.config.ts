/**
 * Trigger.dev Configuration for CBI-V15
 * 
 * Orchestrates all ETL and training jobs:
 * - FRED economic data ingestion
 * - ProFarmer news scraping
 * - Databento futures data
 * - EIA biofuels data
 * - News-to-signals processing
 * - Model training pipelines
 */

import { defineConfig } from "@trigger.dev/sdk/v3";

export default defineConfig({
  project: "cbi-v15",

  // Runtime configuration
  runtime: "node",

  // Retry defaults
  retries: {
    enabledInDev: true,
    default: {
      maxAttempts: 3,
      minTimeoutInMs: 1000,
      maxTimeoutInMs: 10000,
      factor: 2,
    },
  },

  // Directories
  dirs: ["./trigger"],

  // Integrations
  integrations: {
    openai: {
      apiKey: process.env.OPENAI_API_KEY,
    },
    anchor: {
      apiKey: process.env.ANCHOR_API_KEY,
    },
  },

  // Environment variables (set in Vercel)
  // These are loaded automatically from process.env
  // - TRIGGER_SECRET_KEY
  // - MOTHERDUCK_TOKEN
  // - MOTHERDUCK_DB
  // - DATABENTO_API_KEY
  // - FRED_API_KEY
  // - EIA_API_KEY
  // - SCRAPECREATORS_API_KEY
  // - PROFARMER_USERNAME
  // - PROFARMER_PASSWORD
  // - OPENAI_API_KEY
  // - ANCHOR_API_KEY
});

