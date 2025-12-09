/**
 * EIA Procurement Intelligence - Trigger.dev Job
 * 
 * Ingests EIA data critical for soybean oil procurement decisions:
 * 
 * BIOFUEL DEMAND (drives soybean oil demand):
 * - Soybean oil inputs to biodiesel production
 * - Renewable diesel production & capacity
 * - Biodiesel production capacity
 * - Competing feedstocks (corn oil, yellow grease, tallow)
 * 
 * ENERGY COSTS (manufacturing & transport):
 * - ULSD diesel spot prices (Gulf Coast, NY Harbor, LA)
 * - Heating oil prices (biodiesel competitor)
 * - Natural gas prices (plant energy)
 * - Crude oil prices (WTI, Brent)
 * 
 * PETROLEUM CONTEXT:
 * - Refinery utilization (indicates biofuel demand)
 * - Diesel inventories
 * - Petroleum imports
 */

import { schedules, task } from "@trigger.dev/sdk/v3";
import { MotherDuckClient } from "../src/shared/motherduck_client";

const EIA_API_KEY = process.env.EIA_API_KEY!;
const EIA_V2_BASE = "https://api.eia.gov/v2";

// Series configuration for soybean oil procurement intelligence
const EIA_SERIES_CONFIG = {
  // BIOFUEL FEEDSTOCKS (Monthly) - Critical for ZL demand
  biofuel_feedstocks: {
    endpoint: "/petroleum/pnp/feedbiofuel/data",
    frequency: "monthly",
    products: [
      "EPOOBDSOR",  // Soybean Oil for Renewable Diesel
      "EPOOBDBOR",  // Soybean Oil for Biodiesel
      "EPOOBDCNOR", // Corn Oil for RD (competitor)
      "EPOOBD4OR",  // Yellow Grease (competitor)
      "EPOOBD5OR",  // Tallow (competitor)
    ],
    description: "Feedstock inputs to biodiesel/renewable diesel production",
  },

  // SPOT PRICES (Weekly) - Energy costs
  spot_prices: {
    endpoint: "/petroleum/pri/spt/data",
    frequency: "weekly",
    series: [
      "EER_EPD2DXL0_PF4_RGC_DPG",   // Gulf Coast ULSD
      "EER_EPD2DXL0_PF4_Y35NY_DPG", // NY Harbor ULSD
      "EER_EPD2DXL0_PF4_Y05LA_DPG", // LA ULSD
      "EER_EPD2F_PF4_Y35NY_DPG",    // NY Heating Oil
      "EER_EPMRU_PF4_Y35NY_DPG",    // NY RBOB Gasoline
      "RWTC",                        // WTI Crude
    ],
    description: "Petroleum spot prices",
  },

  // REFINERY DATA (Weekly) - Demand indicator
  refinery: {
    endpoint: "/petroleum/pnp/wiup/data",
    frequency: "weekly",
    series: [
      "WPULEUS3",  // US Refinery Utilization %
      "WGFUPUS2",  // Gross Inputs to Refineries
    ],
    description: "Refinery utilization and inputs",
  },

  // DIESEL STOCKS (Weekly) - Supply indicator
  stocks: {
    endpoint: "/petroleum/stoc/wstk/data",
    frequency: "weekly",
    products: ["EPD2"],  // Distillate fuel oil
    description: "Diesel inventories",
  },
};

interface EIAResponse {
  response: {
    data: Array<{
      period: string;
      series?: string;
      product?: string;
      value: string | number | null;
      units?: string;
      [key: string]: unknown;
    }>;
  };
}

/**
 * Fetch data from EIA API v2
 */
async function fetchEIAData(
  endpoint: string,
  params: Record<string, string>
): Promise<EIAResponse> {
  const url = new URL(`${EIA_V2_BASE}${endpoint}`);
  url.searchParams.set("api_key", EIA_API_KEY);

  Object.entries(params).forEach(([key, value]) => {
    url.searchParams.set(key, value);
  });

  console.log(`[EIA] Fetching: ${endpoint}`);

  const response = await fetch(url.toString());
  if (!response.ok) {
    throw new Error(`EIA API error: ${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<EIAResponse>;
}

/**
 * Main ingestion task
 */
export const eiaProcurementIngest = task({
  id: "eia-procurement-ingest",
  retry: {
    maxAttempts: 3,
    factor: 2,
    minTimeoutInMs: 5000,
    maxTimeoutInMs: 60000,
  },
  run: async (payload: { daysBack?: number }, { ctx }) => {
    const daysBack = payload.daysBack || 30;
    const startDate = new Date(Date.now() - daysBack * 24 * 60 * 60 * 1000)
      .toISOString().split("T")[0];

    console.log(`[EIA] Starting procurement data ingestion from ${startDate}`);

    const allRecords: Array<{
      date: string;
      series_id: string;
      value: number;
      category: string;
      units: string;
    }> = [];

    // 1. Biofuel Feedstocks (Monthly)
    try {
      const feedstockData = await fetchEIAData(
        EIA_SERIES_CONFIG.biofuel_feedstocks.endpoint,
        {
          frequency: "monthly",
          "data[0]": "value",
          start: startDate.slice(0, 7),
          length: "500",
        }
      );

      for (const row of feedstockData.response.data) {
        if (row.value !== null && row.series) {
          allRecords.push({
            date: `${row.period}-01`,
            series_id: row.series,
            value: parseFloat(String(row.value)),
            category: "biofuel_feedstock",
            units: String(row.units || ""),
          });
        }
      }
      console.log(`[EIA] Fetched ${feedstockData.response.data.length} feedstock records`);
    } catch (e) {
      console.error("[EIA] Feedstock fetch error:", e);
    }

    // 2. Spot Prices (Weekly)
    try {
      const spotData = await fetchEIAData(
        EIA_SERIES_CONFIG.spot_prices.endpoint,
        {
          frequency: "weekly",
          "data[0]": "value",
          start: startDate,
          length: "500",
        }
      );

      for (const row of spotData.response.data) {
        if (row.value !== null && row.series) {
          allRecords.push({
            date: row.period,
            series_id: row.series,
            value: parseFloat(String(row.value)),
            category: "spot_price",
            units: String(row.units || "$/GAL"),
          });
        }
      }
      console.log(`[EIA] Fetched ${spotData.response.data.length} spot price records`);
    } catch (e) {
      console.error("[EIA] Spot price fetch error:", e);
    }

    // 3. Refinery Utilization (Weekly)
    try {
      const refineryData = await fetchEIAData(
        EIA_SERIES_CONFIG.refinery.endpoint,
        {
          frequency: "weekly",
          "data[0]": "value",
          start: startDate,
          length: "200",
        }
      );

      for (const row of refineryData.response.data) {
        if (row.value !== null && row.series) {
          allRecords.push({
            date: row.period,
            series_id: row.series,
            value: parseFloat(String(row.value)),
            category: "refinery",
            units: String(row.units || "%"),
          });
        }
      }
      console.log(`[EIA] Fetched ${refineryData.response.data.length} refinery records`);
    } catch (e) {
      console.error("[EIA] Refinery fetch error:", e);
    }

    // Load to MotherDuck
    if (allRecords.length > 0) {
      const motherduck = new MotherDuckClient();

      // Transform for eia_petroleum table
      const records = allRecords.map(r => ({
        date: r.date,
        series_id: r.series_id,
        value: r.value,
        category: r.category,
        units: r.units,
      }));

      await motherduck.insertBatch("raw.eia_petroleum", records);
      await motherduck.close();
      console.log(`[EIA] Loaded ${records.length} records to raw.eia_petroleum`);
    }

    return {
      success: true,
      recordsIngested: allRecords.length,
      categories: {
        biofuel_feedstock: allRecords.filter(r => r.category === "biofuel_feedstock").length,
        spot_price: allRecords.filter(r => r.category === "spot_price").length,
        refinery: allRecords.filter(r => r.category === "refinery").length,
      },
      timestamp: new Date().toISOString(),
    };
  },
});

/**
 * Weekly schedule - Runs every Wednesday at 2 PM UTC
 * (EIA releases weekly petroleum data on Wednesdays)
 */
export const eiaProcurementSchedule = schedules.task({
  id: "eia-procurement-weekly",
  cron: "0 14 * * 3", // Wednesday 2 PM UTC
  task: eiaProcurementIngest,
  payload: { daysBack: 14 },
});

