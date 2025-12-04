"use client";

import { useEffect, useState } from "react";
import { ResponsiveChoropleth } from "@nivo/geo";
import worldData from "../../../world_countries.json";

type WeatherRow = {
  date: string;
  region: string;
  metric: string;
  value_mean: number;
};

type WeatherPoint = {
  id: string;
  value: number;
};

function mapRegionToCountryCode(region: string): string {
  // Coarse mapping: we only care about a few ag regions.
  if (region.startsWith("US_")) return "USA";
  if (region.startsWith("BRAZIL")) return "BRA";
  if (region.startsWith("ARGENTINA")) return "ARG";
  return "UNK";
}

export function WeatherChoropleth() {
  const [data, setData] = useState<WeatherPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [dateLabel, setDateLabel] = useState<string | null>(null);

  useEffect(() => {
    async function fetchWeather() {
      try {
        const res = await fetch("/api/weather/summary");
        const json = await res.json();
        if (!json.success || !json.data) {
          setLoading(false);
          return;
        }
        const rows: WeatherRow[] = json.data;
        if (rows.length === 0) {
          setLoading(false);
          return;
        }
        setDateLabel(rows[0].date);

        const byCountry = new Map<string, { sum: number; count: number }>();
        rows.forEach((r) => {
          const code = mapRegionToCountryCode(r.region);
          if (code === "UNK") return;
          const current = byCountry.get(code) || { sum: 0, count: 0 };
          current.sum += r.value_mean;
          current.count += 1;
          byCountry.set(code, current);
        });

        const points: WeatherPoint[] = Array.from(byCountry.entries()).map(
          ([code, agg]) => ({
            id: code,
            value: agg.sum / Math.max(agg.count, 1),
          }),
        );
        setData(points);
      } catch (e) {
        console.error("WeatherChoropleth fetch error", e);
      } finally {
        setLoading(false);
      }
    }
    fetchWeather();
  }, []);

  return (
    <div className="w-full rounded-2xl bg-black/80 border border-slate-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h2 className="text-sm font-medium text-slate-200">Weather – Key Ag Regions</h2>
          <p className="text-[10px] text-slate-500">
            NOAA anomalies (latest day) · TEMP_MEAN_C
          </p>
        </div>
        {dateLabel && (
          <span className="text-[10px] text-slate-500">
            As of {dateLabel}
          </span>
        )}
      </div>
      <div style={{ height: 320 }}>
        {loading ? (
          <div className="h-full flex items-center justify-center text-xs text-slate-500">
            Loading weather data…
          </div>
        ) : data.length === 0 ? (
          <div className="h-full flex items-center justify-center text-xs text-slate-500">
            Weather data not yet available.
          </div>
        ) : (
          <ResponsiveChoropleth
            data={data}
            // world_countries.json is a GeoJSON FeatureCollection
            features={(worldData as any).features}
            margin={{ top: 10, right: 10, bottom: 10, left: 10 }}
            colors="YlGnBu"
            domain={[-5, 5]}
            unknownColor="#020617"
            label="properties.name"
            valueFormat={(v) => `${v.toFixed(1)}°C`}
            projectionScale={120}
            projectionTranslation={[0.5, 0.55]}
            projectionRotation={[0, 0, 0]}
            borderWidth={0.5}
            borderColor="#0f172a"
            legends={[
              {
                anchor: "bottom-left",
                direction: "row",
                justify: false,
                translateX: 0,
                translateY: 20,
                itemsSpacing: 4,
                itemWidth: 60,
                itemHeight: 14,
                itemDirection: "left-to-right",
                itemTextColor: "#9ca3af",
                symbolSize: 10,
                symbolShape: "circle",
              },
            ]}
          />
        )}
      </div>
    </div>
  );
}
