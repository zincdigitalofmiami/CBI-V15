"use client";

import { ResponsiveLine } from "@nivo/line";

type Serie = {
  id: string;
  data: { x: string | number; y: number }[];
};

interface FundamentalChartProps {
  series: Serie[];
  height?: number;
}

const darkTheme = {
  background: "transparent",
  textColor: "#e5e7eb",
  fontSize: 12,
  axis: {
    domain: {
      line: {
        stroke: "#4b5563",
        strokeWidth: 1,
      },
    },
    ticks: {
      line: {
        stroke: "#4b5563",
        strokeWidth: 1,
      },
      text: {
        fill: "#9ca3af",
      },
    },
    legend: {
      text: {
        fill: "#e5e7eb",
      },
    },
  },
  grid: {
    line: {
      stroke: "#1f2937",
      strokeWidth: 1,
    },
  },
  crosshair: {
    line: {
      stroke: "#6b7280",
      strokeWidth: 1,
      strokeOpacity: 0.6,
    },
  },
  tooltip: {
    container: {
      background: "#020617",
      color: "#e5e7eb",
      fontSize: 12,
      borderRadius: 8,
      boxShadow: "0 10px 30px rgba(0,0,0,0.6)",
    },
  },
};

export function FundamentalChart({ series, height = 380 }: FundamentalChartProps) {
  if (!series || series.length === 0) {
    return null;
  }

  return (
    <div className="w-full rounded-2xl bg-black/80 border border-slate-800 p-4">
      <h2 className="text-sm font-medium text-slate-200 mb-3">Fundamental Graphs</h2>
      <div style={{ height }}>
        <ResponsiveLine
          data={series}
          theme={darkTheme as any}
          curve="monotoneX"
          margin={{ top: 20, right: 60, bottom: 40, left: 60 }}
          xScale={{ type: "point" }}
          yScale={{ type: "linear", stacked: false, min: "auto", max: "auto" }}
          axisBottom={{
            tickSize: 0,
            tickPadding: 10,
            legendOffset: 32,
          }}
          axisLeft={{
            tickSize: 0,
            tickPadding: 8,
            legendOffset: -50,
          }}
          enablePoints={true}
          pointSize={6}
          pointBorderWidth={2}
          pointBorderColor="#020617"
          pointColor="#22d3ee"
          useMesh={true}
          enableGridX={false}
          colors={["#22d3ee", "#a855f7", "#f97316", "#22c55e"]}
          enableSlices="x"
          sliceTooltip={({ slice }) => (
            <div className="px-3 py-2">
              <div className="font-medium mb-1">
                {slice.points[0].data.xFormatted ?? slice.points[0].data.x}
              </div>
              {slice.points.map((p) => (
                <div key={p.id} className="flex items-center gap-2 text-xs">
                  <span
                    className="inline-block w-2 h-2 rounded-full"
                    style={{ backgroundColor: (p as any).color ?? "#22d3ee" }}
                  />
                  <span>{(p as any).serieId ?? (p as any).seriesId ?? "Series"}</span>
                  <span className="ml-auto font-semibold">{p.data.yFormatted ?? p.data.y}</span>
                </div>
              ))}
            </div>
          )}
        />
      </div>
    </div>
  );
}

interface YieldCurveChartProps {
  curves: Serie[];
  height?: number;
}

export function YieldCurveChart({ curves, height = 420 }: YieldCurveChartProps) {
  if (!curves || curves.length === 0) {
    return null;
  }

  return (
    <div className="w-full rounded-2xl bg-black/80 border border-slate-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-slate-200">Yield Curves</h2>
        <div className="text-[10px] uppercase tracking-wide text-slate-500">Term structure (%)</div>
      </div>
      <div style={{ height }}>
        <ResponsiveLine
          data={curves}
          theme={darkTheme as any}
          curve="monotoneX"
          margin={{ top: 20, right: 20, bottom: 40, left: 40 }}
          xScale={{ type: "point" }}
          yScale={{ type: "linear", stacked: false }}
          axisBottom={{
            tickSize: 0,
            tickPadding: 12,
          }}
          axisLeft={{
            tickSize: 0,
            tickPadding: 8,
            format: (v) => `${v}%`,
          }}
          enablePoints={true}
          pointSize={6}
          pointBorderWidth={2}
          pointBorderColor="#020617"
          enableGridX={false}
          enableGridY={true}
          colors={["#a855f7", "#22c55e", "#38bdf8", "#f97316"]}
          useMesh={true}
          legends={[
            {
              anchor: "bottom",
              direction: "row",
              translateX: 0,
              translateY: 30,
              itemsSpacing: 12,
              itemWidth: 80,
              itemHeight: 16,
              itemTextColor: "#9ca3af",
              symbolSize: 10,
              symbolShape: "circle",
            },
          ]}
        />
      </div>
    </div>
  );
}
