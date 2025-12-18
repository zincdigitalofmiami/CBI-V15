"use client";

import type { IChartApi, LineData, UTCTimestamp } from "lightweight-charts";
import { useEffect, useMemo, useRef, useState } from "react";
import { cbiTextWatermarkOptions } from "./plugins/watermark";

export type RangeKey = "1D" | "1W" | "1M" | "1Y";
export type RangeSeriesPoint = { time: number; value: number };

type Props = {
  seriesByRange: Record<RangeKey, RangeSeriesPoint[]>;
  height?: number;
  defaultRange?: RangeKey;
};

const RANGE_KEYS: RangeKey[] = ["1D", "1W", "1M", "1Y"];
const RANGE_COLORS: Record<RangeKey, string> = {
  "1D": "#2962FF",
  "1W": "rgb(225, 87, 90)",
  "1M": "rgb(242, 142, 44)",
  "1Y": "rgb(164, 89, 209)",
};

const toUtc = (t: number) => t as UTCTimestamp;

export default function RangeSwitcherChart({
  seriesByRange,
  height = 360,
  defaultRange = "1M",
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [range, setRange] = useState<RangeKey>(defaultRange);

  const data = useMemo(() => seriesByRange[range] || [], [seriesByRange, range]);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    let chart: IChartApi | null = null;
    let handleResize: (() => void) | null = null;

    import("lightweight-charts").then(({ createChart, ColorType, LineSeries, createTextWatermark }) => {
      if (!containerRef.current) return;

      chart = createChart(containerRef.current, {
        height,
        layout: {
          textColor: "#9ca3af",
          background: { type: ColorType.Solid, color: "#0a0e1a" },
        },
        grid: { vertLines: { color: "#1f2937" }, horzLines: { color: "#1f2937" } },
        rightPriceScale: { borderVisible: false },
        timeScale: { borderVisible: false, secondsVisible: false },
        handleScroll: true,
        handleScale: true,
      });

      const panes = chart.panes();
      if (panes.length > 0) createTextWatermark(panes[0], cbiTextWatermarkOptions());

      const lineSeries = chart.addSeries(LineSeries, { color: RANGE_COLORS[range], lineWidth: 2 });
      lineSeries.setData(data.map((p) => ({ time: toUtc(p.time), value: p.value })) as LineData<UTCTimestamp>[]);
      chart.timeScale().fitContent();

      handleResize = () => {
        if (!containerRef.current || !chart) return;
        chart.applyOptions({ width: containerRef.current.clientWidth, height });
      };
      handleResize();
      window.addEventListener("resize", handleResize);
    });

    return () => {
      if (handleResize) window.removeEventListener("resize", handleResize);
      chart?.remove?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height]);

  return (
    <div className="w-full">
      <div className="flex items-center gap-2 mb-3">
        {RANGE_KEYS.map((k) => (
          <button
            key={k}
            onClick={() => setRange(k)}
            className={
              "px-3 py-1.5 rounded-md text-sm font-extralight border transition-colors " +
              (k === range
                ? "bg-zinc-800 border-zinc-600 text-zinc-200"
                : "bg-zinc-950/40 border-zinc-800 text-zinc-500 hover:border-zinc-700 hover:text-zinc-300")
            }
          >
            {k}
          </button>
        ))}
      </div>

      {/* force chart re-init on range change for correctness */}
      <div key={range} ref={containerRef} className="w-full" style={{ height }} />

      {data.length === 0 && (
        <div className="text-xs text-zinc-600 font-extralight mt-2">No data for {range}</div>
      )}
    </div>
  );
}

