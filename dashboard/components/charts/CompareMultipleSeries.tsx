"use client";

import type { IChartApi, LineData, UTCTimestamp } from "lightweight-charts";
import { useEffect, useMemo, useRef } from "react";
import { cbiTextWatermarkOptions } from "./plugins/watermark";

export type CompareSeriesPoint = { time: number; value: number };
export type CompareSeries = {
  id: string;
  name?: string;
  color?: string;
  data: CompareSeriesPoint[];
};

type Props = {
  series: CompareSeries[];
  height?: number;
};

const DEFAULT_COLORS = ["#2962FF", "rgb(225, 87, 90)", "rgb(242, 142, 44)", "rgb(164, 89, 209)"];

const toUtc = (t: number) => t as UTCTimestamp;

export default function CompareMultipleSeries({ series, height = 360 }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  const normalized = useMemo(() => {
    return (series || [])
      .filter((s) => Array.isArray(s.data) && s.data.length > 0)
      .map((s, idx) => ({
        ...s,
        color: s.color || DEFAULT_COLORS[idx % DEFAULT_COLORS.length],
      }));
  }, [series]);

  useEffect(() => {
    if (!containerRef.current || normalized.length === 0) return;

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

      for (const s of normalized) {
        const line = chart.addSeries(LineSeries, { color: s.color, lineWidth: 2 });
        line.setData(s.data.map((p) => ({ time: toUtc(p.time), value: p.value })) as LineData<UTCTimestamp>[]);
      }

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
  }, [normalized, height]);

  return <div ref={containerRef} className="w-full" style={{ height }} />;
}

