"use client";

import type { IChartApi, LineData, UTCTimestamp } from "lightweight-charts";
import { useEffect, useMemo, useRef } from "react";
import { buildPercentBands } from "./plugins/BandsIndicator";
import { cbiTextWatermarkOptions } from "./plugins/watermark";

type Props = {
  data: { time: number; value: number }[];
  bandPct?: number;
  height?: number;
};

export default function BandsChart({ data, bandPct = 0.02, height = 360 }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const bands = useMemo(() => buildPercentBands(data, bandPct), [data, bandPct]);
  const toUtc = (t: number) => t as UTCTimestamp;

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

      // Bands as two faint lines + filled area between them (approximated with an area series on upper)
      const upperLine = chart.addSeries(LineSeries, { color: "rgba(34,197,94,0.35)", lineWidth: 1 });
      const lowerLine = chart.addSeries(LineSeries, { color: "rgba(34,197,94,0.35)", lineWidth: 1 });
      upperLine.setData(
        bands.map((b) => ({ time: toUtc(b.time), value: b.upper })) as LineData<UTCTimestamp>[],
      );
      lowerLine.setData(
        bands.map((b) => ({ time: toUtc(b.time), value: b.lower })) as LineData<UTCTimestamp>[],
      );

      const priceLine = chart.addSeries(LineSeries, { color: "#e5e7eb", lineWidth: 2 });
      priceLine.setData(data.map((p) => ({ time: toUtc(p.time), value: p.value })) as LineData<UTCTimestamp>[]);

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
  }, [data, bands, height]);

  return <div ref={containerRef} className="w-full" style={{ height }} />;
}

