"use client";

import type { LineData, UTCTimestamp } from "lightweight-charts";
import { useEffect, useMemo, useRef } from "react";
import { cbiTextWatermarkOptions } from "./plugins/watermark";

type PricePoint = { time: number; value: number };
type ForecastPoint = { time: number; q10: number; q50: number; q90: number };

type Props = {
  price: { date: string | Date; close: number }[];
  forecasts: { target_date: string; q10: number; q50: number; q90: number; horizon_code: string }[];
};

export default function ForecastFanChart({ price, forecasts }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  const toUtc = (t: number) => t as UTCTimestamp;

  const asEpochSec = (d: unknown): number => {
    // supports: Date | string | { value: string } (MotherDuck-style)
    if (d instanceof Date) return Math.floor(d.getTime() / 1000);
    if (typeof d === "string") return Math.floor(new Date(d).getTime() / 1000);
    if (typeof d === "object" && d !== null && "value" in d) {
      const v = (d as { value?: unknown }).value;
      if (typeof v === "string") return Math.floor(new Date(v).getTime() / 1000);
    }
    return NaN;
  };

  const priceSeries: PricePoint[] = useMemo(
    () =>
      price.map((d) => ({
        time: asEpochSec(d.date),
        value: d.close,
      })),
    [price],
  );

  const futureSeries: ForecastPoint[] = useMemo(
    () =>
      forecasts
        .sort((a, b) => new Date(a.target_date).getTime() - new Date(b.target_date).getTime())
        .map((d) => ({
          time: Math.floor(new Date(d.target_date).getTime() / 1000),
          q10: d.q10,
          q50: d.q50,
          q90: d.q90,
        })),
    [forecasts],
  );

  useEffect(() => {
    if (!containerRef.current || priceSeries.length === 0) return;

    let chart: any = null;
    let handleResize: (() => void) | null = null;

    import("lightweight-charts").then(({ createChart, ColorType, LineSeries, createTextWatermark }) => {
      if (!containerRef.current) return;

      chart = createChart(containerRef.current, {
        layout: { background: { type: ColorType.Solid, color: "#0a0e1a" }, textColor: "#9ca3af" },
        grid: { vertLines: { color: "#1f2937" }, horzLines: { color: "#1f2937" } },
        rightPriceScale: { borderVisible: false },
        timeScale: { borderVisible: false, secondsVisible: false },
        handleScroll: true,
        handleScale: true,
      });

      const panes = chart?.panes?.() ?? [];
      if (Array.isArray(panes) && panes.length > 0) {
        createTextWatermark(panes[0] as never, cbiTextWatermarkOptions({ line2: "FORECAST FAN" }));
      }

      const priceLine = (chart as any).addSeries(LineSeries, { color: "#e5e7eb", lineWidth: 2 });
      priceLine.setData(
        priceSeries
          .filter((p) => Number.isFinite(p.time) && Number.isFinite(p.value))
          .map((p) => ({ time: toUtc(p.time), value: p.value })) as LineData<UTCTimestamp>[],
      );

      // Forecast lines (future only)
      if (futureSeries.length > 0) {
        const q50Line = (chart as any).addSeries(LineSeries, { color: "#22c55e", lineWidth: 2, lineStyle: 0 });
        const q10Line = (chart as any).addSeries(LineSeries, {
          color: "rgba(34,197,94,0.35)",
          lineWidth: 1,
          lineStyle: 1,
        });
        const q90Line = (chart as any).addSeries(LineSeries, {
          color: "rgba(34,197,94,0.35)",
          lineWidth: 1,
          lineStyle: 1,
        });

        // Build lines connecting last price to first forecast, then across horizons
        const last = priceSeries[priceSeries.length - 1];
        const q50 = [
          { time: last.time, value: last.value },
          ...futureSeries.map((p) => ({ time: p.time, value: p.q50 })),
        ];
        const q10 = [
          { time: last.time, value: last.value },
          ...futureSeries.map((p) => ({ time: p.time, value: p.q10 })),
        ];
        const q90 = [
          { time: last.time, value: last.value },
          ...futureSeries.map((p) => ({ time: p.time, value: p.q90 })),
        ];

        q50Line.setData(q50.map((p) => ({ time: toUtc(p.time), value: p.value })) as LineData<UTCTimestamp>[]);
        q10Line.setData(q10.map((p) => ({ time: toUtc(p.time), value: p.value })) as LineData<UTCTimestamp>[]);
        q90Line.setData(q90.map((p) => ({ time: toUtc(p.time), value: p.value })) as LineData<UTCTimestamp>[]);
      }

      handleResize = () => {
        if (!containerRef.current || !chart?.applyOptions) return;
        chart.applyOptions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      };

      handleResize();
      window.addEventListener("resize", handleResize);
    });

    return () => {
      if (handleResize) window.removeEventListener("resize", handleResize);
      chart?.remove?.();
    };
  }, [priceSeries, futureSeries]);

  return <div ref={containerRef} className="h-full w-full" />;
}
