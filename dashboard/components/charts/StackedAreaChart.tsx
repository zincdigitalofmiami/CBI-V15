"use client";

import type { AreaData, IChartApi, UTCTimestamp } from "lightweight-charts";
import { useEffect, useMemo, useRef } from "react";
import { buildStackedAreaSeries, type StackedAreaPoint } from "./plugins/StackedAreaSeries";
import { cbiTextWatermarkOptions } from "./plugins/watermark";

type KeySpec = { id: string; name?: string; color: string };

type Props = {
  points: StackedAreaPoint[];
  keysInOrder: KeySpec[];
  height?: number;
};

export default function StackedAreaChart({ points, keysInOrder, height = 360 }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const toUtc = (t: number) => t as UTCTimestamp;

  const stacks = useMemo(() => buildStackedAreaSeries(points, keysInOrder), [points, keysInOrder]);

  useEffect(() => {
    if (!containerRef.current || points.length === 0 || keysInOrder.length === 0) return;

    let chart: IChartApi | null = null;
    let handleResize: (() => void) | null = null;

    import("lightweight-charts").then(({ createChart, AreaSeries, ColorType, createTextWatermark }) => {
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

      for (const s of stacks) {
        const area = chart.addSeries(AreaSeries, {
          lineColor: s.color,
          topColor: s.color.replace("rgb(", "rgba(").replace(")", ",0.25)"),
          bottomColor: "rgba(0,0,0,0)",
          lineWidth: 1,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
        });
        area.setData(s.data.map((p) => ({ time: toUtc(p.time), value: p.value })) as AreaData<UTCTimestamp>[]);
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
  }, [stacks, points.length, keysInOrder.length, height]);

  return <div ref={containerRef} className="w-full" style={{ height }} />;
}

