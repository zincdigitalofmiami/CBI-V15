"use client";

import { useEffect, useMemo, useRef } from "react";

type PricePoint = { time: number; value: number };
type ForecastPoint = { time: number; q10: number; q50: number; q90: number };

type Props = {
  price: { date: string | Date; close: number }[];
  forecasts: { target_date: string; q10: number; q50: number; q90: number; horizon_code: string }[];
};

export default function ForecastFanChart({ price, forecasts }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  const priceSeries: PricePoint[] = useMemo(
    () =>
      price.map((d) => ({
        time: Math.floor(new Date((d as any).date?.value || d.date).getTime() / 1000),
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

    let chart: any;
    let handleResize: () => void;

    import("lightweight-charts").then(({ createChart }) => {
      if (!containerRef.current) return;

      chart = createChart(containerRef.current, {
        layout: { background: { color: "#0a0e1a" }, textColor: "#9ca3af" },
        grid: { vertLines: { color: "#1f2937" }, horzLines: { color: "#1f2937" } },
        rightPriceScale: { borderVisible: false },
        timeScale: { borderVisible: false, secondsVisible: false },
        handleScroll: true,
        handleScale: true,
      });

      const priceLine = chart.addLineSeries({ color: "#e5e7eb", lineWidth: 2.0 });
      priceLine.setData(priceSeries);

      // Forecast lines (future only)
      if (futureSeries.length > 0) {
        const q50Line = chart.addLineSeries({ color: "#22c55e", lineWidth: 2, lineStyle: 0 });
        const q10Line = chart.addLineSeries({
          color: "rgba(34,197,94,0.35)",
          lineWidth: 1,
          lineStyle: 1,
        });
        const q90Line = chart.addLineSeries({
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

        q50Line.setData(q50);
        q10Line.setData(q10);
        q90Line.setData(q90);
      }

      handleResize = () => {
        if (containerRef.current && chart) {
          chart.applyOptions({
            width: containerRef.current.clientWidth,
            height: containerRef.current.clientHeight,
          });
        }
      };

      handleResize();
      window.addEventListener("resize", handleResize);
    });

    return () => {
      if (handleResize) {
        window.removeEventListener("resize", handleResize);
      }
      chart?.remove();
    };
  }, [priceSeries, futureSeries]);

  return <div ref={containerRef} className="h-full w-full" />;
}
