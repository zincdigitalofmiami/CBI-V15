"use client";

import type { IChartApi, LineData, UTCTimestamp } from "lightweight-charts";
import { useEffect, useMemo, useRef } from "react";
import { cbiTextWatermarkOptions } from "./plugins/watermark";

export type YieldCurvePoint = { maturityMonths: number; value: number };
export type YieldCurve = { id: string; name?: string; color: string; points: YieldCurvePoint[] };

type Props = {
  curves: YieldCurve[];
  height?: number;
};

function maturityLabel(months: number): string {
  if (months < 12) return `${months}M`;
  const years = months / 12;
  if (Number.isInteger(years)) return `${years}Y`;
  return `${years.toFixed(1)}Y`;
}

/**
 * Lightweight Charts is time-series oriented; the "Yield Curve" demo uses a specialized helper.
 * Here we render the curve over a synthetic time axis and hide time labels, while we render our
 * own maturity labels beneath the chart.
 */
export default function YieldCurveChart({ curves, height = 420 }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const toUtc = (t: number) => t as UTCTimestamp;

  const maturities = useMemo(() => {
    const set = new Set<number>();
    for (const c of curves) for (const p of c.points) set.add(p.maturityMonths);
    return Array.from(set).sort((a, b) => a - b);
  }, [curves]);

  const seriesData = useMemo(() => {
    // Map maturity to synthetic timestamp (seconds) so the x-axis is ordered.
    // We will hide the time axis and show our own maturity labels.
    const toT = (m: number) => 1_000_000_000 + m * 86_400; // stable ordering
    return curves.map((c) => ({
      ...c,
      data: c.points
        .slice()
        .sort((a, b) => a.maturityMonths - b.maturityMonths)
        .map((p) => ({
          time: toUtc(toT(p.maturityMonths)),
          value: p.value,
          maturityMonths: p.maturityMonths,
        })),
    }));
  }, [curves]);

  useEffect(() => {
    if (!containerRef.current || seriesData.length === 0) return;

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
        grid: { vertLines: { color: "#111827" }, horzLines: { color: "#1f2937" } },
        rightPriceScale: { borderVisible: false },
        timeScale: {
          borderVisible: false,
          timeVisible: false,
          secondsVisible: false,
          ticksVisible: false,
        },
        handleScroll: false,
        handleScale: false,
      });

      const panes = chart.panes();
      if (panes.length > 0) createTextWatermark(panes[0], cbiTextWatermarkOptions({ line2: "YIELD CURVE" }));

      for (const c of seriesData) {
        const s = chart.addSeries(LineSeries, {
          color: c.color,
          lineWidth: 2,
          crosshairMarkerVisible: true,
          pointMarkersVisible: true,
        });
        s.setData(c.data as unknown as LineData<UTCTimestamp>[]);
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
  }, [seriesData, height]);

  return (
    <div className="w-full">
      <div ref={containerRef} className="w-full" style={{ height }} />
      <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-zinc-600 font-extralight">
        <span className="text-zinc-500">Maturities:</span>
        {maturities.map((m) => (
          <span key={m} className="px-2 py-0.5 rounded border border-zinc-800 bg-zinc-950/40">
            {maturityLabel(m)}
          </span>
        ))}
      </div>
    </div>
  );
}

