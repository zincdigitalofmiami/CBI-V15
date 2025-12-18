"use client";

import type { LineData, UTCTimestamp } from "lightweight-charts";
import { useEffect, useMemo, useRef } from "react";
import { cbiTextWatermarkOptions } from "../../../../components/charts/plugins/watermark";

interface LivePriceMiniProps {
  data: { time: number; value: number }[];
  height?: number;
  lineColor?: string;
  showAxis?: boolean;
}

export default function LivePriceMini({
  data,
  height = 80,
  lineColor = "#22c55e",
  showAxis = false,
}: LivePriceMiniProps) {
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<any>(null);
  const toUtc = (t: number) => t as UTCTimestamp;

  const sortedData = useMemo(() => {
    return [...data].sort((a, b) => a.time - b.time);
  }, [data]);

  useEffect(() => {
    if (!chartContainerRef.current || sortedData.length === 0) return;

    let chart: any;
    let lineSeries: any;

    import("lightweight-charts").then(({ createChart, ColorType, LineSeries, createTextWatermark }) => {
      if (!chartContainerRef.current) return;

      chart = createChart(chartContainerRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: "transparent" },
          textColor: "#6b7280",
        },
        grid: {
          vertLines: { visible: false },
          horzLines: { visible: false },
        },
        rightPriceScale: {
          visible: showAxis,
          borderVisible: false,
        },
        timeScale: {
          visible: showAxis,
          borderVisible: false,
        },
        handleScroll: false,
        handleScale: false,
        crosshair: {
          mode: 0, // hidden
        },
      });

      const panes = chart.panes();
      if (panes.length > 0) createTextWatermark(panes[0], cbiTextWatermarkOptions({ line2: "LIVE" }));

      lineSeries = chart.addSeries(LineSeries, {
        color: lineColor,
        lineWidth: 2,
        lineStyle: 0,
        priceLineVisible: false,
        lastValueVisible: false,
      });

      lineSeries.setData(
        sortedData.map((p) => ({ time: toUtc(p.time), value: p.value })) as LineData<UTCTimestamp>[],
      );
      chart.timeScale().fitContent();

      chartRef.current = chart;

      const handleResize = () => {
        if (chartContainerRef.current && chart) {
          chart.applyOptions({
            width: chartContainerRef.current.clientWidth,
            height,
          });
        }
      };

      handleResize();
      window.addEventListener("resize", handleResize);

      return () => {
        window.removeEventListener("resize", handleResize);
        chart?.remove();
      };
    });

    return () => {
      chartRef.current?.remove();
    };
  }, [sortedData, lineColor, height, showAxis]);

  if (sortedData.length === 0) {
    return (
      <div style={{ height }} className="flex items-center justify-center text-gray-500 text-xs">
        No data
      </div>
    );
  }

  return (
    <div
      ref={chartContainerRef}
      style={{ height, width: "100%" }}
      className="rounded overflow-hidden"
    />
  );
}
