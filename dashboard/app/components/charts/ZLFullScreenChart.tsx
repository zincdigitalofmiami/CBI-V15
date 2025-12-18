"use client";

import { useEffect, useRef, useState } from "react";

type OHLCVRow = {
  date?: string;
  as_of_date?: string;
  ts_event?: string;
  open?: number;
  high?: number;
  low?: number;
  close: number;
  volume?: number;
};


interface ZLFullScreenChartProps {
  height?: string;
}

export default function ZLFullScreenChart({ height = "100%" }: ZLFullScreenChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dataCount, setDataCount] = useState(0);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  useEffect(() => {
    let mounted = true;

    async function initChart() {
      if (!containerRef.current) return;

      try {
        // Dynamically import lightweight-charts
        const { createChart, CandlestickSeries, LineSeries, AreaSeries, ColorType } = await import("lightweight-charts");
        const { createTextWatermark } = await import("lightweight-charts");
        const { cbiTextWatermarkOptions } = await import("@/components/charts/plugins/watermark");

        // Fetch data from API
        const res = await fetch("/api/live/zl");
        const json = await res.json();

        if (!mounted) return;

        if (!json.data || !Array.isArray(json.data) || json.data.length === 0) {
          const errMsg = json.error || "No data from API";
          setError(errMsg);
          setLoading(false);
          return;
        }

        const rows = json.data as OHLCVRow[];
        setDataCount(rows.length);
        setLastRefresh(new Date());

        // Clear existing chart
        if (chartRef.current) {
          (chartRef.current as { remove: () => void }).remove();
        }
        containerRef.current.innerHTML = "";

        // Create chart
        const chart = createChart(containerRef.current, {
          layout: {
            background: { type: ColorType.Solid, color: "transparent" },
            textColor: "rgba(255, 255, 255, 0.5)",
            fontFamily: "system-ui, -apple-system, sans-serif",
          },
          grid: {
            vertLines: { color: "rgba(255, 255, 255, 0.03)" },
            horzLines: { color: "rgba(255, 255, 255, 0.03)" },
          },
          crosshair: {
            mode: 1,
            vertLine: {
              color: "rgba(255, 255, 255, 0.2)",
              width: 1,
              style: 2,
              labelBackgroundColor: "#131722",
            },
            horzLine: {
              color: "rgba(255, 255, 255, 0.2)",
              width: 1,
              style: 2,
              labelBackgroundColor: "#131722",
            },
          },
          timeScale: {
            borderColor: "rgba(255, 255, 255, 0.1)",
            timeVisible: true,
            secondsVisible: false,
          },
          rightPriceScale: {
            borderColor: "rgba(255, 255, 255, 0.1)",
            scaleMargins: { top: 0.1, bottom: 0.1 },
          },
          handleScroll: { mouseWheel: true, pressedMouseMove: true },
          handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
        });

        chartRef.current = chart;

        // Add watermark
        createTextWatermark(chart.panes()[0], cbiTextWatermarkOptions({ line2: "ZL FUTURES" }));

        // Check if we have OHLC data or just close prices
        const hasOHLC = rows.some(r => r.open !== undefined && r.high !== undefined && r.low !== undefined);

        if (hasOHLC) {
          // Use candlestick series
          const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: "#22c55e",
            downColor: "#ef4444",
            borderUpColor: "#22c55e",
            borderDownColor: "#ef4444",
            wickUpColor: "#22c55e",
            wickDownColor: "#ef4444",
          });

          const candleData = rows
            .map((row) => {
              const dateStr = row.date || row.as_of_date || row.ts_event;
              if (!dateStr) return null;
              const time = Math.floor(new Date(dateStr).getTime() / 1000);
              if (isNaN(time)) return null;
              return {
                time: time as import("lightweight-charts").UTCTimestamp,
                open: row.open ?? row.close,
                high: row.high ?? row.close,
                low: row.low ?? row.close,
                close: row.close,
              };
            })
            .filter((d): d is NonNullable<typeof d> => d !== null)
            .sort((a, b) => (a.time as number) - (b.time as number));

          candleSeries.setData(candleData);
        } else {
          // Use area series for close-only data
          const areaSeries = chart.addSeries(AreaSeries, {
            lineColor: "#06b6d4",
            topColor: "rgba(6, 182, 212, 0.3)",
            bottomColor: "rgba(6, 182, 212, 0.0)",
            lineWidth: 2,
          });

          const areaData = rows
            .map((row) => {
              const dateStr = row.date || row.as_of_date || row.ts_event;
              if (!dateStr) return null;
              const time = Math.floor(new Date(dateStr).getTime() / 1000);
              if (isNaN(time)) return null;
              return {
                time: time as import("lightweight-charts").UTCTimestamp,
                value: row.close,
              };
            })
            .filter((d): d is NonNullable<typeof d> => d !== null)
            .sort((a, b) => (a.time as number) - (b.time as number));

          areaSeries.setData(areaData);
        }

        chart.timeScale().fitContent();

        // Handle resize
        const handleResize = () => {
          if (containerRef.current && chartRef.current) {
            (chartRef.current as { applyOptions: (opts: { width: number; height: number }) => void }).applyOptions({
              width: containerRef.current.clientWidth,
              height: containerRef.current.clientHeight,
            });
          }
        };

        window.addEventListener("resize", handleResize);
        handleResize();

        setLoading(false);

        return () => {
          window.removeEventListener("resize", handleResize);
        };
      } catch (err) {
        console.error("Chart error:", err);
        if (mounted) {
          setError(err instanceof Error ? err.message : "Failed to load chart");
          setLoading(false);
        }
      }
    }

    initChart();

    // Refresh every 15 minutes
    const interval = setInterval(() => {
      initChart();
    }, 15 * 60 * 1000);

    return () => {
      mounted = false;
      clearInterval(interval);
      if (chartRef.current) {
        (chartRef.current as { remove: () => void }).remove();
      }
    };
  }, []);

  return (
    <div className="relative w-full" style={{ height }}>
      {/* Chart container */}
      <div ref={containerRef} className="absolute inset-0" />

      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#070a12]/80">
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-zinc-400 text-sm font-extralight">Loading chart data…</span>
          </div>
        </div>
      )}

      {/* Error overlay */}
      {error && !loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#070a12]/80">
          <div className="text-center">
            <div className="text-red-400 text-lg mb-2">⚠️ {error}</div>
            <div className="text-zinc-500 text-sm">Check API connection</div>
          </div>
        </div>
      )}

      {/* Info badge */}
      {!loading && !error && dataCount > 0 && (
        <div className="absolute top-4 left-4 z-10 px-3 py-1.5 rounded-lg bg-black/40 backdrop-blur-sm border border-white/10">
          <span className="text-xs text-zinc-400 font-extralight">
            ZL Soybean Oil • {dataCount} bars
            {lastRefresh && <span className="ml-2 text-zinc-500">• {lastRefresh.toLocaleTimeString()}</span>}
          </span>
        </div>
      )}
    </div>
  );
}
