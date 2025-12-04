'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

type Candle = { time: number; value: number };

export default function ZLChart() {
  const [zlData, setZlData] = useState<any[]>([]);
  const [latestPrice, setLatestPrice] = useState(0);
  const [priceChange, setPriceChange] = useState(0);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch('/api/live/zl');
        const json = await res.json();
        
        if (json.data && json.data.length > 0) {
          setZlData(json.data);
          const latest = json.data[json.data.length - 1].close;
          const prev = json.data.length > 1 ? json.data[json.data.length - 2].close : latest;
          setLatestPrice(latest);
          setPriceChange(((latest - prev) / prev) * 100);
          setLastUpdate(new Date());
        }
      } catch (error) {
        console.error('Error fetching ZL data:', error);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 300000); // 5 minutes
    return () => clearInterval(interval);
  }, []);

  const seriesData: Candle[] = useMemo(() => {
    return zlData.map((d) => ({
      time: Math.floor(new Date(d.date?.value || d.date).getTime() / 1000),
      value: d.close,
    }));
  }, [zlData]);

  const chartContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current || seriesData.length === 0) return;

    let chart: any;
    let lineSeries: any;

    import('lightweight-charts').then(({ createChart }) => {
      chart = createChart(chartContainerRef.current!, {
        layout: { background: { color: '#0a0e1a' }, textColor: '#9ca3af' },
        grid: {
          vertLines: { color: '#1f2937' },
          horzLines: { color: '#1f2937' },
        },
        rightPriceScale: { borderVisible: false },
        timeScale: { borderVisible: false, secondsVisible: false },
        handleScroll: true,
        handleScale: true,
      });

      lineSeries = chart.addLineSeries({
        color: '#ffffff',
        lineWidth: 2.5,
        lineStyle: 0,
      });

      lineSeries.setData(seriesData);

      const handleResize = () => {
        if (chartContainerRef.current) {
          chart.applyOptions({
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
          });
        }
      };

      handleResize();
      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        chart?.remove();
      };
    });
  }, [seriesData]);

  return (
    <main className="h-screen w-screen bg-[#0a0e1a] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-8 py-4 border-b border-[#1a1f2e]">
        <div>
          <h1 className="text-2xl font-thin text-white tracking-wide">ZL Soybean Oil Futures</h1>
          <p className="text-gray-500 text-xs font-light">Live • Databento API • 5min updates</p>
        </div>
        <div className="flex items-baseline gap-4">
          <span className="text-5xl font-extralight text-white">${latestPrice.toFixed(2)}</span>
          <span className={`text-xl font-light ${priceChange >= 0 ? 'text-[#22c55e]' : 'text-red-400'}`}>
            {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
          </span>
          <span className="text-sm font-medium text-[#22c55e] animate-pulse">● LIVE</span>
        </div>
      </div>

      {/* Full-screen chart */}
      <div className="flex-1 p-4">
        {loading ? (
          <div className="h-full flex items-center justify-center">
            <p className="text-gray-500 font-light">Loading live ZL data...</p>
          </div>
        ) : zlData.length > 0 ? (
          <div ref={chartContainerRef} className="h-full w-full" />
        ) : (
          <div className="h-full flex items-center justify-center">
            <p className="text-red-400 font-light">No data available</p>
          </div>
        )}
      </div>

      {lastUpdate && (
        <div className="px-8 py-2 text-right border-t border-[#1a1f2e]">
          <span className="text-xs text-gray-600">Last updated: {lastUpdate.toLocaleTimeString()}</span>
        </div>
      )}
    </main>
  );
}
