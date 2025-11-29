'use client';

import { ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useEffect, useState } from 'react';

export default function ZLChart() {
  const [mergedData, setMergedData] = useState([]);
  const [latestPrice, setLatestPrice] = useState(0);
  const [priceChange, setPriceChange] = useState(0);
  const [priceChangePct, setPriceChangePct] = useState(0);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState('');

  useEffect(() => {
    async function fetchData() {
      try {
        const [zlRes, shapRes] = await Promise.all([
          fetch('/api/live/zl'),
          fetch('/api/shap/zl')
        ]);
        
        const zlJson = await zlRes.json();
        const shapJson = await shapRes.json();
        
        const zlData = zlJson.data || [];
        const shapData = shapJson.data || [];

        const merged = zlData.map((zl: any) => {
          const shap = shapData.find((s: any) => s.date === zl.date.value);
          return {
            date: zl.date.value,
            close: zl.close,
            high: zl.high,
            low: zl.low,
            rins: shap?.RINs_momentum || 0,
            tariff: shap?.Tariff_risk || 0,
            drought: shap?.Drought_zscore || 0,
            crush: shap?.Crush_margin || 0
          };
        });

        setMergedData(merged);
        setLastUpdate(new Date().toLocaleTimeString());

        if (zlData.length > 0) {
          const latest = zlData[zlData.length - 1].close;
          setLatestPrice(latest);
          
          if (zlData.length > 1) {
            const prev = zlData[zlData.length - 2].close;
            const change = latest - prev;
            setPriceChange(change);
            setPriceChangePct((change / prev) * 100);
          }
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 300000);
    return () => clearInterval(interval);
  }, []);

  return (
    <main className="min-h-screen bg-[#0a0e1a]">
      {/* Header */}
      <div className="border-b border-[#1a1f2e] bg-[#0a0e1a]">
        <div className="max-w-[98%] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-thin text-white tracking-wide mb-1">CBI-V15 Dashboard</h1>
              <p className="text-gray-500 text-xs font-light">Soybean Oil Futures (ZL) • Live Databento Feed • Updates Every 5min</p>
            </div>
            <div className="flex items-baseline gap-3">
              <span className="text-5xl font-extralight text-white tracking-tight">${latestPrice.toFixed(2)}</span>
              <span className={`text-xl font-light ${priceChange >= 0 ? 'text-[#84cc16]' : 'text-red-400'}`}>
                {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)} ({priceChangePct >= 0 ? '+' : ''}{priceChangePct.toFixed(2)}%)
              </span>
            </div>
          </div>
          {lastUpdate && (
            <div className="text-right mt-2">
              <span className="text-xs text-gray-600 font-light">Last updated: {lastUpdate}</span>
            </div>
          )}
        </div>
      </div>

      {/* Main Chart - Full Width */}
      <div className="max-w-[98%] mx-auto px-6 py-6">
        <div className="bg-[#0d1220] rounded-lg border border-[#1a1f2e] p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-thin text-white tracking-wide">ZL Price Action + SHAP Force Drivers</h2>
            <div className="flex gap-4 text-xs font-light">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-[#84cc16] rounded-full"></div>
                <span className="text-gray-400">ZL Price (¢/lb)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-[#65a30d] rounded-full"></div>
                <span className="text-gray-400">RINs</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-[#a3e635] rounded-full"></div>
                <span className="text-gray-400">Tariff</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-[#bef264] rounded-full"></div>
                <span className="text-gray-400">Drought</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-[#d9f99d] rounded-full"></div>
                <span className="text-gray-400">Crush</span>
              </div>
            </div>
          </div>

          {!loading && mergedData.length > 0 ? (
            <div className="h-[600px]">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={mergedData} margin={{ top: 5, right: 60, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a1f2e" strokeWidth={0.5} />
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                    stroke="#4b5563"
                    tick={{ fill: '#6b7280', fontSize: 11, fontWeight: 300 }}
                    strokeWidth={0.5}
                  />
                  
                  {/* Left Y-Axis for Price */}
                  <YAxis 
                    yAxisId="left"
                    label={{ value: 'ZL Price (¢/lb)', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 11, fontWeight: 300 }} 
                    stroke="#4b5563"
                    tick={{ fill: '#6b7280', fontSize: 11, fontWeight: 300 }}
                    strokeWidth={0.5}
                  />
                  
                  {/* Right Y-Axis for SHAP */}
                  <YAxis 
                    yAxisId="right"
                    orientation="right"
                    label={{ value: 'SHAP Impact (¢/lb)', angle: 90, position: 'insideRight', fill: '#6b7280', fontSize: 11, fontWeight: 300 }} 
                    stroke="#4b5563"
                    tick={{ fill: '#6b7280', fontSize: 11, fontWeight: 300 }}
                    strokeWidth={0.5}
                    domain={[-15, 20]}
                  />
                  
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#0a0e1a', 
                      border: '1px solid #1a1f2e',
                      borderRadius: '6px',
                      color: '#ffffff',
                      fontSize: '12px',
                      fontWeight: 300
                    }}
                    labelFormatter={(value) => new Date(value).toLocaleDateString('en-US', { 
                      weekday: 'short', 
                      month: 'short', 
                      day: 'numeric',
                      year: 'numeric'
                    })}
                    formatter={(value: number, name: string) => {
                      if (name === 'ZL Price') return [`$${value.toFixed(2)}`, name];
                      return [`${value >= 0 ? '+' : ''}${value.toFixed(2)}¢`, name];
                    }}
                  />
                  
                  {/* ZL Price Line - Electric Lime */}
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="close" 
                    stroke="#84cc16" 
                    strokeWidth={1.5}
                    name="ZL Price"
                    dot={false}
                  />
                  
                  {/* SHAP Driver Lines - Lime/Green variations */}
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="rins" 
                    stroke="#65a30d" 
                    strokeWidth={1}
                    name="RINs"
                    dot={false}
                    strokeDasharray="4 4"
                  />
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="tariff" 
                    stroke="#a3e635" 
                    strokeWidth={1}
                    name="Tariff"
                    dot={false}
                    strokeDasharray="4 4"
                  />
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="drought" 
                    stroke="#bef264" 
                    strokeWidth={1}
                    name="Drought"
                    dot={false}
                    strokeDasharray="4 4"
                  />
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="crush" 
                    stroke="#d9f99d" 
                    strokeWidth={1}
                    name="Crush"
                    dot={false}
                    strokeDasharray="4 4"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-[600px] flex items-center justify-center text-gray-500">
              <p className="text-sm font-light">Loading live ZL data from Databento...</p>
            </div>
          )}

          {/* Stats Row */}
          <div className="mt-6 grid grid-cols-5 gap-4">
            <div className="bg-[#0a0e1a] rounded border border-[#1a1f2e] p-4">
              <p className="text-gray-500 text-xs font-light mb-1">Data Points</p>
              <p className="text-2xl font-extralight text-white">{mergedData.length}</p>
            </div>
            <div className="bg-[#0a0e1a] rounded border border-[#1a1f2e] p-4">
              <p className="text-gray-500 text-xs font-light mb-1">Project</p>
              <p className="text-2xl font-extralight text-[#84cc16]">cbi-v15</p>
            </div>
            <div className="bg-[#0a0e1a] rounded border border-[#1a1f2e] p-4">
              <p className="text-gray-500 text-xs font-light mb-1">Data Source</p>
              <p className="text-2xl font-extralight text-[#84cc16]">Databento</p>
            </div>
            <div className="bg-[#0a0e1a] rounded border border-[#1a1f2e] p-4">
              <p className="text-gray-500 text-xs font-light mb-1">Symbol</p>
              <p className="text-2xl font-extralight text-[#84cc16]">ZL</p>
            </div>
            <div className="bg-[#0a0e1a] rounded border border-[#1a1f2e] p-4">
              <p className="text-gray-500 text-xs font-light mb-1">Update Freq</p>
              <p className="text-2xl font-extralight text-[#84cc16]">5min</p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
