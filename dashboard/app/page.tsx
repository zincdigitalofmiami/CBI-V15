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

        // Merge ZL and SHAP data by date
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
    // PULL EVERY 5 MINUTES (300,000ms)
    const interval = setInterval(fetchData, 300000);
    return () => clearInterval(interval);
  }, []);

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <div className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm">
        <div className="max-w-[98%] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white mb-1">CBI-V15 Dashboard</h1>
              <p className="text-slate-400 text-sm">Soybean Oil Futures (ZL) • Live Databento Feed • Updates Every 5min</p>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-5xl font-bold text-white">${latestPrice.toFixed(2)}</span>
              <span className={`text-xl font-semibold ${priceChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)} ({priceChangePct >= 0 ? '+' : ''}{priceChangePct.toFixed(2)}%)
              </span>
            </div>
          </div>
          {lastUpdate && (
            <div className="text-right mt-2">
              <span className="text-xs text-slate-500">Last updated: {lastUpdate}</span>
            </div>
          )}
        </div>
      </div>

      {/* Main Chart - Full Width */}
      <div className="max-w-[98%] mx-auto px-6 py-6">
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl shadow-2xl border border-slate-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-semibold text-white">ZL Price Action + SHAP Force Drivers</h2>
            <div className="flex gap-3 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span className="text-slate-300">ZL Price (¢/lb)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                <span className="text-slate-300">RINs Momentum</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <span className="text-slate-300">Tariff Risk</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-cyan-500 rounded-full"></div>
                <span className="text-slate-300">Drought Z-Score</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="text-slate-300">Crush Margin</span>
              </div>
            </div>
          </div>

          {!loading && mergedData.length > 0 ? (
            <div className="h-[600px]">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={mergedData} margin={{ top: 5, right: 60, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                    stroke="#94a3b8"
                    tick={{ fill: '#94a3b8' }}
                  />
                  
                  {/* Left Y-Axis for Price */}
                  <YAxis 
                    yAxisId="left"
                    label={{ value: 'ZL Price (¢/lb)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} 
                    stroke="#94a3b8"
                    tick={{ fill: '#94a3b8' }}
                  />
                  
                  {/* Right Y-Axis for SHAP */}
                  <YAxis 
                    yAxisId="right"
                    orientation="right"
                    label={{ value: 'SHAP Impact (¢/lb)', angle: 90, position: 'insideRight', fill: '#94a3b8' }} 
                    stroke="#94a3b8"
                    tick={{ fill: '#94a3b8' }}
                    domain={[-15, 20]}
                  />
                  
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1e293b', 
                      border: '1px solid #475569',
                      borderRadius: '8px',
                      color: '#e2e8f0'
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
                  
                  {/* ZL Price Line */}
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="close" 
                    stroke="#3b82f6" 
                    strokeWidth={3}
                    name="ZL Price"
                    dot={false}
                  />
                  
                  {/* SHAP Driver Lines */}
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="rins" 
                    stroke="#f97316" 
                    strokeWidth={2}
                    name="RINs"
                    dot={false}
                    strokeDasharray="5 5"
                  />
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="tariff" 
                    stroke="#ef4444" 
                    strokeWidth={2}
                    name="Tariff"
                    dot={false}
                    strokeDasharray="5 5"
                  />
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="drought" 
                    stroke="#06b6d4" 
                    strokeWidth={2}
                    name="Drought"
                    dot={false}
                    strokeDasharray="5 5"
                  />
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="crush" 
                    stroke="#10b981" 
                    strokeWidth={2}
                    name="Crush"
                    dot={false}
                    strokeDasharray="5 5"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-[600px] flex items-center justify-center text-slate-400">
              <p className="text-lg">Loading live ZL data from Databento...</p>
            </div>
          )}

          {/* Stats Row */}
          <div className="mt-6 grid grid-cols-5 gap-4">
            <div className="bg-slate-700/50 backdrop-blur-sm rounded-lg p-4 border border-slate-600">
              <p className="text-slate-400 text-xs mb-1">Data Points</p>
              <p className="text-2xl font-bold text-white">{mergedData.length}</p>
            </div>
            <div className="bg-blue-900/30 backdrop-blur-sm rounded-lg p-4 border border-blue-700/50">
              <p className="text-slate-400 text-xs mb-1">Project</p>
              <p className="text-2xl font-bold text-blue-400">cbi-v15</p>
            </div>
            <div className="bg-purple-900/30 backdrop-blur-sm rounded-lg p-4 border border-purple-700/50">
              <p className="text-slate-400 text-xs mb-1">Data Source</p>
              <p className="text-2xl font-bold text-purple-400">Databento</p>
            </div>
            <div className="bg-green-900/30 backdrop-blur-sm rounded-lg p-4 border border-green-700/50">
              <p className="text-slate-400 text-xs mb-1">Symbol</p>
              <p className="text-2xl font-bold text-green-400">ZL</p>
            </div>
            <div className="bg-orange-900/30 backdrop-blur-sm rounded-lg p-4 border border-orange-700/50">
              <p className="text-slate-400 text-xs mb-1">Update Freq</p>
              <p className="text-2xl font-bold text-orange-400">5min</p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
