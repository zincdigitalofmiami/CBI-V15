'use client';

import dynamic from 'next/dynamic';
import { useEffect, useState } from 'react';

// Dynamic import for Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function ZLChart() {
  const [zlData, setZlData] = useState<any[]>([]);
  const [shapData, setShapData] = useState<any[]>([]);
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
        
        const zl = zlJson.data || [];
        const shap = shapJson.data || [];

        setZlData(zl);
        setShapData(shap);
        setLastUpdate(new Date().toLocaleTimeString());

        if (zl.length > 0) {
          const latest = zl[zl.length - 1].close;
          setLatestPrice(latest);
          
          if (zl.length > 1) {
            const prev = zl[zl.length - 2].close;
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
    const interval = setInterval(fetchData, 300000); // 5 minutes
    return () => clearInterval(interval);
  }, []);

  // Prepare Plotly data
  const dates = zlData.map(d => d.date.value);
  const closes = zlData.map(d => d.close);
  const highs = zlData.map(d => d.high);
  const lows = zlData.map(d => d.low);

  // SHAP data by date
  const shapByDate = new Map();
  shapData.forEach(s => {
    if (!shapByDate.has(s.date)) {
      shapByDate.set(s.date, {});
    }
    shapByDate.get(s.date)[s.feature_name] = s.shap_value_cents;
  });

  const rins = dates.map(d => shapByDate.get(d)?.RINs_momentum || 0);
  const tariff = dates.map(d => shapByDate.get(d)?.Tariff_risk || 0);
  const drought = dates.map(d => shapByDate.get(d)?.Drought_zscore || 0);
  const crush = dates.map(d => shapByDate.get(d)?.Crush_margin || 0);

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

      {/* Main Chart - Full Width with Plotly */}
      <div className="max-w-[98%] mx-auto px-6 py-6">
        <div className="bg-[#0d1220] rounded-lg border border-[#1a1f2e] p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-thin text-white tracking-wide">ZL Price Action + SHAP Force Drivers</h2>
            <div className="flex gap-4 text-xs font-light">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-[#84cc16] rounded-full"></div>
                <span className="text-gray-400">ZL Price</span>
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

          {!loading && zlData.length > 0 ? (
            <div className="h-[650px]">
              <Plot
                data={[
                  // ZL Price Line (Left Y-axis)
                  {
                    x: dates,
                    y: closes,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'ZL Price',
                    line: {
                      color: '#84cc16',
                      width: 2,
                      shape: 'spline'
                    },
                    yaxis: 'y',
                    hovertemplate: '$%{y:.2f}<extra></extra>'
                  },
                  // High/Low Range
                  {
                    x: dates,
                    y: highs,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'High',
                    line: {
                      color: '#84cc16',
                      width: 0.5,
                      dash: 'dot'
                    },
                    opacity: 0.3,
                    yaxis: 'y',
                    showlegend: false,
                    hoverinfo: 'skip'
                  },
                  {
                    x: dates,
                    y: lows,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Low',
                    fill: 'tonexty',
                    fillcolor: 'rgba(132, 204, 22, 0.05)',
                    line: {
                      color: '#84cc16',
                      width: 0.5,
                      dash: 'dot'
                    },
                    opacity: 0.3,
                    yaxis: 'y',
                    showlegend: false,
                    hoverinfo: 'skip'
                  },
                  // SHAP Drivers (Right Y-axis)
                  {
                    x: dates,
                    y: rins,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'RINs Momentum',
                    line: {
                      color: '#65a30d',
                      width: 1.5,
                      dash: 'dash'
                    },
                    yaxis: 'y2',
                    hovertemplate: '%{y:+.2f}¢<extra></extra>'
                  },
                  {
                    x: dates,
                    y: tariff,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Tariff Risk',
                    line: {
                      color: '#a3e635',
                      width: 1.5,
                      dash: 'dash'
                    },
                    yaxis: 'y2',
                    hovertemplate: '%{y:+.2f}¢<extra></extra>'
                  },
                  {
                    x: dates,
                    y: drought,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Drought Z-Score',
                    line: {
                      color: '#bef264',
                      width: 1.5,
                      dash: 'dash'
                    },
                    yaxis: 'y2',
                    hovertemplate: '%{y:+.2f}¢<extra></extra>'
                  },
                  {
                    x: dates,
                    y: crush,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Crush Margin',
                    line: {
                      color: '#d9f99d',
                      width: 1.5,
                      dash: 'dash'
                    },
                    yaxis: 'y2',
                    hovertemplate: '%{y:+.2f}¢<extra></extra>'
                  }
                ]}
                layout={{
                  paper_bgcolor: '#0d1220',
                  plot_bgcolor: '#0d1220',
                  font: {
                    family: 'system-ui, -apple-system, sans-serif',
                    size: 11,
                    color: '#6b7280',
                    weight: 300
                  },
                  margin: { t: 20, r: 80, b: 50, l: 60 },
                  hovermode: 'x unified',
                  showlegend: true,
                  legend: {
                    orientation: 'h',
                    yanchor: 'bottom',
                    y: 1.02,
                    xanchor: 'right',
                    x: 1,
                    font: { size: 10, color: '#9ca3af' }
                  },
                  xaxis: {
                    gridcolor: '#1a1f2e',
                    gridwidth: 0.5,
                    showline: true,
                    linecolor: '#1a1f2e',
                    linewidth: 0.5,
                    tickfont: { size: 10, color: '#6b7280' },
                    zeroline: false
                  },
                  yaxis: {
                    title: {
                      text: 'ZL Price (¢/lb)',
                      font: { size: 11, color: '#9ca3af', weight: 300 }
                    },
                    gridcolor: '#1a1f2e',
                    gridwidth: 0.5,
                    showline: true,
                    linecolor: '#1a1f2e',
                    linewidth: 0.5,
                    tickfont: { size: 10, color: '#6b7280' },
                    zeroline: false,
                    side: 'left'
                  },
                  yaxis2: {
                    title: {
                      text: 'SHAP Impact (¢/lb)',
                      font: { size: 11, color: '#9ca3af', weight: 300 }
                    },
                    gridcolor: 'transparent',
                    showline: true,
                    linecolor: '#1a1f2e',
                    linewidth: 0.5,
                    tickfont: { size: 10, color: '#6b7280' },
                    zeroline: true,
                    zerolinecolor: '#1a1f2e',
                    zerolinewidth: 1,
                    overlaying: 'y',
                    side: 'right',
                    range: [-15, 20]
                  },
                  dragmode: 'pan'
                }}
                config={{
                  responsive: true,
                  displayModeBar: true,
                  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
                  displaylogo: false,
                  toImageButtonOptions: {
                    format: 'png',
                    filename: 'zl_shap_chart',
                    height: 800,
                    width: 1600,
                    scale: 2
                  }
                }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true}
              />
            </div>
          ) : (
            <div className="h-[650px] flex items-center justify-center text-gray-500">
              <p className="text-sm font-light">Loading live ZL data from Databento...</p>
            </div>
          )}

          {/* Stats Row */}
          <div className="mt-6 grid grid-cols-5 gap-4">
            <div className="bg-[#0a0e1a] rounded border border-[#1a1f2e] p-4">
              <p className="text-gray-500 text-xs font-light mb-1">Data Points</p>
              <p className="text-2xl font-extralight text-white">{zlData.length}</p>
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
