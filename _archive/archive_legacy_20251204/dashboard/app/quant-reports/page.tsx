"use client";

import { useEffect, useState } from 'react';

export default function QuantReportsPage() {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/quant-reports')
      .then(res => res.json())
      .then(json => {
        if (json.success) setData(json.data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="min-h-screen bg-[#020617] flex items-center justify-center text-slate-400 font-light">Loading AnoFox Metrics...</div>;

  return (
    <div className="min-h-screen bg-[#020617] text-slate-100">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="flex justify-between items-end mb-8">
            <div>
                <h1 className="text-3xl font-thin tracking-wide mb-2">Quant Reports</h1>
                <p className="text-sm text-slate-400 font-light">AnoFox Performance & Data Quality (The Iron Gate)</p>
            </div>
            <div className="text-right">
                <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">System Status</div>
                <div className="text-green-400 font-light flex items-center gap-2 justify-end">
                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                    Online
                </div>
            </div>
        </div>

        <div className="grid gap-6 md:grid-cols-3 mb-12">
            {data.map((row, i) => (
                <div key={i} className="bg-[#020617] border border-slate-800 rounded-xl p-6 hover:border-slate-700 transition-colors">
                    <div className="flex justify-between items-start mb-4">
                        <h3 className="text-lg font-thin text-white capitalize">{row.bucket} Bucket</h3>
                        <span className="text-xs font-mono text-slate-500 bg-slate-900 px-2 py-1 rounded">v1.4</span>
                    </div>
                    
                    <div className="space-y-4">
                        <div>
                            <div className="flex justify-between text-xs text-slate-400 mb-1">
                                <span>MAPE</span>
                                <span>{row.mape ? row.mape.toFixed(2) : '0.00'}%</span>
                            </div>
                            <div className="h-1 bg-slate-900 rounded-full overflow-hidden">
                                <div className="h-full bg-blue-500" style={{ width: `${Math.min(row.mape || 0, 100)}%` }}></div>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4 pt-2 border-t border-slate-900">
                            <div>
                                <div className="text-[10px] text-slate-500 uppercase">RMSE</div>
                                <div className="text-sm font-light text-slate-200">{row.rmse ? row.rmse.toFixed(3) : '0.000'}</div>
                            </div>
                            <div>
                                <div className="text-[10px] text-slate-500 uppercase">Coverage (90%)</div>
                                <div className={`text-sm font-light ${row.coverage_90 > 0.85 ? 'text-green-400' : 'text-yellow-400'}`}>
                                    {row.coverage_90 ? (row.coverage_90 * 100).toFixed(1) : '0.0'}%
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            ))}
        </div>

        <div className="bg-[#020617] border border-slate-800 rounded-xl p-8">
            <h3 className="text-xl font-thin mb-6">Data Quality Gates</h3>
            <div className="overflow-x-auto">
                <table className="w-full text-left text-sm text-slate-400">
                    <thead className="text-xs uppercase text-slate-500 border-b border-slate-800">
                        <tr>
                            <th className="pb-3 font-medium">Table</th>
                            <th className="pb-3 font-medium">Total Rows</th>
                            <th className="pb-3 font-medium">Null Ratio</th>
                            <th className="pb-3 font-medium">Status</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-900">
                        {data.length > 0 && (
                            <tr>
                                <td className="py-3 text-slate-300 font-light">staging.market_daily</td>
                                <td className="py-3 font-mono">{parseInt(data[0].total_rows).toLocaleString()}</td>
                                <td className="py-3 font-mono text-yellow-500">{(data[0].avg_null_ratio * 100).toFixed(4)}%</td>
                                <td className="py-3"><span className="text-green-400 border border-green-900 bg-green-900/20 px-2 py-0.5 rounded text-xs">PASSED</span></td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
      </div>
    </div>
  );
}

