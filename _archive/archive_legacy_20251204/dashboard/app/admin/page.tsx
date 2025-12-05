"use client";

import { useState, useEffect } from 'react';

interface IngestionStatus {
  databento: { status: string; lastUpdate: string; rowCount: number; tables: string[] };
  fred: { status: string; lastUpdate: string; rowCount: number; tables: string[] };
  duckdb: { status: string; lastUpdate: string; totalRows: number; tablesLoaded: number };
}

interface SystemHealth {
  duckdb: { status: string; databasePath: string; sizeMB: number; lastBackup: string };
  ingestion: { status: string; lastRun: string; nextRun: string; errors: number };
  training: { status: string; lastRun: string; nextRun: string; models: Record<string, { status: string; mape: number }> };
  dashboard: { status: string; uptime: string; lastDeploy: string };
}

export default function AdminPage() {
  const [ingestionStatus, setIngestionStatus] = useState<IngestionStatus | null>(null);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [statusRes, healthRes] = await Promise.all([
        fetch('/api/admin/ingestion-status'),
        fetch('/api/admin/system-health')
      ]);
      
      const status = await statusRes.json();
      const health = await healthRes.json();
      
      setIngestionStatus(status);
      setSystemHealth(health);
    } catch (error) {
      console.error('Error fetching admin data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async (source: string) => {
    setRefreshing(source);
    try {
      const res = await fetch('/api/admin/refresh-data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source })
      });
      const data = await res.json();
      alert(`Refresh triggered: ${data.message}`);
      // Refresh data after a delay
      setTimeout(() => fetchData(), 2000);
    } catch (error) {
      alert('Error triggering refresh');
    } finally {
      setRefreshing(null);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#020617] text-slate-100 flex items-center justify-center">
        <div className="text-slate-400">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#020617] text-slate-100">
      <div className="max-w-6xl mx-auto px-6 py-8">
        <h2 className="text-2xl font-thin tracking-wide mb-2">
          Admin & System Status
        </h2>
        <p className="text-sm text-slate-400 mb-6">
          High‑level view of ingestion, training, and dashboard health.
        </p>

        {/* System Health Overview */}
        {systemHealth && (
          <div className="grid gap-4 mb-6 md:grid-cols-4">
            <div className="bg-[#020617] border border-slate-800 rounded-xl p-4">
              <div className="text-xs text-slate-400 mb-1">DuckDB</div>
              <div className="text-lg font-medium text-slate-100">{systemHealth.duckdb.status}</div>
              <div className="text-xs text-slate-500 mt-1">{systemHealth.duckdb.sizeMB.toFixed(1)} MB</div>
            </div>
            <div className="bg-[#020617] border border-slate-800 rounded-xl p-4">
              <div className="text-xs text-slate-400 mb-1">Ingestion</div>
              <div className="text-lg font-medium text-slate-100">{systemHealth.ingestion.status}</div>
              <div className="text-xs text-slate-500 mt-1">{systemHealth.ingestion.errors} errors</div>
            </div>
            <div className="bg-[#020617] border border-slate-800 rounded-xl p-4">
              <div className="text-xs text-slate-400 mb-1">Training</div>
              <div className="text-lg font-medium text-slate-100">{systemHealth.training.status}</div>
              <div className="text-xs text-slate-500 mt-1">{Object.keys(systemHealth.training.models).length} models</div>
            </div>
            <div className="bg-[#020617] border border-slate-800 rounded-xl p-4">
              <div className="text-xs text-slate-400 mb-1">Dashboard</div>
              <div className="text-lg font-medium text-slate-100">{systemHealth.dashboard.status}</div>
              <div className="text-xs text-slate-500 mt-1">{systemHealth.dashboard.uptime} uptime</div>
            </div>
          </div>
        )}

        {/* Ingestion Status */}
        {ingestionStatus && (
          <section className="bg-[#020617] border border-slate-800 rounded-xl p-5 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-slate-200">Data Ingestion</h3>
              <button
                onClick={() => handleRefresh('all')}
                disabled={refreshing === 'all'}
                className="px-3 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-50"
              >
                {refreshing === 'all' ? 'Refreshing...' : 'Refresh All'}
              </button>
            </div>
            <div className="grid gap-4 md:grid-cols-3 text-[13px] text-slate-300">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">Databento</span>
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    ingestionStatus.databento.status === 'active' ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
                  }`}>
                    {ingestionStatus.databento.status}
                  </span>
                </div>
                <div className="text-xs text-slate-500 space-y-1">
                  <div>Rows: {ingestionStatus.databento.rowCount.toLocaleString()}</div>
                  <div>Last: {new Date(ingestionStatus.databento.lastUpdate).toLocaleString()}</div>
                  <button
                    onClick={() => handleRefresh('databento')}
                    disabled={refreshing === 'databento'}
                    className="text-blue-400 hover:text-blue-300 text-xs mt-1"
                  >
                    {refreshing === 'databento' ? 'Refreshing...' : 'Refresh'}
                  </button>
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">FRED</span>
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    ingestionStatus.fred.status === 'active' ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
                  }`}>
                    {ingestionStatus.fred.status}
                  </span>
                </div>
                <div className="text-xs text-slate-500 space-y-1">
                  <div>Rows: {ingestionStatus.fred.rowCount.toLocaleString()}</div>
                  <div>Last: {new Date(ingestionStatus.fred.lastUpdate).toLocaleString()}</div>
                  <button
                    onClick={() => handleRefresh('fred')}
                    disabled={refreshing === 'fred'}
                    className="text-blue-400 hover:text-blue-300 text-xs mt-1"
                  >
                    {refreshing === 'fred' ? 'Refreshing...' : 'Refresh'}
                  </button>
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">DuckDB</span>
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    ingestionStatus.duckdb.status === 'active' ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
                  }`}>
                    {ingestionStatus.duckdb.status}
                  </span>
                </div>
                <div className="text-xs text-slate-500 space-y-1">
                  <div>Total Rows: {ingestionStatus.duckdb.totalRows.toLocaleString()}</div>
                  <div>Tables: {ingestionStatus.duckdb.tablesLoaded}</div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Training Status */}
        {systemHealth && (
          <section className="bg-[#020617] border border-slate-800 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-200 mb-4">Model Training</h3>
            <div className="grid gap-4 md:grid-cols-4 text-[13px]">
              {Object.entries(systemHealth.training.models).map(([horizon, model]) => (
                <div key={horizon} className="border border-slate-800 rounded-lg p-3">
                  <div className="font-medium text-slate-200 mb-2">{horizon.toUpperCase()}</div>
                  <div className="text-xs text-slate-400 space-y-1">
                    <div>Status: <span className="text-slate-300">{model.status}</span></div>
                    <div>MAPE: <span className="text-slate-300">{model.mape.toFixed(2)}%</span></div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

