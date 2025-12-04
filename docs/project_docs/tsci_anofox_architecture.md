# TimeSeriesScientist ↔ Anofox Integration Architecture

## Executive Summary

**TimeSeriesScientist (TSci)** is the agentic orchestrator that **wraps and directs Anofox** for the ZL Futures Forecasting System. TSci provides the intelligence and decision-making, while Anofox provides the high-performance SQL-native execution engine within DuckDB.

## Core Relationship

> **CRITICAL**: Anofox creates features FIRST, then TSci analyzes/models them.

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Flow Pipeline                      │
└─────────────────────────────────────────────────────────────┘

1. RAW DATA (Databento) → DuckDB
   ↓
2. ANOFOX FEATURE ENGINEERING (creates 276 features)
   - anofox_sma(close, 5)
   - anofox_volatility(close, 21)
   - anofox_rsi(close, 14)
   - ... all features precalculated in SQL
   ↓
3. ENRICHED DATASET → Exported to Python DataFrame
   ↓
4. TIMESERIESSCIENTIST (analyzes enriched data)
   ┌────────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐
   │  Curator   │→ │ Planner  │→ │ Forecaster│→ │ Reporter │
   │   Agent    │  │  Agent   │  │   Agent   │  │  Agent   │
   └─────┬──────┘  └────┬─────┘  └─────┬─────┘  └────┬─────┘
         │              │               │              │
         ↓              ↓               ↓              ↓
   "Clean data"   "Select model" "Generate     "Explain
   "(outliers)"   "What regime?" forecast"     results"
```

**Key Insight**: TSci does NOT create features. It expects you to provide a fully-featured dataset. Anofox creates those features efficiently in SQL before TSci ever sees the data.

## Integration Flow

### 1. Data Ingestion (TSci Curator → Anofox Tabular)

**TSci Decision**:
```python
# TSci analyzes incoming Databento data
curator_decision = {
    "data_quality": "missing values detected in 3.2% of records",
    "outlier_strategy": "clip outliers beyond 3 sigma",
    "recommendation": "use_anofox_tabular_extension"
}
```

**Anofox Execution**:
```sql
-- DuckDB with Anofox Tabular extension
SELECT anofox_gap_fill(
    date, close, 
    method := 'linear', 
    max_gap := '5 days'
) FROM zl_raw_prices;

SELECT anofox_outlier_detect(
    close, 
    method := 'zscore', 
    threshold := 3.0
) FROM zl_cleaned_prices;
```

### 2. Feature Engineering (TSci Planner → Anofox Statistics)

**TSci Decision**:
```python
# TSci determines which features are most predictive
planner_decision = {
    "regime": "high_volatility",
    "key_features": ["volatility_21d", "wti_correlation", "vix_level"],
    "model_recommendation": "ensemble_with_robust_baseline"
}
```

**Anofox Execution**:
```sql
-- Calculate features using Anofox Statistics
CREATE TABLE zl_features AS
SELECT 
    date,
    close,
    anofox_volatility(close, window := 21) AS volatility_21d,
    anofox_trend_strength(close, window := 60) AS trend_60d,
    anofox_correlation(close, wti_close, window := 90) AS wti_corr
FROM zl_cleaned_prices;
```

### 3. Model Selection & Forecasting (TSci Forecaster → Anofox Forecast)

**TSci Decision**:
```python
# TSci decides on ensemble weights based on regime
forecaster_decision = {
    "primary_models": ["AutoETS", "ARIMA", "Prophet"],
    "ensemble_strategy": "weighted_by_recent_mae",
    "statistical_baseline_weight": 0.6,  # High because of volatility
    "deep_learning_weight": 0.4
}
```

**Anofox Execution**:
```sql
-- Anofox generates statistical baselines rapidly
CREATE TABLE anofox_baseline_forecasts AS
SELECT 
    TS_FORECAST(
        (SELECT date, close FROM zl_features),
        'date', 'close',
        method := 'AutoETS',
        horizon := 30
    ) AS ets_forecast,
    TS_FORECAST(
        (SELECT date, close FROM zl_features),
        'date', 'close',
        method := 'ARIMA',
        horizon := 30
    ) AS arima_forecast
FROM zl_features;
```

### 4. Reporting (TSci Reporter → Anofox Metrics)

**TSci Decision**:
```python
# TSci writes the narrative
report = {
    "summary": "Switched to robust ensemble due to regime shift",
    "rationale": "Anofox detected 23% increase in volatility vs. historical avg",
    "confidence": "medium",
    "next_actions": "monitor_vix_correlation"
}
```

**Anofox Execution**:
```sql
-- Anofox provides the raw metrics for the report
SELECT 
    anofox_regime_detect(close, method := 'structural_break') AS regime_status,
    anofox_forecast_quality(actual, predicted) AS mae_by_horizon
FROM zl_validation_set;
```

## Setup & Integration Path

### Phase 1: TSci Framework Setup ✅ COMPLETE
- [x] Clone TimeSeriesScientist repo
- [x] Install Python dependencies (fixed `theta`, `langchain_core` imports)
- [x] Configure OpenAI API key
- [x] Verify end-to-end execution with test data
- [x] Create `/quant-admin` page to display TSci reports

### Phase 2: Anofox Installation → NEXT
- [ ] Install Anofox extensions in local DuckDB
  ```sql
  INSTALL anofox_tabular FROM community;
  INSTALL anofox_forecast FROM community;
  INSTALL anofox_statistics FROM community;
  LOAD anofox_tabular;
  LOAD anofox_forecast;
  LOAD anofox_statistics;
  ```
- [ ] Verify extensions with test queries
- [ ] Document available Anofox functions

### Phase 3: Custom TSci → Anofox Bridge
- [ ] Create Python wrapper class `AnofoxBridge`
  ```python
  class AnofoxBridge:
      def __init__(self, duckdb_path):
          self.conn = duckdb.connect(duckdb_path)
          self.conn.execute("LOAD anofox_tabular;")
          
      def clean_data(self, table_name, strategy):
          """TSci Curator calls this"""
          if strategy == "gap_fill":
              return self.conn.execute(
                  f"SELECT anofox_gap_fill(...) FROM {table_name}"
              ).fetchdf()
      
      def calculate_features(self, table_name):
          """TSci Planner calls this"""
          return self.conn.execute(
              f"SELECT anofox_volatility(...) FROM {table_name}"
              ).fetchdf()
      
      def generate_baseline(self, table_name, method):
          """TSci Forecaster calls this"""
          return self.conn.execute(
              f"SELECT TS_FORECAST(..., method := '{method}') FROM {table_name}"
          ).fetchdf()
  ```

- [ ] Modify TSci agent files to use `AnofoxBridge`:
  - `time_series_agent/agents/curator_agent.py` → add `self.anofox = AnofoxBridge()`
  - `time_series_agent/agents/planner_agent.py` → call `self.anofox.calculate_features()`
  - `time_series_agent/agents/forecast_agent.py` → call `self.anofox.generate_baseline()`

### Phase 4: ZL Data Integration
- [ ] Set up DuckDB schema for ZL futures (`ZL_FACT_PRICES_FEATURES` table)
- [ ] Create Databento ingestion script
- [ ] Load historical ZL data into DuckDB
- [ ] Run TSci + Anofox on real ZL data
- [ ] Validate forecasts vs. Databento actuals

### Phase 5: Production Pipeline
- [ ] Schedule automated TSci runs (cron job)
- [ ] Push results to MotherDuck
- [ ] Update `/quant-admin` to show live ZL forecasts
- [ ] Set up alerts for regime changes

## Key Design Principles

1. **TSci is the orchestrator**: It makes all strategic decisions (what to clean, which model, how to ensemble)
2. **Anofox is the executor**: It performs all heavy computations efficiently in SQL
3. **Single source of truth**: All feature calculations use Anofox SQL functions (no Python/Anofox drift)
4. **Transparent reasoning**: TSci reports explain which Anofox functions were used and why
5. **Regime-aware**: TSci adjusts Anofox model weights based on detected market regime

## Current Status

| Component | Status | Location |
|-----------|--------|----------|
| TimeSeriesScientist | ✅ Installed | `/TimeSeriesScientist/` |
| TSci Test Run | ✅ Verified | Ensemble MAPE: 8.09% |
| Quant Admin UI | ✅ Deployed | `web-gv13weo16...vercel.app/quant-admin` |
| Anofox Extensions | ⏳ Pending | Need to `INSTALL` in DuckDB |
| TSci→Anofox Bridge | ⏳ Pending | Need to create `AnofoxBridge` class |
| ZL Data Pipeline | ⏳ Pending | Need Databento→DuckDB script |

## Next Immediate Actions

1. **Install Anofox** in local DuckDB instance
2. **Document Anofox API** - list all available functions and their signatures
3. **Create AnofoxBridge** - Python class to connect TSci agents to Anofox SQL
4. **Test Integration** - Run TSci with Anofox on sample ZL data

---

**Critical Insight**: TSci doesn't replace Anofox - it **orchestrates** it. Anofox provides the computational muscle (fast SQL forecasting), while TSci provides the strategic brain (which forecast to trust, when to switch models, how to explain results).
