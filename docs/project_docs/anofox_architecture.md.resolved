# AnoFox-First Architecture: Full Exploitation Strategy

## Core Philosophy

**OLD APPROACH**: Your schema + your models  
**NEW APPROACH**: AnoFox models + your buckets as regime enhancers

## AnoFox Full Suite (What We'll Exploit)

### 1. **Tabular Extension** (Data Quality & Anomalies)
- Auto anomaly detection
- Data quality validation
- Schema inference
- Missing value detection
- **USE CASE**: Validate incoming Databento data, flag regime shifts

### 2. **Forecast Extension** (31 Models)
- AutoETS (auto seasonality)
- ARIMA variants
- Prophet
- Theta
- **USE CASE**: Main price forecasting pipeline

### 3. **Statistics Extension** (Regression & Inference)
- Regression analysis
- Price elasticity
- Factor attribution
- **USE CASE**: SHAP-like attribution for Big 8 Drivers

### 4. **Shock Detection** (undocumented, need to verify)
- Volatility spikes
- Regime changes
- **USE CASE**: Trigger bucket reweighting

### 5. **Sentiment Analysis** (if available)
- News sentiment scoring
- **USE CASE**: Feed your 7 news buckets

## Your 7 News Buckets (Preserved as Regime Enhancers)

Based on CBI-V15, you have buckets for:
1. **Biofuel Policy** (RFS, 45Z, LCFS)
2. **China Demand** (Import volumes, trade tensions)
3. **Tariffs/Trade Policy** (Trump actions, trade wars)
4. **Weather** (Brazil/Argentina/US harvest)
5. **Energy Markets** (WTI crude correlation)
6. **Fed Policy** (Interest rates, DXY impact)
7. **Volatility Regime** (VIX, market stress)

**How They Work**: Each bucket gets its own **AnoFox model ensemble**, then you use **bucket weights** to combine forecasts based on current regime.

## Architecture: Specialized Models → Main Pipeline

```
[Databento Raw Data]
    ↓
[AnoFox Tabular: Validate & Clean]
    ↓
[Your 276 Features: Calculate]
    ↓
    ├─→ [AnoFox Model: Volatility] → bucket_volatility_forecast
    ├─→ [AnoFox Model: Weather Impact] → bucket_weather_forecast
    ├─→ [AnoFox Model: Sentiment] → bucket_sentiment_forecast
    ├─→ [AnoFox Model: Energy] → bucket_energy_forecast
    ├─→ [AnoFox Model: China] → bucket_china_forecast
    ├─→ [AnoFox Model: Policy] → bucket_policy_forecast
    └─→ [AnoFox Model: Tariffs] → bucket_tariff_forecast
    ↓
[Regime Detector] → Determines active bucket weights
    ↓
[Ensemble Combiner] → Weighted avg of 7 bucket forecasts
    ↓
[Final Forecast with P10/P50/P90 bands]
    ↓
[MotherDuck] → Vercel Dashboard
```

## Critical: Schema Consistency

**THE PROBLEM**: If historical MA calculation ≠ ongoing MA calculation, model breaks.

**THE SOLUTION**: Single source of truth for all calculations.

### DuckDB Feature Calculator (Lives on External Drive)

```sql
-- /Volumes/Satechi Hub/ZL-Intelligence/duckdb/calculators/features.sql

-- This EXACT SQL runs on both historical and ongoing data
CREATE OR REPLACE FUNCTION calculate_features(input_table STRING) AS (
    SELECT 
        date,
        close AS price_current,
        
        -- Moving Averages (EXACT same formula)
        AVG(close) OVER (ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS sma_5,
        AVG(close) OVER (ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS sma_10,
        AVG(close) OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS sma_21,
        
        -- Volatility (EXACT same window)
        STDDEV(LN(close / LAG(close, 1) OVER (ORDER BY date))) 
            OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) * SQRT(252) AS volatility_21d,
        
        -- ... ALL 276 features
        
    FROM query_table(input_table)
);

-- Use it EVERYWHERE:
-- Historical: SELECT * FROM calculate_features('historical_data');
-- Ongoing: SELECT * FROM calculate_features('latest_tick');
```

## Bucket-Specific Backtesting

For each of your 7 buckets, create **dedicated AnoFox models**:

```sql
-- Example: Volatility Bucket Model
CREATE OR REPLACE TABLE bucket_volatility_model AS
SELECT TS_FORECAST(
    (SELECT calculate_features('historical_data') WHERE regime_bucket = 'volatility'),
    'date',
    'price_current',
    'AutoETS',
    30  -- 30-day horizon
) AS forecast;

-- Repeat for all 7 buckets
```

## AnoFox Auto-Scoping (Your Question)

**YES** - AnoFox's `Tabular` extension can auto-detect:
- Data types
- Anomalies
- Schema
- Missing patterns

**BUT** - You still need to tell it which features to use for forecasting.

**RECOMMENDATION**: 
1. Let AnoFox validate data quality (Tabular)
2. You define feature set (your 276 features)
3. AnoFox trains models (Forecast)
4. You combine via buckets (your regime logic)

## Implementation Phases

### Phase 1: Data Foundation (Week 1)
- [ ] Migrate CBI-V15 → Local DuckDB
- [ ] Install AnoFox extensions
- [ ] Create single-source feature calculator
- [ ] Validate historical = ongoing calculations

### Phase 2: Specialized Models (Week 2)
- [ ] Train 7 bucket-specific AnoFox models
- [ ] Backtest each bucket independently
- [ ] Store bucket forecasts in MotherDuck

### Phase 3: Ensemble Pipeline (Week 3)
- [ ] Build regime detector (which bucket is active)
- [ ] Create ensemble combiner
- [ ] Generate P10/P50/P90 bands
- [ ] Push to MotherDuck

### Phase 4: Dashboard Integration (Week 4)
- [ ] Update Vercel to query MotherDuck
- [ ] Display bucket weights
- [ ] Show specialized model outputs
- [ ] Create Quant Admin page

## Next Steps: What to Build First?

1. **Feature Calculator SQL** (single source of truth)
2. **DuckDB Migration Script** (BigQuery → Local)
3. **AnoFox Installation Guide**
4. **Bucket Model Training Scripts**

Which would you like me to start with?
