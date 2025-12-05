---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.
All data must come from authenticated APIs, official sources, or validated historical records.
---

# CBI-V15 Complete System Flow (LEGACY V14/BQML VIEW)
**Last Updated**: November 12, 2025  
**Status**: LEGACY – kept for historical context only  
**Authoritative Flow**: See `docs/architecture/MASTER_PLAN.md` and `docs/architecture/TRAINING_PLAN.md` for the current V15 Mac‑only, Python‑first architecture (no Dataform).

---

## 📊 System Overview (Legacy Description)

## 🔄 COMPLETE DATA FLOW

### PHASE 1: DATA INGESTION

#### 1.1 Data Sources

**External APIs & Websites:**
- **EPA RIN Prices** - Renewable fuel credit prices (scraped from EPA website)
- **EPA RFS Mandates** - Biofuel policy data
- **USDA Harvest Data** - Crop progress reports
- **EIA Biofuel Production** - Energy Information Administration
- **NOAA Weather** - Weather data for US, Brazil, Argentina
- **CFTC Positioning** - Commodity Futures Trading Commission positioning data
- **FRED Economic Data** - Federal Reserve economic indicators
- **Baltic Dry Index** - Shipping cost indicators
- **Argentina Port Logistics** - Export capacity data
- **Social Media Intelligence** - Twitter/Truth Social monitoring (Trump policy signals)

**Historical Data (Integrated Nov 12, 2025):**
- **Economic Indicators** - 126 years (1900-2026), 7,523 rows

#### 1.2 Ingestion Scripts Location

**Primary Location**: `src/ingestion/` (78 Python scripts)

**Key Ingestion Scripts:**
```
src/ingestion/
├── ingest_epa_rin_prices.py          # EPA RIN prices (weekly)
├── ingest_epa_rfs_mandates.py        # Biofuel policy mandates
├── ingest_usda_harvest_real.py       # USDA harvest progress
├── ingest_eia_biofuel_real.py        # EIA biofuel production
├── ingest_weather_noaa.py            # NOAA weather data
├── ingest_cftc_positioning_REAL.py   # CFTC positioning data
├── ingest_volatility.py              # VIX and volatility data
├── ingest_baltic_dry_index.py        # Shipping cost indicators
├── ingest_argentina_port_logistics.py # Export capacity
└── trump_truth_social_monitor.py     # Social media intelligence
```

#### 1.3 Ingestion Process

**Step 1: Data Collection**
- Each ingestion script:
  1. Connects to external API/website
  2. Fetches latest data (with retry logic)
  3. Validates data quality (price bounds, date ranges)
  4. Transforms to standardized schema

**Step 2: Data Parsing**
- Raw data parsed into structured format
- Date normalization (all dates → DATE or DATETIME)
- Price normalization (currency, units standardized)
- Missing value handling (forward-fill, interpolation)

**Example Flow (EPA RIN Prices):**
```python
# From ingest_epa_rin_prices.py
1. Scrape EPA website → BeautifulSoup parsing
2. Extract RIN price tables → Pandas DataFrame
3. Validate prices (0.0 - 10.0 range)
4. Standardize date format
5. Upload to BigQuery
```

#### 1.4 Data Storage (BigQuery)

**Primary Warehouse**: `cbi-v15.forecasting_data_warehouse.*`

**Tables Created by Ingestion:**
```
forecasting_data_warehouse/
├── soybean_oil_prices          # Target commodity (6,057 rows, 2000-2025)
├── soybean_prices              # Related commodity
├── corn_prices                  # Related commodity
├── wheat_prices                 # Related commodity
├── crude_oil_prices            # Energy correlation
├── palm_oil_prices             # Substitution correlation
├── usd_index_prices             # Currency impact
├── vix_daily                   # Volatility indicator
├── biofuel_prices              # EPA RIN prices
├── weather_data                # NOAA weather
├── cftc_positioning            # Speculator positioning
└── [70+ other source tables]
```

**Historical Integration Tables:**
```
├── all_symbols_20yr            # 57,397 rows
├── biofuel_components_raw      # 42,367 rows
└── biofuel_components_canonical # 6,475 rows
```

---

### PHASE 2: DATA CONSOLIDATION & FEATURE ENGINEERING

#### 2.1 Data Consolidation

**Script**: `scripts/run_ultimate_consolidation.sh`  
**SQL**: `config/bigquery/bigquery-sql/ULTIMATE_DATA_CONSOLIDATION.sql`

**Process:**
1. **Backup existing production data**
2. **Combine all data sources** into unified tables
3. **Fill gaps** (e.g., Sep 11-Oct 27 with Vertex AI data)
4. **Forward-fill sparse features** (missing dates)
5. **Update with current signals** (Big 8 signals, latest date)

**Output Tables:**
- `cbi-v15.models_v4.production_training_data_1w`
- `cbi-v15.models_v4.production_training_data_1m`
- `cbi-v15.models_v4.production_training_data_3m`
- `cbi-v15.models_v4.production_training_data_6m`
- `cbi-v15.models_v4.production_training_data_12m`

**Each table contains:**
- **290+ features** (price, correlation, sentiment, weather, etc.)
- **Target variables** (target_1w, target_1m, target_3m, target_6m, target_12m)
- **Date column** (partitioned by month, clustered by date)

#### 2.2 Feature Engineering

**SQL-Based Features** (in BigQuery):
- **Price features**: Lags (1d, 7d, 30d), returns, moving averages, volatility
- **Correlation features**: Rolling correlations (7d, 30d, 90d, 180d, 365d) with crude, palm, VIX, USD, corn, wheat
- **Big 8 Signals**: VIX stress, harvest pace, China relations, tariff threat, geopolitical volatility, biofuel cascade, hidden correlation, biofuel ethanol
- **Crush margins**: Oil/bean/meal prices, margin calculations, moving averages
- **China import tracker**: Mentions, sentiment, policy impact, demand index
- **Brazil export**: Seasonality, weather, harvest pressure, export capacity
- **Trump-Xi volatility**: Mentions, sentiment, tension index, policy impact
- **Trade war impact**: Tariff rates, market share, export impact
- **Event-driven**: WASDE days, FOMC days, crop reports, holidays
- **Cross-asset lead/lag**: Palm/crude/VIX/DXY momentum and direction signals
- **Weather**: Brazil, Argentina, US temperature and precipitation
- **Sentiment**: Social media sentiment, volatility, volume

**Python-Based Features** (local processing):
- **Script**: `scripts/build_features.py`
- **Location**: `TrainingData/processed/`
- **Process**:
  1. Load exported Parquet files
  2. Apply additional transformations (rolling windows, momentum)
  3. Time-based features (day_of_week, month, quarter, year)
  4. Save to `TrainingData/processed/processed_training_data_{horizon}.parquet`

---

### PHASE 3: DATA EXPORT & PREPARATION

#### 3.1 Export to Local Training Data

**Script**: `scripts/export_training_data.py`

**Process:**
1. **Query BigQuery tables** (`production_training_data_*`)
2. **Export to Parquet format** (optimized for ML training)
3. **Save to external drive**: `TrainingData/exports/`

**Exported Files:**
```
TrainingData/exports/
├── production_training_data_1w.parquet    # 290+ features
├── production_training_data_1m.parquet    # 290+ features
├── production_training_data_3m.parquet   # 290+ features
├── production_training_data_6m.parquet    # 290+ features
├── production_training_data_12m.parquet  # 290+ features
├── trump_rich_2023_2025.parquet          # 42 features, 782 rows
├── trump_2.0_2023_2025.parquet           # Regime-specific
├── trade_war_2017_2019.parquet           # Regime-specific
├── inflation_2021_2022.parquet           # Regime-specific
├── crisis_2008_2020.parquet              # Regime-specific
└── historical_pre2000.parquet            # Regime-specific

TrainingData/raw/
└── historical_full.parquet                # 25 years (2000-2025)
```

#### 3.2 Data Quality Validation

**Script**: `scripts/data_quality_checks.py`

**Validations:**
- Missing value checks
- Data type validation
- Date range validation
- Price bounds validation
- Feature completeness checks
- Target variable validation

---

### PHASE 4: MODEL TRAINING

#### 4.1 BQML Training (Production Track 1)

**Location**: `config/bigquery/bigquery-sql/PRODUCTION_HORIZON_SPECIFIC/`

**Training SQL Files:**
- `TRAIN_BQML_1W_PRODUCTION.sql`
- `TRAIN_BQML_1M_PRODUCTION.sql`
- `TRAIN_BQML_3M_PRODUCTION.sql`
- `TRAIN_BQML_6M_PRODUCTION.sql`
- `TRAIN_BQML_12M_PRODUCTION.sql` (if exists)

**Training Process:**
```sql
-- Example: TRAIN_BQML_1M_PRODUCTION.sql
CREATE OR REPLACE MODEL `cbi-v15.models_v4.bqml_1m`
OPTIONS(
  model_type='BOOSTED_TREE_REGRESSOR',
  input_label_cols=['target_1m'],
  max_iterations=100,
  learn_rate=0.1,
  early_stop=False
) AS
SELECT 
  target_1m,
  * EXCEPT(target_1w, target_1m, target_3m, target_6m, date, ...)
FROM `cbi-v15.models_v4.production_training_data_1m`
WHERE target_1m IS NOT NULL;
```

**Trained Models:**
- `cbi-v15.models_v4.bqml_1w` (MAPE 0.7-1.3%, R² > 0.95)
- `cbi-v15.models_v4.bqml_1m` (MAPE 0.7-1.3%, R² > 0.95)
- `cbi-v15.models_v4.bqml_3m` (MAPE 0.7-1.3%, R² > 0.95)
- `cbi-v15.models_v4.bqml_6m` (MAPE 0.7-1.3%, R² > 0.95)
- `cbi-v15.models_v4.bqml_12m` (MAPE 0.7-1.3%, R² > 0.95)

**Training Command:**
```bash
bq query --nouse_legacy_sql < config/bigquery/bigquery-sql/PRODUCTION_HORIZON_SPECIFIC/TRAIN_BQML_1M_PRODUCTION.sql
```

#### 4.2 Local Training (Production Track 2)

**Location**: `src/training/baselines/`

**Training Scripts:**
- `train_statistical.py` - ARIMA, Prophet, Exponential Smoothing
- `train_tree.py` - LightGBM, XGBoost
- `train_simple_neural.py` - LSTM, GRU (TensorFlow Metal GPU)

**Training Process:**
1. **Load processed data** from `TrainingData/processed/`
2. **Split data** (train/validation/test, time-based)
3. **Train model** (with Metal GPU acceleration for neural)
4. **Evaluate** (MAPE, R², RMSE)
5. **Save model** to `Models/local/baselines/`
6. **Log metrics** to MLflow (`Models/mlflow/`)

**Model Storage:**
```
Models/
├── local/
│   └── baselines/
│       ├── arima_1m.pkl
│       ├── lightgbm_1m.pkl
│       ├── lstm_1m.h5
│       └── [60-70 models total]
├── vertex-ai/
│   └── [SavedModels for deployment]
└── mlflow/
    └── [Experiment tracking]
```

#### 4.3 Vertex AI Deployment Pipeline

**Location**: `vertex-ai/deployment/`

**Deployment Scripts:**
- `train_local_deploy_vertex.py` - Complete workflow orchestrator
- `export_savedmodel.py` - Export TensorFlow model to SavedModel format
- `upload_to_vertex.py` - Upload SavedModel to Vertex AI Model Registry
- `create_endpoint.py` - Deploy model to Vertex AI endpoint

**Deployment Process:**
1. **Train locally** on M4 Mac (TensorFlow Metal GPU)
2. **Export to SavedModel** format
3. **Upload to Vertex AI** Model Registry
4. **Deploy endpoint** for predictions
5. **Test endpoint** with sample predictions

**Deployment Command:**
```bash
python vertex-ai/deployment/train_local_deploy_vertex.py --horizon=1m
```

---

### PHASE 5: PREDICTION GENERATION

#### 5.1 Daily Forecast Generation

**Script**: `src/prediction/generate_forecasts.py`

**Process:**
1. **Load latest features** from BigQuery (`production_training_data_*`)
2. **Prepare prediction input** (last 30 days for sequence models)
3. **Call BQML models** or **local models** (depending on track)
4. **Generate predictions** for all 5 horizons
5. **Calculate confidence intervals** (if available)
6. **Save to BigQuery**: `cbi-v15.predictions.daily_forecasts`

**Prediction Table Schema:**
```sql
predictions.daily_forecasts
├── prediction_date (DATE)
├── horizon (STRING) - '1W', '1M', '3M', '6M', '12M'
├── predicted_price (FLOAT64)
├── confidence_lower (FLOAT64)
├── confidence_upper (FLOAT64)
├── mape (FLOAT64)
├── model_id (STRING)
├── model_name (STRING)
├── target_date (DATE)
└── created_at (TIMESTAMP)
```

**Prediction Command:**
```bash
python src/prediction/generate_forecasts.py --horizon=all
```

#### 5.2 SHAP Explanations

**Script**: `src/prediction/shap_explanations.py`

**Process:**
1. **Load trained model** and latest data
2. **Calculate SHAP values** (feature importance per prediction)
3. **Save explanations** to BigQuery or local files
4. **Used by dashboard** to show "why prices are moving"

**Command:**
```bash
python src/prediction/shap_explanations.py --horizon=1m
```

---

### PHASE 6: WEB APP DISPLAY

#### 6.1 Dashboard Architecture

**Location**: `dashboard-nextjs/`  
**Framework**: Next.js 14+ (TypeScript, React)  
**Deployment**: Vercel

#### 6.2 API Endpoints

**Location**: `dashboard-nextjs/src/app/api/v4/`

**Forecast Endpoints:**
- `/api/v4/forecast/1w` - 1-week forecast
- `/api/v4/forecast/1m` - 1-month forecast
- `/api/v4/forecast/3m` - 3-month forecast
- `/api/v4/forecast/6m` - 6-month forecast

**Data Flow (Example: `/api/v4/forecast/1m`):**
```typescript
// 1. Query BigQuery for latest prediction
const forecastQuery = `
  SELECT predicted_price, confidence_lower, confidence_upper, mape, model_id
  FROM \`cbi-v15.predictions.daily_forecasts\`
  WHERE horizon = '1M'
  ORDER BY created_at DESC
  LIMIT 1
`

// 2. Query current price
const priceQuery = `
  SELECT close as current_price
  FROM \`cbi-v15.forecasting_data_warehouse.soybean_oil_prices\`
  ORDER BY time DESC
  LIMIT 1
`

// 3. Calculate change and return JSON
return {
  horizon: '1m',
  current_price: priceResult[0].current_price,
  prediction: forecast.predicted_price,
  predicted_change: prediction - currentPrice,
  predicted_change_pct: (change / currentPrice) * 100,
  confidence_metrics: { mape, r2 },
  timestamp: new Date().toISOString()
}
```

**Other Endpoints:**
- `/api/v4/big-eight-signals` - Big 8 signal values
- `/api/v4/feature-importance/[horizon]` - SHAP feature importance
- `/api/v4/forward-curve` - Forward curve visualization
- `/api/v4/procurement-timing` - Procurement recommendations
- `/api/v4/risk-radar` - Risk indicators
- `/api/v4/price-drivers` - Price driver analysis

#### 6.3 Frontend Components

**Location**: `dashboard-nextjs/src/components/dashboard/`

**Key Components:**
- `ForecastCards.tsx` - Main forecast display (1w, 1m, 3m)
- `CurrentPrice.tsx` - Current soybean oil price
- `ProcurementSignal.tsx` - BUY/WAIT/MONITOR recommendations
- `FeatureImportanceCard.tsx` - SHAP explanations
- `ForwardCurve.tsx` - Forward curve visualization
- `BigEightSignals.tsx` - Big 8 signal dashboard
- `RiskRadar.tsx` - Risk indicators
- `PriceDrivers.tsx` - Price driver analysis

**Data Fetching:**
- Uses React Query (`useQuery`) for data fetching
- Auto-refresh every 2 minutes (`refetchInterval: 120000`)
- Error handling and loading states

**Example Component Flow (`ForecastCards.tsx`):**
```typescript
// 1. Fetch forecasts for all horizons
const horizons = ['1w', '1m', '3m']
const forecastPromises = horizons.map(h => 
  fetch(`/api/v4/forecast/${h}`).then(r => r.json())
)

// 2. Process results
const forecasts = results
  .filter(r => r && !r.error && r.current_price && r.prediction)
  .map(item => ({
    horizon: item.horizon,
    current_price: item.current_price,
    predicted_price: item.prediction,
    change_pct: item.predicted_change_pct || 0,
    confidence: item.confidence_metrics?.r2 ? item.confidence_metrics.r2 * 100 : 85,
    recommendation: item.predicted_change_pct > 2 ? 'Buy now' : 
                   item.predicted_change_pct < -2 ? 'Wait' : 'Monitor'
  }))

// 3. Render cards
return <ForecastCards forecasts={forecasts} />
```

#### 6.4 Display Flow

**User View:**
1. **Dashboard loads** → Fetches current price and forecasts
2. **Forecast cards display** → Shows 1w, 1m, 3m predictions with:
   - Current price
   - Predicted price
   - % change
   - Confidence (R²)
   - Recommendation (BUY/WAIT/MONITOR)
3. **Auto-refresh** → Updates every 2 minutes
4. **Feature importance** → Shows SHAP explanations on demand
5. **Risk radar** → Displays risk indicators
6. **Forward curve** → Shows price curve across horizons

---

## 🔄 AUTOMATION & SCHEDULING

### Cron Jobs

**Script**: `scripts/crontab_setup.sh`

**Scheduled Jobs:**
- **3:30 AM** - Feature engineering (`scripts/build_features.py`)
- **4:00 AM** - Model training (if needed)
- **5:00 AM** - Prediction generation (`src/prediction/generate_forecasts.py`)
- **Daily** - Data consolidation (`scripts/run_ultimate_consolidation.sh`)

### Manual Triggers

**Status Check:**
```bash
./scripts/status_check.sh
```

**Data Consolidation:**
```bash
./scripts/run_ultimate_consolidation.sh
```

**Export Training Data:**
```bash
python scripts/export_training_data.py
```

**Train BQML:**
```bash
bq query --nouse_legacy_sql < config/bigquery/bigquery-sql/PRODUCTION_HORIZON_SPECIFIC/TRAIN_BQML_1M_PRODUCTION.sql
```

---

## 📊 DATA FLOW SUMMARY

```
EXTERNAL SOURCES
    ↓
[Ingestion Scripts] → BigQuery (forecasting_data_warehouse.*)
    ↓
[Consolidation SQL] → BigQuery (models_v4.production_training_data_*)
    ↓
[Export Script] → Parquet Files (TrainingData/exports/)
    ↓
[Feature Engineering] → Processed Parquet (TrainingData/processed/)
    ↓
[TRAINING]
    ├─→ BQML Models (models_v4.bqml_*) [Production Track 1]
    └─→ Local Models (Models/local/) → Vertex AI [Production Track 2]
    ↓
[Prediction Generation] → BigQuery (predictions.daily_forecasts)
    ↓
[Dashboard API] → Next.js API Routes (/api/v4/forecast/*)
    ↓
[Frontend Components] → User Display (ForecastCards, CurrentPrice, etc.)
```

---

## 🔑 KEY TABLES & LOCATIONS

### BigQuery Tables

**Source Data:**
- `forecasting_data_warehouse.soybean_oil_prices` - Target commodity prices
- `forecasting_data_warehouse.*` - 70+ source tables

**Training Data:**
- `models_v4.production_training_data_1w` - 1-week horizon (290+ features)
- `models_v4.production_training_data_1m` - 1-month horizon (290+ features)
- `models_v4.production_training_data_3m` - 3-month horizon (290+ features)
- `models_v4.production_training_data_6m` - 6-month horizon (290+ features)
- `models_v4.production_training_data_12m` - 12-month horizon (290+ features)

**Trained Models:**
- `models_v4.bqml_1w` - BQML 1-week model
- `models_v4.bqml_1m` - BQML 1-month model
- `models_v4.bqml_3m` - BQML 3-month model
- `models_v4.bqml_6m` - BQML 6-month model
- `models_v4.bqml_12m` - BQML 12-month model

**Predictions:**
- `predictions.daily_forecasts` - Daily predictions for all horizons

### Local Files

**Training Data:**
- `TrainingData/exports/` - Exported Parquet files from BigQuery
- `TrainingData/processed/` - Feature-engineered Parquet files
- `TrainingData/raw/` - Raw historical data

**Models:**
- `Models/local/baselines/` - Local trained models (PKL, H5)
- `Models/vertex-ai/` - SavedModels for Vertex AI deployment
- `Models/mlflow/` - MLflow experiment tracking

**Scripts:**
- `src/ingestion/` - 78 ingestion scripts
- `src/training/baselines/` - Training scripts
- `src/prediction/` - Prediction generation scripts
- `scripts/` - Utility scripts (export, build_features, etc.)

**Dashboard:**
- `dashboard-nextjs/src/app/api/v4/` - API endpoints
- `dashboard-nextjs/src/components/dashboard/` - React components

---

## 🎯 COMPLETE END-TO-END EXAMPLE

### Scenario: User views 1-month forecast on dashboard

1. **Data Ingestion** (already completed)
   - EPA RIN prices ingested → `forecasting_data_warehouse.biofuel_prices`
   - Weather data ingested → `forecasting_data_warehouse.weather_data`
   - Price data ingested → `forecasting_data_warehouse.soybean_oil_prices`

2. **Consolidation** (runs daily at 3:30 AM)
   - `scripts/run_ultimate_consolidation.sh` runs
   - Updates `models_v4.production_training_data_1m` with latest features

3. **Training** (runs monthly or on-demand)
   - `TRAIN_BQML_1M_PRODUCTION.sql` trains model
   - Model saved as `models_v4.bqml_1m`

4. **Prediction** (runs daily at 5:00 AM)
   - `src/prediction/generate_forecasts.py` runs
   - Queries latest features from `production_training_data_1m`
   - Calls BQML model `bqml_1m` for prediction
   - Saves to `predictions.daily_forecasts`

5. **API Request** (user loads dashboard)
   - Browser requests `/api/v4/forecast/1m`
   - Next.js API route queries `predictions.daily_forecasts`
   - Returns JSON with prediction, current price, change %

6. **Display** (React component)
   - `ForecastCards.tsx` receives data
   - Renders forecast card with:
     - Current price: $45.20
     - Predicted price: $46.50
     - Change: +2.9%
     - Recommendation: "Buy now"
     - Confidence: 95%

---

## 📝 NOTES

- **Data freshness**: Production tables updated daily via consolidation script
- **Model retraining**: BQML models retrained monthly or on-demand
- **Predictions**: Generated daily at 5 AM, cached in BigQuery
- **Dashboard**: Auto-refreshes every 2 minutes
- **Historical data**: 25+ years available (2000-2025) after Nov 12 integration
- **Two-track approach**: BQML (active) + Vertex AI (in progress)

---

**For questions or updates, see:**
- `active-plans/MASTER_EXECUTION_PLAN.md` - Current strategy
- `QUICK_REFERENCE.txt` - Quick command reference
- `docs/handoffs/` - Detailed handoff documentation
