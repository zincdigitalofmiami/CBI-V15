# Quant Admin / Training + AnoFox (`/quant-admin`) – Internal

## Purpose
Internal quant cockpit for:
- Pipeline health (ingestion + feature + training)
- Feature and training matrix completeness
- Model registry and experiment metrics
- Training run monitoring and QA checks

## Data Sources (MotherDuck)
- `ops.ingestion_status`, `ops.pipeline_metrics`
- `features.daily_ml_matrix_zl`
- `training.daily_ml_matrix_zl`
- `reference.feature_catalog`, `reference.model_registry`
- `forecasts.zl_predictions`, `training.model_runs`

## Key Components

### 1. Pipeline Health
- Ingestion lag (hours since last successful run)
- Error counts by source
- Last successful run timestamp
- Data freshness by table

### 2. Matrix Status
- Row counts by date
- Missing dates (gaps in coverage)
- Feature coverage (% of expected features populated)
- Training/validation/test split distribution

### 3. Model Registry
- Active models by horizon
- Metrics from `metrics_json`:
  - MAE, RMSE, R²
  - Sharpe ratio
  - Hit rate
- Champion vs challenger comparison

### 4. Model / Forecast Runs
- Table of recent model runs with filters
- Status (running, completed, failed)
- Links to logs and artifacts
- Narratives and recommendations from AutoGluon + SQL outputs

## Schema Reference

### `ops.ingestion_status`
```sql
CREATE TABLE ops.ingestion_status (
  source TEXT,
  last_run TIMESTAMP,
  status TEXT, -- 'success', 'failed', 'running'
  rows_ingested BIGINT,
  error_message TEXT
);
```

### `reference.model_registry`
```sql
CREATE TABLE reference.model_registry (
  model_id TEXT PRIMARY KEY,
  horizon TEXT, -- '1W', '1M', '3M', '6M'
  model_type TEXT, -- 'chronos', 'autogluon', 'ensemble'
  status TEXT, -- 'champion', 'challenger', 'retired'
  metrics_json JSON,
  created_at TIMESTAMP
);
```

### `forecasts.zl_predictions` (example)
```sql
SELECT as_of_date, horizon, p10, p50, p90, metadata
FROM forecasts.zl_predictions
ORDER BY as_of_date DESC;
```

## Metrics Display

### Model Performance
- **MAE** (Mean Absolute Error) - Lower is better
- **RMSE** (Root Mean Squared Error) - Lower is better
- **R²** (Coefficient of Determination) - Higher is better
- **Sharpe** - Risk-adjusted returns
- **Hit Rate** - % of directional calls correct

### Pipeline Health
- **Ingestion Lag** - Hours since last successful run
- **Error Rate** - % of failed ingestion attempts
- **Data Freshness** - Latest date in each table

## Notes
- This route should **not appear in the main nav**.
- Authentication/authorization required (dev/ops only).
- Displays training run artifacts and model performance metrics.
- No business-friendly simplification needed; this is the cockpit.

## Visual Design

### DashdarkX Theme
- **Background:** `rgb(0, 0, 0)` - pure black
- **All text:** `font-extralight` with `font-mono` for metrics
- **Status indicators:** Color-coded by health
- **Tables:** `border-zinc-800` with alternating row shading

### Pipeline Health Colors
| Status | Color | Indicator |
|--------|-------|----------|
| Healthy | Green | `bg-green-500` dot |
| Warning | Yellow | `bg-yellow-500` dot |
| Error | Red | `bg-red-500` dot |
| Running | Blue | `bg-blue-500` animated pulse |

### Model Registry Display
- Champion models highlighted with gold border
- Challenger models with dashed border
- Retired models grayed out
- Metrics displayed in `font-mono` for alignment

### Training Run Cards
- Status badge (running/completed/failed)
- Expandable log viewer
- Performance metrics in card body
- Timestamp in `text-zinc-500`
