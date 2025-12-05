# Model Registry

**Status:** Production  
**Last Updated:** December 3, 2025

## Active Production Models

All models stored in MotherDuck `model_registry` table.

## Ensemble Configuration

### 1-Week Horizon
1. **AutoETS** (weight: 0.50)
2. **ARIMA** (weight: 0.30)
3. **Theta** (weight: 0.20)

### 1-Month Horizon
1. **AutoETS** (weight: 0.45)
2. **TBATS** (weight: 0.35)
3. **OptimizedTheta** (weight: 0.20)

### 3-Month Horizon
1. **AutoETS** (weight: 0.40)
2. **MSTL** (weight: 0.35)
3. **TBATS** (weight: 0.25)

### 6-Month Horizon
1. **AutoARIMA** (weight: 0.50)
2. **TBATS** (weight: 0.30)
3. **AutoETS** (weight: 0.20)

### 12-Month Horizon
1. **AutoARIMA** (weight: 0.45)
2. **DynamicTheta** (weight: 0.35)
3. **MSTL** (weight: 0.20)

## Model Metadata

Each model includes:
- `model_id` - Unique identifier
- `model_name` - E.g., "AutoETS", "TBATS"
- `bucket` - Bucket context (if bucket-specific)
- `horizon` - Forecast horizon (days)
- `mape` - Latest validation MAPE
- `ensemble_weight` - Weight in ensemble (0-1)
- `is_active` - Production status
- `artifact_path` - Storage location

## Model Artifacts

Stored in: `Data/models/production/YYYYMM/`

Format: DuckDB extension artifacts (native AnoFox format)

## Version Control

Model versions tracked via `model_id` naming:
- `model_autoets_1w_v1` - Initial version
- `model_autoets_1w_v2` - First retrain

Latest version always marked `is_active=TRUE`

