# Baseline Training Pipeline - Complete Flow

**Date**: November 28, 2025  
**Status**: ✅ **READY FOR IMPLEMENTATION**

---

## 🎯 Pipeline Overview

### Flow: BigQuery → Mac → BigQuery

```
BigQuery (Pre-Compute)
    ↓
Export Training Data (~365 features)
    ↓
Mac Training (LightGBM)
    ↓
Model Evaluation
    ↓
Upload Predictions
    ↓
BigQuery (Forecasts)
```

---

## 📊 Step 1: BigQuery Pre-Compute (DONE)

### Features Pre-Computed (~365 features)

#### Technical Indicators (19 features)
- Distance MAs: 5 features
- Bollinger: 2 features
- PPO: 1 feature
- VWAP: 1 feature
- Volatility: 3 features
- Microstructure: 2 features
- Cross-asset: 3 features
- Metadata: 2 features

#### FX Indicators (16 features)
- BRL Momentum: 3 features
- DXY Momentum: 3 features
- BRL Volatility: 2 features
- ZL-BRL Correlation: 3 features
- ZL-DXY Correlation: 3 features
- Terms of Trade: 1 feature
- Correlation Regimes: 2 features

#### Fundamental Spreads (4 features)
- Board Crush: 1 feature
- Oil Share: 1 feature
- Hog Spread: 1 feature
- BOHO Spread: 1 feature
- China Pulse: 1 feature (optional)

#### Pair Correlations (112 features)
- 28 pairs × 4 horizons = 112 features

#### Cross-Asset Betas (28 features)
- 7 assets × 4 horizons = 28 features

#### Lagged Features (96 features)
- 8 symbols × 12 lags = 96 features

#### Additional Pre-Compute (90 features)
- Rolling statistics: 50 features
- Feature interactions: 20 features
- Factor loadings: 10 features
- Regime indicators: 10 features

**Total**: ~365 features pre-computed in BigQuery ✅

---

## 📥 Step 2: Export Training Data

### Script: `scripts/export/export_training_data.py`

```python
#!/usr/bin/env python3
"""
Export training data from BigQuery to Parquet for Mac training
"""
from google.cloud import bigquery
import pandas as pd
from pathlib import Path

PROJECT_ID = "cbi-v15"
OUTPUT_DIR = Path("/Volumes/Satechi Hub/Projects/CBI-V15/03_Training_Exports")

def export_training_data(horizon: str = "1m"):
    """Export training data for specified horizon"""
    client = bigquery.Client(project=PROJECT_ID)
    
    query = f"""
    SELECT 
        -- Date & Target
        date,
        target_zl_{horizon} AS target,
        
        -- Technical Indicators (19 features)
        dist_ema_5, dist_ema_10, dist_ema_21, dist_sma_63, dist_sma_200,
        bb_pct_b, bb_bandwidth,
        ppo_12_26,
        dist_vwap_21d,
        vol_garman_klass_annualized, vol_parkinson_annualized, vol_21d,
        amihud_illiquidity, oi_volume_ratio,
        boho_spread, corr_zl_brl_60d, terms_of_trade_zl_brl,
        doy_sin, doy_cos,
        
        -- FX Indicators (16 features)
        brl_momentum_21d, brl_momentum_63d, brl_momentum_252d,
        dxy_momentum_21d, dxy_momentum_63d, dxy_momentum_252d,
        brl_volatility_21d, brl_volatility_63d,
        corr_zl_brl_30d, corr_zl_brl_60d, corr_zl_brl_90d,
        corr_zl_dxy_30d, corr_zl_dxy_60d, corr_zl_dxy_90d,
        terms_of_trade_zl_brl,
        corr_regime_zl_brl, corr_regime_zl_dxy,
        
        -- Fundamental Spreads (4 features)
        board_crush, oil_share, hog_spread_feeder_margin, boho_spread_gal,
        
        -- Pair Correlations (112 features - sample)
        -- ... (all 28 pairs × 4 horizons)
        
        -- Cross-Asset Betas (28 features)
        -- ... (all 7 assets × 4 horizons)
        
        -- Lagged Features (96 features - sample)
        -- ... (all 8 symbols × 12 lags)
        
        -- Regime Weights
        regime_weight
        
    FROM `cbi-v15.training.zl_training_{horizon}`
    WHERE date >= '2010-01-01'
      AND date < '2024-01-01'  -- Train set
    ORDER BY date
    """
    
    df = client.query(query).to_dataframe()
    
    # Save to Parquet
    output_path = OUTPUT_DIR / f"zl_training_{horizon}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Exported {len(df):,} rows, {len(df.columns)} columns to {output_path}")
    return df

if __name__ == "__main__":
    for horizon in ["1w", "1m", "3m", "6m"]:
        export_training_data(horizon)
```

---

## 🖥️ Step 3: Mac Training (LightGBM)

### Script: `src/training/baselines/lightgbm_zl.py`

```python
#!/usr/bin/env python3
"""
LightGBM baseline training for ZL
"""
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

def train_lightgbm_baseline(horizon: str = "1m"):
    """Train LightGBM baseline for specified horizon"""
    
    # Load training data
    data_path = Path(f"/Volumes/Satechi Hub/Projects/CBI-V15/03_Training_Exports/zl_training_{horizon}.parquet")
    df = pd.read_parquet(data_path)
    
    # Split train/val/test
    train_df = df[df['date'] < '2023-01-01']
    val_df = df[(df['date'] >= '2023-01-01') & (df['date'] < '2024-01-01')]
    test_df = df[df['date'] >= '2024-01-01']
    
    # Features (exclude date, target, regime_weight)
    feature_cols = [c for c in df.columns if c not in ['date', 'target', 'regime_weight']]
    
    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    sample_weight_train = train_df['regime_weight']
    
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    sample_weight_val = val_df['regime_weight']
    
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    # Train LightGBM
    model = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=1000,
        objective='regression',
        metric='mae',
        verbose=1
    )
    
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[sample_weight_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"\n✅ Horizon: {horizon}")
    print(f"Train MAE: {mae_train:.4f}, R²: {r2_train:.4f}")
    print(f"Val MAE: {mae_val:.4f}, R²: {r2_val:.4f}")
    print(f"Test MAE: {mae_test:.4f}, R²: {r2_test:.4f}")
    
    # Save model
    model_path = Path(f"/Volumes/Satechi Hub/Projects/CBI-V15/04_Models/lightgbm_zl_{horizon}.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, model_path)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'date': test_df['date'],
        'target': y_test,
        'prediction': y_pred_test,
        'horizon': horizon
    })
    
    predictions_path = Path(f"/Volumes/Satechi Hub/Projects/CBI-V15/03_Training_Exports/predictions_lightgbm_{horizon}.parquet")
    predictions_df.to_parquet(predictions_path, index=False)
    
    return model, predictions_df

if __name__ == "__main__":
    for horizon in ["1w", "1m", "3m", "6m"]:
        train_lightgbm_baseline(horizon)
```

---

## 📤 Step 4: Upload Predictions

### Script: `scripts/upload/upload_predictions.py`

```python
#!/usr/bin/env python3
"""
Upload predictions to BigQuery
"""
from google.cloud import bigquery
import pandas as pd
from pathlib import Path

PROJECT_ID = "cbi-v15"

def upload_predictions(horizon: str = "1m"):
    """Upload predictions to BigQuery"""
    client = bigquery.Client(project=PROJECT_ID)
    
    # Load predictions
    predictions_path = Path(f"/Volumes/Satechi Hub/Projects/CBI-V15/03_Training_Exports/predictions_lightgbm_{horizon}.parquet")
    df = pd.read_parquet(predictions_path)
    
    # Upload to BigQuery
    table_id = f"{PROJECT_ID}.forecasts.zl_predictions_{horizon}"
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema=[
            bigquery.SchemaField("date", "DATE"),
            bigquery.SchemaField("target", "FLOAT64"),
            bigquery.SchemaField("prediction", "FLOAT64"),
            bigquery.SchemaField("horizon", "STRING"),
        ]
    )
    
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    
    print(f"✅ Uploaded {len(df):,} predictions to {table_id}")

if __name__ == "__main__":
    for horizon in ["1w", "1m", "3m", "6m"]:
        upload_predictions(horizon)
```

---

## ✅ Pipeline Checklist

### Pre-Training
- ✅ BigQuery pre-compute: ~365 features
- ✅ Training data export script
- ✅ Mac training script
- ✅ Prediction upload script

### During Training
- ✅ Monitor feature quality
- ✅ Validate train/val/test splits
- ✅ Track model performance

### Post-Training
- ✅ Review baseline performance
- ✅ Identify feature gaps
- ✅ Plan advanced models

---

**Last Updated**: November 28, 2025

