# ZL Forecasting Engine Specification

**Status:** Architecture Finalized  
**Last Updated:** December 3, 2025  
**Purpose:** Complete model stack for ZL soybean oil procurement forecasting

---

## Compute Topology

**MotherDuck:** Central warehouse + serving (schemas: raw, raw_staging, staging, features, training, forecast, reference, signals, ops, archive)

**Engine Host:** Mac M4 (future: VM) running DuckDB + AnoFox + PyTorch
- Attaches to MotherDuck: `md:usoil_intelligence`
- Executes all 45 models
- Writes results via idempotent MERGE (raw_staging → canonical)

**Vercel:** Read-only dashboard queries, API routes, no training

---

## Three-Tier Model Stack

### **Tier 1: AnoFox 31 Statistical Models (SQL-Native)**

**Three Sub-Engines:**

#### 1. ZL Price Level Engine
**Data:** ZL daily close from `training.daily_ml_matrix`

**Mandatory Models:**
- **Baselines:** Naive, SeasonalNaive, SeasonalWindowAverage, RandomWalkDrift
- **ETS Family:** ETS, AutoETS, Holt-Winters
- **ARIMA Family:** ARIMA, AutoARIMA
- **Theta Family:** Theta
- **Advanced Seasonal:** TBATS, MSTL, MFLES

**Output:** Price forecasts for 1W, 1M, 3M, 6M, 12M

#### 2. ZL Volatility Engine
**Data:** Realized vol (5d, 21d, 63d), VIX proxies

**Mandatory Models:**
- **GARCH** (conditional variance)
- ARIMA (log-vol if needed)
- ETS/Theta (slow vol drifts)

**Output:** Volatility forecasts + regime indicators

#### 3. Macro/Driver Baseline Engine
**Data:** FRED macro, USDA supply, policy indices

**Mandatory Models:**
- ETS/AutoETS (slow-moving trends)
- TBATS/MSTL (multi-seasonality: planting cycles)

**Output:** Macro baseline forecasts

**All outputs → `forecast.zl_anofox_panel`**

---

### **Tier 2: 7 Neural Models (PyTorch on Engine Host)**

| Model | Task | Role | Priority |
|-------|------|------|----------|
| **LSTM** | ZL price sequence modeling | Capture nonlinear temporal structure | MANDATORY |
| **GRU** | Same as LSTM | Diversify deep learners, faster convergence | MANDATORY |
| **TCN** | Short-medium horizon patterns | Best for 1W/1M in stable regimes | MANDATORY |
| **TFT** | Multi-horizon + exogenous vars | Foundation forecaster with attention | MANDATORY |
| **GARCH-LSTM** | Volatility forecasting | Hybrid: GARCH variance → LSTM residuals | MANDATORY (VIX) |
| **HAR-LSTM** | Multi-horizon vol (daily/weekly/monthly) | Realized vol nonlinear interactions | MANDATORY (VIX) |
| **N-BEATS** | Decomposable ZL price | Interpretable trend+seasonal blocks | OPTIONAL |

**Inputs:** AnoFox-cleaned data from `training.daily_ml_matrix`

**Outputs:** `forecast.zl_neural_panel`

---

### **Tier 3: 7 Meta-Ensemble Models**

| Model | Input | Role | Regime |
|-------|-------|------|--------|
| **XGBoost Meta-Learner** | 38 model forecasts | Nonlinear stacking | ALL |
| **LightGBM Meta-Learner** | 38 forecasts + regime features | Regime-aware weights | ALL |
| **Random Forest** | Subset of forecasts | Robust, low-variance blender | ALL |
| **ARIMA-LSTM Hybrid** | ARIMA residuals | Residual learner for 1M/3M | CALM/STRESSED |
| **ETS-LSTM Hybrid** | ETS residuals | Nonlinear around trend/seasonal | CALM |
| **MSTL-LSTM Hybrid** | MSTL components + residuals | Multiple seasonalities | STRESSED |
| **Transformer-GARCH Hybrid** | Vol series + regime features | Vol regime forecaster (gates weights) | CRISIS |

**Outputs:** `forecast.zl_ensemble_output`

---

## Model Selection Per Horizon (Production Ensemble)

**Active models per horizon: 5-7 maximum**

### 1-Week Horizon
**Regime: CALM**
1. SeasonalNaive (AnoFox) - weight 0.15
2. AutoETS (AnoFox) - weight 0.25
3. TCN (Neural) - weight 0.30
4. ARIMA-LSTM (Meta) - weight 0.20
5. XGBoost Meta-Learner - weight 0.10

**Regime: STRESSED**
1. GARCH (AnoFox) - weight 0.20
2. AutoARIMA (AnoFox) - weight 0.20
3. GARCH-LSTM (Neural) - weight 0.30
4. XGBoost Meta - weight 0.30

**Regime: CRISIS**
1. GARCH (AnoFox) - weight 0.35
2. GARCH-LSTM (Neural) - weight 0.35
3. Transformer-GARCH (Meta) - weight 0.30

### 1-Month Horizon
**Regime: CALM**
1. AutoETS (AnoFox) - weight 0.25
2. Theta (AnoFox) - weight 0.20
3. TFT (Neural) - weight 0.25
4. ARIMA-LSTM (Meta) - weight 0.20
5. XGBoost Meta - weight 0.10

**Regime: STRESSED**
1. TBATS (AnoFox) - weight 0.20
2. AutoARIMA (AnoFox) - weight 0.20
3. TFT (Neural) - weight 0.25
4. LightGBM Meta - weight 0.35

**Regime: CRISIS**
1. GARCH (AnoFox) - weight 0.30
2. GARCH-LSTM (Neural) - weight 0.30
3. HAR-LSTM (Neural) - weight 0.25
4. Transformer-GARCH (Meta) - weight 0.15

### 3-Month Horizon
**Regime: CALM**
1. AutoETS (AnoFox) - weight 0.25
2. TBATS (AnoFox) - weight 0.25
3. TFT (Neural) - weight 0.25
4. ETS-LSTM (Meta) - weight 0.15
5. XGBoost Meta - weight 0.10

**Regime: STRESSED**
1. MSTL (AnoFox) - weight 0.25
2. TBATS (AnoFox) - weight 0.20
3. TFT (Neural) - weight 0.25
4. LightGBM Meta - weight 0.30

**Regime: CRISIS**
1. GARCH (AnoFox) - weight 0.25
2. AutoARIMA (AnoFox) - weight 0.20
3. HAR-LSTM (Neural) - weight 0.30
4. Transformer-GARCH (Meta) - weight 0.25

### 6-Month Horizon
**Regime: CALM**
1. AutoETS (AnoFox) - weight 0.30
2. Theta (AnoFox) - weight 0.25
3. TFT (Neural) - weight 0.25
4. XGBoost Meta - weight 0.20

**Regime: STRESSED**
1. TBATS (AnoFox) - weight 0.30
2. MSTL (AnoFox) - weight 0.25
3. TFT (Neural) - weight 0.25
4. LightGBM Meta - weight 0.20

**Regime: CRISIS**
1. AutoARIMA (AnoFox) - weight 0.35
2. HAR-LSTM (Neural) - weight 0.35
3. LightGBM Meta - weight 0.30

### 12-Month Horizon
**Regime: CALM**
1. AutoETS (AnoFox) - weight 0.35
2. Theta (AnoFox) - weight 0.30
3. TFT (Neural) - weight 0.20
4. XGBoost Meta - weight 0.15

**Regime: STRESSED**
1. MSTL (AnoFox) - weight 0.35
2. TBATS (AnoFox) - weight 0.30
3. TFT (Neural) - weight 0.20
4. LightGBM Meta - weight 0.15

**Regime: CRISIS**
1. AutoARIMA (AnoFox) - weight 0.40
2. MSTL (AnoFox) - weight 0.30
3. LightGBM Meta - weight 0.30

---

## Driver Group Attribution

**Explainability Sources:**

1. **AnoFox Statistics** - OLS/Ridge/WLS regression (beta, t-stat)
2. **TFT Attention** - Per-feature attention weights
3. **XGBoost/LightGBM SHAP** - Tree-based feature attribution

**Aggregation:**
- All attributions mapped to driver groups via `reference.feature_to_driver_group_map`
- Output: `signals.driver_group_score_daily`, `signals.driver_group_contribution`

**Driver Groups:**
- BIOFUEL_ENERGY
- TRADE_TARIFF
- WEATHER_SUPPLY
- PALM_SUBSTITUTION
- MACRO_RISK
- POSITIONING
- POLICY_REGULATION
- TECHNICAL/REGIME

---

## Testing & Validation

### Vendor Tests
- AnoFox C++/SQLLogic test suite (vendor responsibility)

### Pipeline Tests (`ops.pipeline_test_runs`)
- Extension load checks
- Synthetic forecast sanity
- Daily execution timing

### Data Quality (`ops.data_quality_zl_daily`)
- QC flags (gaps, outliers, regime breaks)

### Model Validation (`training.model_backtest_zl`)
- Rolling backtest results
- Walk-forward validation
- Coverage calibration

### Operational Reporting (`signals.pipeline_metrics_daily`)
- Test pass rates
- Data quality scores
- Model edge vs. baseline
- Regime classification
- Composite pipeline score

---

## Schema Compliance

**All work stays within 8 datasets:**
- `raw` - Vendor data as-delivered
- `raw_staging` - Per-run temp (`<source>_<bucket>_<run_id>`)
- `staging` - Normalized panels
- `features` - Engineered (prefix-enforced)
- `training` - ML-ready matrix
- `reference` - Driver groups, model registry
- `signals` - Live scores, ensemble output
- `ops` - QC, tests, logs

**Column Prefix Contract (ruthless):**
- Unprefixed: `date`, `symbol` only
- All others: `fred_*`, `databento_*`, `weather_*`, `eia_*`, `usda_*`, `cftc_*`, `scrc_*`, `policy_trump_*`, `fx_*`

**Idempotence:**
- raw_staging (WRITE_TRUNCATE) → MERGE → raw.* on primary key
- Log to `ops.ingestion_completion`

---

## Production Deployment Rules

### Model Activation
**Maximum 7 active models per horizon per regime**

**Regime Detection:**
- GARCH conditional variance
- HAR-LSTM vol forecast
- Transformer-GARCH regime classifier

**Regime Definitions:**
- **CALM:** Vol Z-score < 1.0
- **STRESSED:** Vol Z-score 1.0-2.0
- **CRISIS:** Vol Z-score > 2.0

### Weight Recalibration
**Weekly:** Update ensemble weights based on 30-day MAE
**Monthly:** Re-evaluate model selection (add/remove from ensemble)
**Quarterly:** Full model review (manual)

### Guardrails
- No single model > 0.40 weight
- At least 1 AnoFox statistical model in every ensemble
- At least 1 neural model for horizons ≥ 1M
- GARCH mandatory for CRISIS regime

---

## Next Steps

1. Pin exact AnoFox 31 model list from docs
2. Build neural models 3-7 (GRU, TCN, GARCH-LSTM, HAR-LSTM, N-BEATS)
3. Build meta-ensemble models 1-7
4. Define walk-forward validation protocol
5. Create model registry schema in `reference.model_registry`

