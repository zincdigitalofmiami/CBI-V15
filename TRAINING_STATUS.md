# CBI-V15 Training Status & Data Gaps

**Last Updated:** December 16, 2025  
**Commit:** `9b7d037` - Phase 0 infrastructure + training framework complete

---

## ✅ Training Framework Complete

### AutoGluon 1.4 Hybrid Architecture

#### L0 Bucket Specialists
**File:** `src/training/autogluon/bucket_specialist.py`

- **Status:** ✅ Implementation complete, ready for first training run
- **Architecture:**
  - 8 TabularPredictor models (one per Big 8 bucket)
  - `presets='extreme_quality'` with foundation models
  - Foundation models: Mitra, TabPFNv2, TabICL, TabM
  - Quantile regression: P10, P50, P90
  - Horizons: 1w, 1m, 3m, 6m
  - Output: `training.bucket_predictions` (OOF predictions)

#### TimeSeriesPredictor
**Files:** 
- `src/training/autogluon/timeseries_trainer.py`
- `src/training/autogluon/mitra_trainer.py`

- **Status:** ✅ Implementation complete
- **Models:**
  - Chronos-Bolt (zero-shot baseline, CPU-compatible)
  - Mitra (Metal-accelerated time series, Mac M4 optimized)

#### Baseline Models
**Directory:** `src/training/baselines/`

- **Status:** ✅ Complete
- **Models:** XGBoost, CatBoost, LightGBM
- **Purpose:** Validation and comparison benchmarks

---

## ✅ Data Ingestion Complete

### Databento Futures OHLCV

#### Daily Collection
**File:** `src/ingestion/databento/collect_daily.py`

- **Status:** ✅ Production ready
- **Coverage:** 38 futures symbols (full Big 8 coverage)
- **Target:** `raw.databento_futures_ohlcv_1d`
- **Frequency:** Daily (GitHub Actions: 4x/day)
- **Features:**
  - Fixed column naming (`as_of_date`)
  - Idempotent `INSERT OR REPLACE` logic
  - 1000-day backfill capability

#### Hourly Collection
**File:** `src/ingestion/databento/collect_hourly.py`

- **Status:** ✅ Production ready
- **Coverage:** High-liquidity subset (ZL/ZS/ZM/ZC/ZW + Energy + Metals)
- **Target:** `raw.databento_futures_ohlcv_1h`
- **Frequency:** Daily (GitHub Actions: 4x/day)
- **Features:**
  - Rolling 365-day backfill window
  - Batch download optimization
  - 14+ years of historical data available (2010-06-06 to present)

#### Options Collection
**File:** `src/ingestion/databento/collect_options_daily.py`

- **Status:** ✅ Implementation complete (not yet in production)
- **Coverage:** ZL options (soybean oil)
- **Target:** `raw.databento_options_ohlcv`
- **Purpose:** Implied volatility surface for Volatility bucket

---

## 🚧 Missing Data Sources (Phase 1)

### Critical Priority (Blocks Big 8 Buckets)

#### 1. EPA RIN Prices (FREE)
**Status:** ❌ NOT IMPLEMENTED  
**Impact:** Blocks Biofuel bucket (6/8)  
**Source:** https://www.epa.gov/fuels-registration-reporting-and-compliance-help/rin-trades-and-price-information  
**Data:** Weekly volume-weighted average RIN prices (D3, D4, D5, D6)  
**Target Table:** `raw.eia_petroleum` (series_id: rin_d3_price, rin_d4_price, rin_d5_price, rin_d6_price)  
**Frequency:** Weekly (updated monthly by EPA)  
**Implementation:** Need Trigger.dev job `trigger/epa_rin_prices.ts`

#### 2. USDA Export Sales (FREE)
**Status:** ❌ MOCK DATA ONLY  
**Impact:** Blocks China bucket (2/8)  
**Source:** https://apps.fas.usda.gov/esrquery/api/v1/export-sales  
**Data:** Weekly export sales by commodity and country  
**Target Table:** `raw.usda_export_sales`  
**Frequency:** Weekly (Thursday releases)  
**Implementation:** 
- Remove mock data from `src/ingestion/usda/ingest_export_sales.py`
- Create Trigger.dev job `trigger/usda_export_sales.ts`

#### 3. CFTC COT (FREE)
**Status:** ❌ NOT IMPLEMENTED  
**Impact:** Blocks Volatility bucket (8/8)  
**Source:** https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm  
**Data:** Weekly Commitments of Traders positioning (Managed Money, Commercials, Non-reportable)  
**Target Table:** `raw.cftc_cot`  
**Frequency:** Weekly (Friday 3:30 PM ET)  
**Implementation:** Create Trigger.dev job `trigger/cftc_cot.ts`

#### 4. FRED Missing Series (FREE)
**Status:** ⚠️ PARTIAL  
**Impact:** Blocks Fed (4/8) + Volatility (8/8) buckets  
**Missing Series:**
- `DFEDTARU` - Fed Funds Target Rate (Upper Bound)
- `VIXCLS` - CBOE Volatility Index (VIX)

**Current Coverage:** 24 series (DGS2, DGS10, T10Y2Y, DTWEXBGS, etc.)  
**Target Table:** `raw.fred_economic`  
**Frequency:** Daily  
**Implementation:** Add missing series to `config/fred_price_series.yaml`

#### 5. Farm Policy News (MANDATORY)
**Status:** ❌ NOT IMPLEMENTED  
**Impact:** Blocks China (2/8) + Tariff (5/8) buckets  
**Source:** https://farmpolicynews.illinois.edu/  
**Data:** Trade policy news, tariff announcements, China demand signals  
**Target Table:** `raw.farm_policy_news`  
**Frequency:** Daily  
**Implementation:** Create Trigger.dev job `trigger/farm_policy_news.ts` with ScrapeCreator API

#### 6. farmdoc Daily (FREE)
**Status:** ❌ NOT IMPLEMENTED  
**Impact:** Supplemental for all 8 buckets  
**Source:** https://farmdocdaily.illinois.edu/  
**Data:** University of Illinois agricultural analysis and commentary  
**Target Table:** `raw.farmdoc_daily`  
**Frequency:** Daily  
**Implementation:** Create Trigger.dev job `trigger/farmdoc_daily.ts` with ScrapeCreator API

#### 7. USDA WASDE Reports (FREE)
**Status:** ❌ NOT IMPLEMENTED  
**Impact:** Blocks Crush (1/8) + China (2/8) buckets  
**Source:** https://www.usda.gov/oce/commodity/wasde  
**Data:** Monthly World Agricultural Supply and Demand Estimates  
**Target Table:** `raw.usda_wasde`  
**Frequency:** Monthly (typically 12th of each month)  
**Implementation:** Create Trigger.dev job `trigger/usda_wasde.ts`

---

## 📊 Big 8 Bucket Data Coverage

| Bucket | Required Data Sources | Status | Coverage |
|--------|----------------------|--------|----------|
| **1. Crush** | Databento (ZL/ZS/ZM) ✅, NOPA ✅, USDA WASDE ❌ | 🟡 Partial | 80% |
| **2. China** | Databento (HG) ✅, USDA Export Sales ❌, Farm Policy News ❌ | 🔴 Critical | 40% |
| **3. FX** | FRED ✅, Databento (6L/DX) ✅ | 🟢 Complete | 100% |
| **4. Fed** | FRED ✅ (missing DFEDTARU) | 🟡 Partial | 90% |
| **5. Tariff** | Farm Policy News ❌ | 🔴 Critical | 0% |
| **6. Biofuel** | EIA ✅, EPA RIN Prices ❌ | 🟡 Partial | 50% |
| **7. Energy** | EIA ✅, Databento (CL/HO/RB) ✅ | 🟢 Complete | 100% |
| **8. Volatility** | FRED ✅ (missing VIXCLS), Databento VIX ✅, CFTC COT ❌ | 🟡 Partial | 90% |

### Legend
- 🟢 **Complete** (100%): All required data sources operational
- 🟡 **Partial** (50-90%): Core data available, missing supplemental sources
- 🔴 **Critical** (0-40%): Missing critical data sources, bucket cannot train effectively

---

## 🎯 Phase 1 Implementation Priority

### Week 1 (High Priority)
1. **EPA RIN Prices** - Biofuel bucket blocker
2. **USDA Export Sales** - China bucket blocker
3. **Farm Policy News** - China + Tariff bucket blocker

### Week 2 (Medium Priority)
4. **CFTC COT** - Volatility bucket enhancement
5. **FRED Missing Series** - Fed + Volatility bucket enhancement
6. **farmdoc Daily** - All buckets supplemental

### Week 3 (Low Priority)
7. **USDA WASDE** - Crush + China bucket enhancement

---

## 🚀 First Training Run Readiness

### Ready to Train (3/8 buckets)
- ✅ **FX Bucket** (100% data coverage)
- ✅ **Energy Bucket** (100% data coverage)
- 🟡 **Fed Bucket** (90% data coverage - can train with current data)

### Needs Data (5/8 buckets)
- ❌ **Crush Bucket** (80% - missing WASDE)
- ❌ **China Bucket** (40% - missing Export Sales + Farm Policy News)
- ❌ **Tariff Bucket** (0% - missing Farm Policy News)
- ❌ **Biofuel Bucket** (50% - missing EPA RIN Prices)
- ❌ **Volatility Bucket** (90% - missing CFTC COT)

### Recommendation
**Option 1 (Partial Training):** Train 3 ready buckets (FX, Energy, Fed) to validate pipeline  
**Option 2 (Wait for Data):** Implement all 7 missing data sources before first training run  
**Option 3 (Phased Approach):** Train ready buckets now, retrain all 8 buckets after Phase 1 complete

---

## 📝 Training Execution Checklist

### Pre-Training (Phase 0 Complete ✅)
- [x] AutoGluon 1.4 installed on Mac M4
- [x] Local DuckDB mirror architecture documented
- [x] MotherDuck sync script operational
- [x] Bucket specialist training script complete
- [x] Feature engineering SQL macros validated
- [x] Databento daily + hourly ingestion operational

### Phase 1 (Data Sources)
- [ ] EPA RIN Prices ingestion
- [ ] USDA Export Sales ingestion (remove mock data)
- [ ] CFTC COT ingestion
- [ ] FRED missing series added
- [ ] Farm Policy News ingestion
- [ ] farmdoc Daily ingestion
- [ ] USDA WASDE ingestion

### Phase 2 (First Training Run)
- [ ] Sync MotherDuck → Local DuckDB
- [ ] Validate feature matrix completeness
- [ ] Train 8 bucket specialists (L0)
- [ ] Validate OOF predictions in `training.bucket_predictions`
- [ ] Train L1 meta-learner
- [ ] Generate L2 ensemble forecasts
- [ ] Upload predictions to MotherDuck (`forecasts.zl_predictions`)

### Phase 3 (Production Deployment)
- [ ] Schedule daily training job (Trigger.dev)
- [ ] Schedule daily forecast job (Trigger.dev)
- [ ] Schedule model monitoring job (Trigger.dev)
- [ ] Configure Slack/email notifications
- [ ] Deploy dashboard to Vercel

---

## 📈 Expected Timeline

| Phase | Tasks | Duration | Status |
|-------|-------|----------|--------|
| Phase 0 | Infrastructure + Training Framework | 8-12 hours | ✅ COMPLETE |
| Phase 1 | Missing Data Sources (7 sources) | 12-16 hours | 🚧 IN PROGRESS |
| Phase 2 | First Training Run | 8-10 hours | ⏳ PENDING |
| Phase 3 | Production Deployment | 8-12 hours | ⏳ PENDING |

**Total Estimated Time:** 36-50 hours (1.5-2 weeks at 4-6 hours/day)

---

## 🔗 Key Files

### Training
- `src/training/autogluon/bucket_specialist.py` - L0 bucket specialist trainer
- `src/training/autogluon/timeseries_trainer.py` - TimeSeriesPredictor wrapper
- `src/training/autogluon/mitra_trainer.py` - Mitra foundation model trainer
- `scripts/train_bucket_specialists.sh` - Training orchestration script

### Data Ingestion
- `src/ingestion/databento/collect_daily.py` - Daily OHLCV (38 symbols)
- `src/ingestion/databento/collect_hourly.py` - Hourly OHLCV (high-liquidity subset)
- `src/ingestion/databento/collect_options_daily.py` - Options data (ZL)

### Configuration
- `config/bucket_feature_selectors.yaml` - Big 8 bucket feature mappings
- `config/training/model_config.yaml` - AutoGluon training parameters
- `config/fred_price_series.yaml` - FRED series configuration

### Documentation
- `.cursor/plans/ALL_PHASES_INDEX.md` - Master implementation plan
- `.cursor/plans/PHASE_1_DETAILED.md` - Phase 1 data source details
- `docs/architecture/MASTER_PLAN.md` - V15.1 architecture overview
- `AGENTS.md` - Engineering agent guardrails

---

## 💡 Next Actions

1. **Implement EPA RIN Prices ingestion** (highest priority - blocks Biofuel bucket)
2. **Remove USDA Export Sales mock data** (high priority - blocks China bucket)
3. **Implement Farm Policy News scraper** (high priority - blocks China + Tariff buckets)
4. **Add FRED missing series** (medium priority - enhances Fed + Volatility buckets)
5. **Implement CFTC COT ingestion** (medium priority - enhances Volatility bucket)
6. **Consider partial training run** (3 ready buckets: FX, Energy, Fed) to validate pipeline

---

**Status:** Phase 0 complete ✅ | Phase 1 in progress 🚧 | Ready for first partial training run 🎯
