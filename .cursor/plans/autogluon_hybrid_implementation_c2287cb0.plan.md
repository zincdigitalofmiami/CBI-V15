---
name: ""
overview: ""
todos: []
---

# AutoGluon 1.4 Hybrid Implementation Plan

## Architecture Summary

**PRIMARY GOAL:** Predict probabilistic ZL (soybean oil futures) returns/levels at 1w, 1m, 3m, 6m horizons using AutoGluon 1.4 ensemble on Mac M4.

### Model Stack (4 Layers):

**L0: Bucket Specialists** (9 TabularPredictors)

- 8 Big 8 bucket-specific specialists + 1 main ZL predictor  
- Each uses `presets='extreme_quality'`
   - Problem type: `quantile` regression (P10, P50, P90)
   - Models: LightGBM, CatBoost, XGBoost, FastAI, PyTorch NN, TabPFNv2, Mitra, TabICL, TabM
   - AutoGluon trains 10-15 models per specialist, creates WeightedEnsemble_L2 automatically

**L1: Meta-Learner** (AutoGluon Stacking Layer)

- AutoGluon trains additional models on L0 OOF predictions
- Automatically created when `num_stack_levels=1` is set
- NO manual code needed (AutoGluon handles this internally)

**L2: Greedy Weighted Ensemble** (AutoGluon WeightedEnsemble_L2)

- AutoGluon automatically creates WeightedEnsemble_L2
- Combines L0 + L1 models with learned optimal weights
- Appears in `predictor.leaderboard()` as "WeightedEnsemble_L2"
- This IS the final ensemble - NO manual ensemble_combiner.py needed!

**L2: Production Forecasts**

- Upload to MotherDuck: `forecasts.zl_predictions` (horizons: 1w, 1m, 3m, 6m)
- Dashboard queries this table

**L3: Monte Carlo Simulation** (Risk Metrics ONLY)

- Input: L2 predictions (P10, P50, P90)
- Purpose: VaR/CVaR calculation (NOT forecasting)
- Output: `forecasts.monte_carlo_scenarios` (analytics only)
- Location: `src/simulators/monte_carlo_sim.py` (NOT in baselines)

### Feature Architecture (ADDITIVE):

**Layer 1: Core Macro/FX** (~50 features)

- ALL buckets inherit these as their foundation
- FX (16): BRL/DXY momentum, volatility, correlations, Terms of Trade
- Macro (12): Fed funds, yields, curves, NFCI, STLFSI4, VIX
- Price/Volume (3): ZL close, volume, OI
- Cross-Asset (5): Board crush, oil share, BOHO, HG proxy, DX
- View: `features.core_macro_fx`

**Layer 2: Bucket-Specific** (10-25 features ADDED per bucket)

- Crush: ZL/ZS/ZM spreads, board crush specifics
- China: USDA export sales, HG-ZS correlation, Farm Policy News
- FX: CNY/MXN pairs, multi-pair correlations
- Fed: Policy indicators, FOMC analysis
- Tariff: Trump sentiment, Section 301, trade policy
- Biofuel: EPA RIN D4/D6 (weekly→daily filled), BOHO, biodiesel
- Energy: CL/HO/RB, crack spreads, CL-ZL correlation
- Volatility: VIX term structure, realized vol, stress indices

**Main ZL Predictor:** ALL ~300 features (complete 360° view)

### High-Volume News & Market Data Feeds:

**Dedicated Trigger.dev jobs per bucket** (hundreds of articles/data points daily):

Each of the 8 Big 8 buckets receives:

1. **Crush**: farmdoc Grain Outlook articles (daily), crush spread market data
2. **China**: Farm Policy News (trade category), USDA export sales (weekly), HG copper data (daily)
3. **FX**: BRL/DXY/CNY/MXN futures data (daily), multi-currency correlations
4. **Fed**: Farm Policy News (budget category), Fed speeches, FRED macro series (daily)
5. **Tariff**: Trump Truth Social posts, Farm Policy News (trade), ScrapeCreators tariff bucket (hundreds/day)
6. **Biofuel**: farmdoc Daily RIN analysis, EPA RIN prices D3/D4/D5/D6 (weekly), EIA biodiesel data
7. **Energy**: CL/HO/RB/NG prices (daily), EIA petroleum reports, crack spreads
8. **Volatility**: VIX (daily), STLFSI4 (daily), stress indices, realized volatility calculations

**Plus Main ZL Predictor**: ALL data from all 8 buckets (~300 features total)

**Data Volume**: Hundreds of news articles + thousands of market data points per day across all feeds

---

## Phase 0: Critical Infrastructure + Bug Fixes (Day 1-2)

**DEPENDENCY CHAIN**: Must execute in EXACT order (0.1 → 0.2 → 0.3 → 0.4)

**WHY THIS EXACT ORDER?**

- **0.1** (Install libomp) → Without this, LightGBM SEGFAULTS on Mac M4 (blocks ALL training)
- **0.2** (Fix as_of_date) → Without this, SQL joins FAIL (blocks view creation)
- **0.3** (Create core_macro_fx) → Without this, NO bucket has base features (blocks training)
- **0.4** (Verify Terms of Trade) → Without this, Inf/NaN CRASH training (blocks stability)

**Timeline**: 10-20 minutes total for all 4 scripts

**Critical Files Created in Phase 0.0:**

- ✅ `scripts/setup/install_autogluon_mac.sh` - Mac M4 libomp fix
- ✅ `database/definitions/03_features/core_macro_fx.sql` - ~50 base features view  
- ✅ `scripts/validation/verify_core_macro_fx.py` - Terms of Trade validator
- ✅ `src/reporting/training_auditor.py` - Hot-audit loop (immediate reporting)
- ✅ `database/macros/anofox_guards.sql` - SQL data quality guards
- ✅ `config/bucket_feature_selectors.yaml` - ADDITIVE feature model (8 buckets + main)

### 0.0 ✅ AI Agent Context Files (COMPLETED)

All context files updated:

- `docs/architecture/MASTER_PLAN.md` - Updated for DuckDB/MotherDuck + AutoGluon (`presets='extreme_quality'`)
- `.cursorrules` - Updated for DuckDB/MotherDuck architecture  
- `.augment.md` - Created (289 lines workspace instructions for Augment Code)
- `AGENTS.md` - Added AI plan building guide (150+ lines)
- `config/bucket_feature_selectors.yaml` - Created (ADDITIVE feature model)
- `src/reporting/training_auditor.py` - Created (hot-audit loop)
- `database/macros/anofox_guards.sql` - Created (SQL data quality guards)
- `scripts/sync_motherduck_to_local.py` - Created (MotherDuck → Local sync)
- Removed all BigQuery/Dataform references (62 files archived)
- Removed all Copilot references (not used)
- Fixed: "BEST" preset → `presets='extreme_quality'` (full power)
- Removed foundation model GPU warnings; they will run on M4 CPU (just slower)

### 0.1 ⚠️ Install libomp (Mac M4) - MUST RUN FIRST

**Script:** `bash scripts/setup/install_autogluon_mac.sh`

**Why First:**

- LightGBM (AutoGluon's core engine) will **SEGFAULT** without OpenMP (libomp)
- Apple Silicon does NOT include OpenMP by default
- **BLOCKS**: All AutoGluon training (nothing works without this)
- **Time**: 5-10 minutes

**What it does:**

1. `brew install libomp` - Install OpenMP library
2. Build LightGBM from source with explicit libomp linker flags
3. Install AutoGluon 1.4 (tabular + timeseries)
4. Verify all imports work (LightGBM, CatBoost, XGBoost, FastAI)

**Dependencies:** None (just Homebrew)

**Validation:**

```bash
python3 -c "from autogluon.tabular import TabularPredictor; print('AutoGluon ready!')"
# Should print: AutoGluon ready!
# Should NOT segfault
```

---

### 0.2 ⚠️ Fix Column Names: date → as_of_date

**Files:**

- `src/ingestion/databento/collect_daily.py` (lines 146, 160, 165, 196, 234)
- Any other ingestion scripts using `date` column

**Why Second:**

- `features.core_macro_fx` view relies on **consistent join keys**
- All tables MUST use `as_of_date` (not `date`, not `timestamp`, not `ts`)
- **BLOCKS**: Script 0.3 (view creation will fail if joins break)
- **Time**: 2-5 minutes

**Fix:**

```python
# BEFORE
df.rename(columns={"ts_event": "date"})

# AFTER
df.rename(columns={"ts_event": "as_of_date"})
```

**Dependencies:** None (just Python file edits)

**Validation:**

```bash
python src/ingestion/databento/collect_daily.py --symbol ZL --days 5
# Should insert to raw.databento_ohlcv_daily without column errors
```

---

### 0.3 ⚠️ CREATE features.core_macro_fx View - CRITICAL

**Script:** `python scripts/setup_database.py --both` (after creating SQL file)

**File:** `database/definitions/03_features/core_macro_fx.sql` ✅ CREATED

**Why Third:**

- **ALL 8 bucket specialists** inherit this view as their base feature set (~50 features)
- Without this, buckets have NO core macro/FX features
- **BLOCKS**: ALL bucket training (buckets query this view)
- **Time**: 1 minute (SQL view creation)

**What it creates:**

```sql
CREATE OR REPLACE VIEW features.core_macro_fx AS
-- ~50 base features:
-- • FX (16): Rate diff, BRL/DXY momentum, volatility, correlations, Terms of Trade
-- • Macro (12): Fed funds, yields, curves, NFCI, STLFSI4, VIX, UNRATE, CPI
-- • Price/Volume (3): ZL close, volume, OI
-- • Cross-Asset (5): Board crush, oil share, BOHO, HG proxy, DX
```

**Dependencies:** 0.2 (needs `as_of_date` for joins)

**Validation:**

```sql
SELECT COUNT(*) FROM features.core_macro_fx;
-- Should return >1000 rows (daily data from 2010+)
```

---

### 0.4 ⚠️ VERIFY Terms of Trade - Data Quality Gate

**Script:** `python scripts/validation/verify_core_macro_fx.py` ✅ CREATED

**Why Fourth:**

- Terms of Trade divides by BRL price (6L futures)
- If BRL = 0 or missing → **Inf/NaN crashes AutoGluon**
- Must validate BEFORE training starts
- **BLOCKS**: Training stability (prevents crashes mid-training)
- **Time**: 1 minute

**What it checks:**

1. Terms of Trade returns non-null values (>95% coverage)
2. No Inf/NaN in calculations (max < 1e10)
3. BRL prices never zero (guard works)
4. All core features have <10% nulls

**Dependencies:** 0.3 (view must exist first)

**Validation:**

```bash
python scripts/validation/verify_core_macro_fx.py
# Should print: ✅ CORE_MACRO_FX VERIFICATION COMPLETE
# Should NOT print: ❌ FAIL
```

**Success Criteria:**

- Terms of Trade null% < 5%
- No Inf values (max < 1e10, min > -1e10)
- All core features null% < 10%
- View has >1000 rows

---

### 0.5 ✅ Additional Phase 0 Fixes (COMPLETED)

These critical fixes were already completed in Phase 0.0:

**Data Pipeline Fixes:**

- ✅ FRED table: `fred_observations` → `fred_economic` in `big8_bucket_features.sql`
- ✅ EIA table: `eia_biofuels` → `eia_petroleum` in `big8_bucket_features.sql`
- ✅ Weekly→Daily fill: Added `LAST_VALUE` carry-forward for EPA RIN prices (prevents 80% data loss in Biofuel bucket)

**Infrastructure:**

- ✅ Sync script: Created `scripts/sync_motherduck_to_local.py` (MotherDuck → Local mirror)
- ✅ Hot-audit: Created `src/reporting/training_auditor.py` (immediate reporting after each bucket)
- ✅ SQL guards: Created `database/macros/anofox_guards.sql` (fail-fast data validation)
- ✅ Training logs: Created `database/definitions/01_raw/ops_training_logs.sql` (audit storage)

**Configuration:**

- ✅ Feature config: Created `config/bucket_feature_selectors.yaml` (ADDITIVE model: core + bucket-specific)
- ✅ Augment setup: Created `.augment.md` (289 lines workspace instructions for Augment Code)

**Documentation:**

- ✅ Fixed MASTER_PLAN.md: Updated to `presets='extreme_quality'` (Mac M4 CPU-compatible, slower without GPU)
- ✅ Fixed requirements.txt: Added GPU warning for foundation models
- ✅ Updated AGENTS.md: Added AI plan building guide (150+ lines)

---

## ✅ Phase 0 Completion Checklist

Before proceeding to Phase 1, verify ALL 4 scripts completed successfully:

```bash
# Script 0.1: Install libomp
bash scripts/setup/install_autogluon_mac.sh
# Expected: ✅ AutoGluon 1.4 Installation Complete

# Script 0.2: Fix column names
# Manual edit or run sed replacement
# Expected: All ingestion scripts use as_of_date

# Script 0.3: Deploy core_macro_fx view
python scripts/setup_database.py --both
# Expected: features.core_macro_fx view created

# Script 0.4: Verify Terms of Trade
python scripts/validation/verify_core_macro_fx.py
# Expected: ✅ CORE_MACRO_FX VERIFICATION COMPLETE

# Final check: Sync test
python scripts/sync_motherduck_to_local.py --dry-run
# Expected: Shows tables to sync from MotherDuck
```

**Success Criteria:**

- ✅ AutoGluon imports work (no segfault)
- ✅ All tables use `as_of_date` column
- ✅ `features.core_macro_fx` view exists with ~50 features
- ✅ Terms of Trade has <5% nulls, no Inf/NaN
- ✅ Local DuckDB can sync from MotherDuck

**IF ALL PASS** → Proceed to Phase 1 (Data Ingestion)

**IF ANY FAIL** → STOP and debug before proceeding

### 0.5 Add Missing FRED Series

**File:** `src/ingestion/fred/collect_fred_financial_conditions.py`

**Missing:** `DFEDTARU` (Fed Funds Target), `VIXCLS` (VIX)

**Fix:** Add to SERIES list in collector

---

## Phase 1: Critical Data Feeds (Week 1)

### 1.1 EPA RIN Prices (CRITICAL - FREE)

**Creates:** `trigger/ingestion/energy_biofuels/epa_rin_prices.ts`

**Source:** EPA EMTS via Qlik Sense reports

**URLs:**

- `https://www.epa.gov/fuels-registration-reporting-and-compliance-help/rin-trades-and-price-information`
- `https://www.epa.gov/fuels-registration-reporting-and-compliance-help/public-data-renewable-fuel-standard`

**Data:** Weekly volume-weighted average RIN prices (D3, D4, D5, D6)

**Target:** `raw.epa_rin_prices` (CRITICAL - referenced in SQL macros, table definition must be created)

**Frequency:** Weekly (updated monthly by EPA)

**Backup:** Growth Energy charts at `https://growthenergy.org/data-set-category/rin-prices/`

**Note:** Table `raw.epa_rin_prices` must be created in `database/definitions/01_raw/epa_rin_prices.sql` before job can run.

### 1.2 USDA Export Sales (Remove Mock Data)

**File:** `src/ingestion/usda/ingest_export_sales.py`

**Issue:** Uses `generate_mock_export_sales()` - violates NO FAKE DATA rule

**Fix:** Implement real USDA FAS API: `https://apps.fas.usda.gov/esrquery/`

**Creates:** `trigger/ingestion/trade_supply/usda_fas_exports.ts`

**Target:** `raw.usda_export_sales` (existing table in `usda_data.sql`)

### 1.3 CFTC COT Trigger Job

**Existing:** `src/ingestion/cftc/ingest_cot.py` (script exists)

**Missing:** No Trigger.dev job

**Creates:** `trigger/ingestion/trade_supply/cftc_cot_reports.ts`

**Target:** `raw.cftc_cot_disaggregated` (used by SQL macros in `big8_bucket_features.sql`)

### 1.4 FRED Daily Ingest Job

**Existing:** `trigger/fred_seed_harvest.ts` (discovery only)

**Missing:** Daily data ingest job

**Creates:** `trigger/ingestion/macro/fred_rates_and_spreads.ts` (or split into multiple jobs per Trigger plan)

**Target:** `raw.fred_economic` (single table used by SQL macros, not split into 3 tables)

### 1.5 Remove Mock Data from USDA/NOAA

**Files:**

- `src/ingestion/usda/ingest_wasde.py` - has `generate_mock_wasde_data()`
- `src/ingestion/noaa/ingest_weather.py` - has `generate_mock_weather_data()`

**Fix:** Implement real APIs or mark as TODO with clear warnings

### 1.6 University of Illinois Intelligence Feeds (NEW - FREE, MANDATORY)

#### 1.6.1 Farm Policy News (CRITICAL for China/Tariff)

**Creates:** `trigger/ingestion/media_ag_markets/farmpolicynews.ts`

**Source:** https://farmpolicynews.illinois.edu/

**Why Critical:** Real-time policy news directly impacting soybeans

**Categories:**

| Category | Target Bucket | Example Headlines |

|----------|---------------|-------------------|

| `trade` | China, Tariff | "China Soybean Buying Deadline Now February" |

| `ethanol` | Biofuel | RFS policy updates |

| `budget` | Fed | "$11B Bridge Farm Aid" |

| `regulations` | Tariff, Biofuel | EPA rules, trade policy |

| `immigration` | Policy risk | Farm labor impacts |

**Target:** `raw.bucket_news` (source: 'farm_policy_news')

**Frequency:** Hourly check for new articles

#### 1.6.2 farmdoc Daily (Market Intelligence)

**Creates:** `trigger/ingestion/media_ag_markets/farmdoc_daily.ts`

**Source:** https://farmdoc.illinois.edu/ + https://farmdocdaily.illinois.edu/

**Categories:**

| Category | Target Bucket | Key Authors |

|----------|---------------|-------------|

| `biofuels/rins` | Biofuel | Scott Irwin (D4 RIN pricing) |

| `agricultural-policy/trade` | China, Tariff | Carl Zulauf |

| `marketing-and-outlook/grain-outlook` | Crush | Nick Paulson |

| `finance/interest-rates` | Fed | Michael Langemeier |

| `marketing-and-outlook/weekly-outlook` | All | Multiple |

**Target:** `raw.bucket_news` (source: 'farmdoc_daily')

**Frequency:** Daily at 6 AM UTC

#### 1.6.3 ProFarmer Premium (CRITICAL - PAID)

**Existing:** `trigger/ingestion/media_ag_markets/profarmer_all_urls.ts` ✅ CREATED

**Source:** https://www.profarmer.com (authenticated access via Anchor browser automation)

**Coverage:** 22+ URLs comprehensive coverage:

- Daily Editions: First Thing Today, Ahead of the Open, After the Bell
- News Sections: Agriculture, Market, Policy, Weather
- Newsletters: Weekly Outlook
- Market Analysis: Grains, Livestock, Energy
- Commodity Reports: Soybeans, Soybean Oil, Soybean Meal, Corn, Wheat, Crude Oil
- Weather: Forecasts, Crop Conditions

**Target:** `raw.bucket_news` (source: 'profarmer_all_urls')

**Frequency:** 3x daily (6 AM, 12 PM, 6 PM UTC)

**Buckets:** Crush, China, Biofuel, Weather, Tariff

**Note:** This is the PRIMARY ProFarmer job with all URLs. Other ProFarmer jobs in root `trigger/` are deprecated.

---

## Phase 1 Summary: Critical Data Feeds Alignment

**All Phase 1 jobs now align with Trigger plan structure:**

| Job | Trigger Plan Location | Target Table | Status |

|-----|----------------------|--------------|--------|

| EPA RIN | `trigger/ingestion/energy_biofuels/epa_rin_prices.ts` | `raw.epa_rin_prices` | ⚠️ Table definition needed |

| USDA Exports | `trigger/ingestion/trade_supply/usda_fas_exports.ts` | `raw.usda_export_sales` | ✅ Table exists |

| CFTC COT | `trigger/ingestion/trade_supply/cftc_cot_reports.ts` | `raw.cftc_cot_disaggregated` | ✅ Table exists |

| FRED Daily | `trigger/ingestion/macro/fred_rates_and_spreads.ts` | `raw.fred_economic` | ✅ Table exists |

| Farm Policy News | `trigger/ingestion/media_ag_markets/farmpolicynews.ts` | `raw.bucket_news` | ✅ Table exists |

| farmdoc Daily | `trigger/ingestion/media_ag_markets/farmdoc_daily.ts` | `raw.bucket_news` | ✅ Table exists |

| ProFarmer | `trigger/ingestion/media_ag_markets/profarmer_all_urls.ts` | `raw.bucket_news` | ✅ Job exists |

**Existing:** `trigger/ingestion/media_ag_markets/profarmer_all_urls.ts` ✅ CREATED

**Source:** https://www.profarmer.com (authenticated access via Anchor browser automation)

**Coverage:** 22+ URLs comprehensive coverage:

- Daily Editions: First Thing Today, Ahead of the Open, After the Bell
- News Sections: Agriculture, Market, Policy, Weather
- Newsletters: Weekly Outlook
- Market Analysis: Grains, Livestock, Energy
- Commodity Reports: Soybeans, Soybean Oil, Soybean Meal, Corn, Wheat, Crude Oil
- Weather: Forecasts, Crop Conditions

**Target:** `raw.bucket_news` (source: 'profarmer_all_urls')

**Frequency:** 3x daily (6 AM, 12 PM, 6 PM UTC)

**Buckets:** Crush, China, Biofuel, Weather, Tariff

**Note:** This is the PRIMARY ProFarmer job with all URLs. Other ProFarmer jobs in root `trigger/` are deprecated.

---

## Big 8 Bucket Source Matrix (Updated)

| Bucket | Primary Sources | Secondary Sources |

|--------|----------------|-------------------|

| **Crush** | Databento (ZL/ZS/ZM), NOPA | farmdoc: Grain Outlook, Weekly Outlook |

| **China** | **Farm Policy News: Trade**, USDA Export Sales | farmdoc: Trade Policy, ScrapeCreators |

| **FX** | FRED FX series, Databento (6L/DX) | - |

| **Fed** | FRED rates/curve | Farm Policy News: Budget, farmdoc: Interest Rates |

| **Tariff** | **Farm Policy News: Trade**, ScrapeCreators Trump | farmdoc: Gardner Policy, Federal Register |

| **Biofuel** | **EPA RIN Prices (D3/D4/D5/D6)**, EIA | farmdoc: RINs (Scott Irwin), Farm Policy: Ethanol, Growth Energy |

| **Energy** | EIA petroleum, Databento (CL/HO/RB) | - |

| **Volatility** | FRED VIXCLS, Databento VIX | - |

---

## Phase 2: AutoGluon Integration (Week 1-2)

### 2.1 Add AutoGluon to Requirements

**File:** `config/requirements/requirements.txt`

**Add:**

```
autogluon>=1.4.0
torch>=2.0
transformers>=4.40
```

### 2.2 Mac M4 Setup Script

**Creates:** `scripts/setup/install_autogluon_mac.sh`

**Contents:** libomp fix + CPU-only AutoGluon install

### 2.3 AutoGluon Tabular Module

**Creates:** `src/training/autogluon/`

```
src/training/autogluon/
├── __init__.py
├── tabular_trainer.py      # TabularPredictor wrapper (presets='extreme_quality', quantile)
├── timeseries_trainer.py   # TimeSeriesPredictor wrapper (CPU-compatible)
├── bucket_specialist.py    # Bucket-specific training (presets='extreme_quality')
└── meta_learner.py         # Meta-learner (ensemble of 9 OOF predictions)

Note: foundation_models.py and ensemble_combiner.py NOT needed
      (AutoGluon does this automatically with WeightedEnsemble_L2)
```

### 2.4 TabularPredictor Wrapper

**File:** `src/training/autogluon/tabular_trainer.py`

**Features:**

- Quantile regression (`problem_type='quantile'`)
- `presets='extreme_quality'` for ALL specialists
   - `presets='extreme_quality'` for main predictor (all features)
   - Auto-stacking with OOF predictions (AutoGluon automatic)

### 2.5 TimeSeriesPredictor Wrapper

**File:** `src/training/autogluon/timeseries_trainer.py`

**Features:**

- Chronos-Bolt zero-shot baseline
- PerStepTabular with CatBoost
- Big 8 as `known_covariates`

### 2.6 Update Engine Registry

**File:** `src/engines/engine_registry.py`

**Add:** AutoGluon models to `MODEL_FAMILIES` dict

---

## Phase 3: Bucket Specialist Infrastructure (Week 2)

### 3.1 Bucket Feature Selector Config

**Creates:** `config/bucket_feature_selectors.yaml`

**Contents:** Feature lists for each of 8 buckets

### 3.2 Bucket Specialist Trainers

**Creates:** 8 bucket-specific training configs

**Location:** `config/training/buckets/`

```
crush.yaml
china.yaml
fx.yaml
fed.yaml
tariff.yaml
biofuel.yamof l
energy.yaml
volatility.yaml
```

### 3.3 Training Orchestrator

**Creates:** `src/training/autogluon/train_all_buckets.py`

**Features:**

- Train 8 bucket specialists with `presets='extreme_quality'`
   - Train 1 main predictor with `presets='extreme_quality'` (all features)
   - Optional: Train TimeSeriesPredictor baseline (Chronos-Bolt, CPU-compatible)
   - Save all models + OOF predictions to local DuckDB

---

## Phase 4: Ensemble + Monte Carlo (Week 2-3)

### 4.1 Ensemble Tables

**File:** `database/definitions/04_training/ensemble_tables.sql`

**Creates:**

- `training.bucket_predictions` (OOF predictions per bucket)
- `training.ensemble_weights` (learned weights)
- `training.stacking_features` (meta-features)

### 4.2 Greedy Ensemble

**File:** `src/ensemble/greedy_ensemble.py`

**Features:**

- Combine bucket specialist + main predictor + Chronos outputs
- Optimize weights via validation pinball loss
- Output final P10/P50/P90

### 4.3 Monte Carlo Integration

**File:** `src/simulators/monte_carlo_sim.py` (exists)

**Update:** Accept AutoGluon quantile outputs

- Sample from P10/P50/P90 distributions
- Generate 10,000 scenarios
- Compute VaR, CVaR, Expected Shortfall

---

## Phase 5: Trigger.dev Orchestration (Week 3)

**Note:** Phase 5 jobs align with Trigger plan structure. See `/Users/zincdigital/.cursor/plans/cbi-v15-trigger-ingestion_4a9b434d.plan.md` for complete ingestion orchestration.

### 5.1 Training Orchestration Jobs

**Creates:**

- `trigger/training/train_big8_zl_models.ts` (matches Trigger plan naming)
  - Orchestrates training of all 8 bucket specialists + main ZL predictor
  - Runs on weekly schedule after feature engineering completes
- `trigger/training/train_chronos_baseline.ts` (optional)
  - TimeSeriesPredictor baseline for comparison

### 5.2 Feature Engineering Jobs

**Note:** Feature jobs run in `trigger/features/` per Trigger plan:

- `trigger/features/build_feature_views.ts` - Builds `features.*` views from `raw.*`
- `trigger/features/materialize_feature_tables.ts` - Materializes `features.daily_ml_matrix_zl`
- `trigger/features/validate_feature_integrity.ts` - Data quality checks

### 5.3 Daily Forecast Pipeline

**Sequence:**

1. **Ingestion** (Trigger plan): All source jobs run on schedule → `raw.*` tables
2. **Feature Engineering** (Trigger plan): Feature jobs run → `features.daily_ml_matrix_zl`
3. **Sync to Local**: `scripts/sync_motherduck_to_local.py` → Local DuckDB
4. **Training**: `src/training/autogluon/train_all_buckets.py` → Reads local DuckDB
5. **Generate Forecasts**: AutoGluon predictions → MotherDuck `forecasts.zl_predictions`
6. **Monte Carlo**: Risk metrics → `forecasts.monte_carlo_scenarios`
7. **Dashboard**: Vercel queries `forecasts.zl_predictions` directly from MotherDuck

**Note:** No separate "feed" jobs needed - training reads directly from `features.daily_ml_matrix_zl` after sync.

---

## Files to Create (18 new files + 1 critical table definition)

| File | Purpose |

|------|---------|

| `trigger/ingestion/energy_biofuels/epa_rin_prices.ts` | EPA RIN price ingestion |

| `trigger/ingestion/trade_supply/usda_fas_exports.ts` | Export sales ingestion |

| `trigger/ingestion/trade_supply/cftc_cot_reports.ts` | CFTC COT ingestion |

| `trigger/ingestion/macro/fred_rates_and_spreads.ts` | FRED daily data |

| `trigger/ingestion/media_ag_markets/farmpolicynews.ts` | Farm Policy News scraper |

| `trigger/ingestion/media_ag_markets/farmdoc_daily.ts` | farmdoc Daily scraper |

| `scripts/setup/install_autogluon_mac.sh` | Mac M4 setup |

| `scripts/sync_motherduck_to_local.py` | MotherDuck → local sync |

| `src/training/autogluon/__init__.py` | Module init |

| `src/training/autogluon/tabular_trainer.py` | TabularPredictor wrapper |

| `src/training/autogluon/timeseries_trainer.py` | TimeSeriesPredictor wrapper |

| `src/training/autogluon/bucket_specialist.py` | Bucket training |

| `src/training/autogluon/foundation_models.py` | NOT NEEDED (AutoGluon handles automatically) |

| `src/training/autogluon/ensemble_combiner.py` | NOT NEEDED (AutoGluon WeightedEnsemble_L2 handles this) |

| `src/training/autogluon/train_all_buckets.py` | Training orchestrator |

| `config/bucket_feature_selectors.yaml` | Feature selectors |

| `database/definitions/04_training/ensemble_tables.sql` | Ensemble schema |

| `database/definitions/01_raw/epa_rin_prices.sql` | **CRITICAL** - EPA RIN prices table (referenced in SQL macros) |

---

## Files to Modify (8 existing files)

| File | Change |

|------|--------|

| `src/ingestion/databento/collect_daily.py` | `date` → `as_of_date` |

| `src/ingestion/fred/collect_fred_*.py` | Add DFEDTARU, VIXCLS |

| `database/macros/big8_bucket_features.sql` | Fix table references |

| `src/ingestion/usda/ingest_export_sales.py` | Remove mock data |

| `config/requirements/requirements.txt` | Add AutoGluon |

| `src/engines/engine_registry.py` | Add AutoGluon models |

| `src/simulators/monte_carlo_sim.py` | Accept AutoGluon output |

| `src/ensemble/qra_ensemble.py` | Integrate with AutoGluon |

---

## Validation Checkpoints

### After Phase 0

- [ ] Databento ingestion succeeds with `as_of_date`
- [ ] FRED series include DFEDTARU, VIXCLS
- [ ] Big 8 bucket scores compute without NULL
- [ ] Local DuckDB mirror created and syncing

### After Phase 1

- [ ] EPA RIN prices populated (not NULL)
- [ ] Export sales real data (not mock)
- [ ] CFTC COT running on schedule
- [ ] Farm Policy News scraping

### After Phase 2

- [ ] AutoGluon imports successfully on Mac M4 (no segfault)
- [ ] TabularPredictor trains with `presets='extreme_quality'` (CPU-compatible, slower without GPU)
- [ ] TimeSeriesPredictor trains with PerStepTabular (CPU-compatible)

### After Phase 3

- [ ] All 8 bucket specialists trained
- [ ] OOF predictions saved to database
- [ ] Feature importance available per bucket

### After Phase 4

- [ ] Ensemble weights optimized
- [ ] Monte Carlo generates 10K scenarios
- [ ] VaR/CVaR computed correctly

### After Phase 5

- [ ] Trigger jobs run on schedule
- [ ] Daily forecasts appear in `forecasts.zl_predictions`
- [ ] Dashboard displays P10/P50/P90

---

## Model Summary

| Component | Count | Algorithm |

|-----------|-------|-----------|

| Bucket Specialists | 8 × 10-15 | LightGBM, CatBoost, XGBoost, FastAI, PyTorch NN, Random Forest, Extra Trees |

| Main ZL Predictor | 10-15 | Same (all models, all features) |

| Chronos Baseline | 2 | Chronos-Bolt, PerStepTabular |

| L1 Stack | ~10 | Meta-learners |

| L2 Ensemble | 1 | Greedy weighted |

| **Total** | **~70** | |

---

## Timeline

| Phase | Duration | Deliverable |

|-------|----------|-------------|

| Phase 0 | Days 1-2 | Bug fixes complete, data flowing |

| Phase 1 | Days 3-7 | Critical data feeds active |

| Phase 2 | Days 8-14 | AutoGluon integrated |

| Phase 3 | Days 15-21 | Bucket specialists trained |

| Phase 4 | Days 22-28 | Ensemble + Monte Carlo working |

| Phase 5 | Days 29-35 | Trigger.dev orchestration live |