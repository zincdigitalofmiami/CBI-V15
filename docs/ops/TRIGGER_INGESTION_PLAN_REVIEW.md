# CBI-V15 Trigger Ingestion Plan Review

**Date:** December 10, 2025  
**Plan File:** `/Users/zincdigital/.cursor/plans/cbi-v15-trigger-ingestion_4a9b434d.plan.md`  
**Status:** ⚠️ **CRITICAL ISSUES IDENTIFIED** - Must fix before execution

---

## Executive Summary

The plan is **architecturally sound** but has **critical naming mismatches** and **missing table definitions** that will cause execution failures. This review identifies:

1. **🔴 CRITICAL**: 40+ table name mismatches between plan and actual database
2. **🟡 HIGH**: Existing Trigger jobs conflict with proposed structure
3. **🟡 HIGH**: Missing table definitions for 30+ sources
4. **🟢 MEDIUM**: Naming convention inconsistencies
5. **🟢 MEDIUM**: Lineage validation needed

---

## 1. Schema & Table Name Mismatches (CRITICAL)

### 1.1 Market Data

| Plan Table Name | Actual Database Table | Status | Fix Required |
|----------------|----------------------|--------|--------------|
| `raw.databento_futures` | `raw.databento_ohlcv_daily` | ❌ **MISMATCH** | Plan uses wrong name |
| `raw.databento_futures` | `raw.databento_futures` (exists) | ⚠️ **CONFLICT** | Two different tables? |

**Issue**: Plan references `raw.databento_futures` but actual table is `raw.databento_ohlcv_daily`. Existing code uses `raw.databento_ohlcv_daily` throughout SQL macros.

**Fix**: Update plan to use `raw.databento_ohlcv_daily` OR create `raw.databento_futures` if it's meant to be different (intraday vs daily).

### 1.2 Macro Data (FRED)

| Plan Table Name | Actual Database Table | Status | Fix Required |
|----------------|----------------------|--------|--------------|
| `raw.fred_rates_spreads` | `raw.fred_economic` | ❌ **MISMATCH** | Plan table doesn't exist |
| `raw.fred_financial_conditions` | `raw.fred_economic` | ❌ **MISMATCH** | Plan table doesn't exist |
| `raw.fred_real_economy` | `raw.fred_economic` | ❌ **MISMATCH** | Plan table doesn't exist |
| `raw.fred_observations` | `raw.fred_observations` | ✅ **EXISTS** | No change needed |
| `raw.fred_series_metadata` | `raw.fred_series_metadata` | ✅ **EXISTS** | No change needed |

**Issue**: Plan splits FRED into 3 separate tables, but actual database has a single `raw.fred_economic` table. SQL macros reference `raw.fred_economic`.

**Fix**: Either:
- **Option A**: Update plan to use `raw.fred_economic` (simpler, matches existing code)
- **Option B**: Create separate tables and update all SQL macros (major refactor)

**Recommendation**: **Option A** - Use `raw.fred_economic` and add a `category` column if needed.

### 1.3 Trade & Supply

| Plan Table Name | Actual Database Table | Status | Fix Required |
|----------------|----------------------|--------|--------------|
| `raw.usda_fas_exports` | `raw.usda_export_sales` (in `usda_data.sql`) | ❌ **MISMATCH** | Plan uses wrong name |
| `raw.usda_gain` | ❌ **NOT DEFINED** | ❌ **MISSING** | Create table definition |
| `raw.usda_nass` | ❌ **NOT DEFINED** | ❌ **MISSING** | Create table definition |
| `raw.usda_open_data` | ❌ **NOT DEFINED** | ❌ **MISSING** | Create table definition |
| `raw.usda_wasde_pdf` | `raw.usda_wasde` (in `usda_data.sql`) | ❌ **MISMATCH** | Plan uses wrong name |
| `raw.conab_crops` | ❌ **NOT DEFINED** | ❌ **MISSING** | Create table definition |
| `raw.abiove_crush` | ❌ **NOT DEFINED** | ❌ **MISSING** | Create table definition |
| `raw.cftc_cot` | `raw.cftc_cot_disaggregated` + `raw.cftc_cot_tff` | ⚠️ **PARTIAL** | Plan needs to specify which table |

**Issue**: Plan uses different naming than actual database. Existing code uses `raw.usda_export_sales` and `raw.usda_wasde`.

**Fix**: Update plan to match existing table names OR create new tables and update all SQL macros.

### 1.4 Energy & Biofuels

| Plan Table Name | Actual Database Table | Status | Fix Required |
|----------------|----------------------|--------|--------------|
| `raw.eia_core` | ❌ **DEPRECATED** | ❌ **MISMATCH** | Update plan to use `raw.eia_biofuels` for EIA biodiesel/biofuel data only |
| `raw.eia_bulk` | ❌ **NOT DEFINED** | ❌ **MISSING** | Create table definition (if still needed) or deprecate from plan |
| `raw.eia_biodiesel` | `raw.eia_biofuels` | ❌ **MISMATCH** | Plan uses wrong name; standardize on `raw.eia_biofuels` |
| `raw.epa_rin_prices` | `raw.epa_rin_prices` | ✅ **MATCH** | None - table definition now exists and is referenced by SQL macros |
| `raw.epa_main_pages` | ❌ **NOT DEFINED** | ❌ **MISSING** | Create table definition |

**Issue**: `raw.epa_rin_prices` is **CRITICAL** (referenced in `database/definitions/01_raw/epa_rin_prices.sql` and `database/macros/big8_bucket_features.sql`) and now has a table definition. Ensure ingestion jobs populate it.

**Fix**: Implement EPA RIN ingestion to write into `raw.epa_rin_prices`.

### 1.5 Weather & Climate

| Plan Table Name | Actual Database Table | Status | Fix Required |
|----------------|----------------------|--------|--------------|
| `raw.inmet_stations` | ❌ **NOT DEFINED** | ❌ **MISSING** | Create table definition |
| `raw.inmet_daily` | ❌ **NOT DEFINED** | ❌ **MISSING** | Create table definition |
| `raw.inmet_historical` | ❌ **NOT DEFINED** | ❌ **MISSING** | Create table definition |
| `raw.smn_weather` | ❌ **NOT DEFINED** | ❌ **MISSING** | Create table definition |
| `raw.noaa_cdo_daily` | `raw.noaa_weather_daily` (in `noaa_weather.sql`) | ❌ **MISMATCH** | Plan uses wrong name |
| `raw.noaa_cdo_hist` | ❌ **NOT DEFINED** | ❌ **MISSING** | Create table definition |
| `raw.noaa_gfs_nomads` | ❌ **NOT DEFINED** | ❌ **MISSING** | Create table definition |
| `raw.copernicus_cds` | ❌ **NOT DEFINED** | ❌ **MISSING** | Create table definition |
| `raw.meteomatics` | ❌ **NOT DEFINED** | ❌ **MISSING** | Create table definition |

**Issue**: Most weather tables don't exist. Only `raw.noaa_weather_daily` exists (not `raw.noaa_cdo_daily`).

### 1.6 Policy & Media (ALL MISSING)

**All policy and media tables are missing:**
- `raw.federal_register`
- `raw.ice_news`
- `raw.dhs_news`
- `raw.cbp_news`
- `raw.fdacs_news`
- `raw.texas_ag_news`
- `raw.piie_trade_war`
- `raw.csis_trade_monitor`
- `raw.us_china_council`
- `raw.heritage_ag`
- `raw.afpi`
- `raw.tax_foundation_trade`
- `raw.aei_trade_policy`
- `raw.aic_immigration`
- `raw.mpi`
- `raw.splc_immigrant_justice`
- `raw.floc`
- `raw.ufw`
- `raw.wga`
- `raw.fb_news`
- `raw.cfbf_news`
- `raw.gfb_news`
- `raw.cleanfuels`
- `raw.oil_world`
- `raw.nopa_crush_pdf`
- `raw.agricensus`
- `raw.reuters_commodities`
- `raw.dtn_pf`
- `raw.soybeans_corn_advisor`
- `raw.agweb_soybeans`
- `raw.farmprogress_soybeans`
- `raw.agriculture_markets`
- `raw.agrimoney_grains`
- `raw.agrimoney_china`
- `raw.world_grain`
- `raw.farmpolicynews`
- `raw.farmdoc_daily`
- `raw.profarmer`
- `raw.truthsocial_trump`
- `raw.analyst_karen_braun`
- `raw.analyst_arlan_suderman`
- `raw.analyst_scott_irwin`
- `raw.analyst_cordonnier`
- `raw.analyst_javier_blas`
- `raw.vegas_eater`
- `raw.glide_vegas`

**Issue**: **40+ tables are missing**. These need to be created before jobs can write to them.

---

## 2. Existing Trigger Jobs vs Plan Structure

### 2.1 Existing Jobs (in `trigger/` root)

| Existing Job | Plan Location | Conflict | Fix Required |
|-------------|---------------|----------|--------------|
| `databento_ingest_job.ts` | `trigger/ingestion/market_data/databento_futures.ts` | ⚠️ **NAMING** | Move to plan structure OR update plan |
| `fred_seed_harvest.ts` | `trigger/ingestion/macro/fred_rates_and_spreads.ts` | ⚠️ **NAMING** | Move to plan structure OR update plan |
| `eia_procurement_ingest.ts` | `trigger/ingestion/energy_biofuels/eia_core_api.ts` | ⚠️ **NAMING** | Move to plan structure OR update plan |
| `profarmer_ingest_job.ts` | `trigger/ingestion/media_ag_markets/profarmer_premium.ts` | ⚠️ **NAMING** | Move to plan structure OR update plan |
| `profarmer_anchor_scraper.ts` | `trigger/ingestion/media_ag_markets/profarmer_premium.ts` | ⚠️ **DUPLICATE** | Consolidate into one job |
| `profarmer_all_urls.ts` | `trigger/ingestion/media_ag_markets/profarmer_premium.ts` | ⚠️ **DUPLICATE** | Consolidate into one job |
| `vegas_intel_job.ts` | `trigger/ingestion/vegas_intel/vegas_eater.ts` | ⚠️ **NAMING** | Move to plan structure OR update plan |
| `tradingeconomics_goldmine.ts` | `trigger/ingestion/macro/tradingeconomics_macro.ts` | ⚠️ **NAMING** | Move to plan structure OR update plan |
| `news_to_signals_openai_agent.ts` | `trigger/ingestion/news_pipeline/news_to_signals_openai_agent.ts` | ✅ **MATCHES** | No change needed |
| `intelligent_news_pipeline.ts` | `trigger/ingestion/news_pipeline/` | ⚠️ **OVERLAP** | Clarify relationship with `news_to_signals_openai_agent.ts` |
| `multi_source_etl.ts` | ❌ **NOT IN PLAN** | ⚠️ **ORPHAN** | Either add to plan or remove |

**Issue**: Existing jobs are in root `trigger/` directory, but plan specifies domain-based subdirectories. Need to either:
- **Option A**: Move existing jobs to plan structure
- **Option B**: Update plan to match existing structure

**Recommendation**: **Option A** - Move to plan structure for consistency.

---

## 3. Naming Convention Issues

### 3.1 Table Naming

**Plan Pattern**: `raw.{source}_{subgroup}` (e.g., `raw.fred_rates_spreads`)  
**Actual Pattern**: `raw.{source}_{category}` (e.g., `raw.fred_economic`)

**Issue**: Inconsistent naming. Plan uses descriptive suffixes, actual uses generic categories.

**Fix**: Standardize on one pattern. Recommend using **source prefix** pattern from `AGENTS.md`:
- ✅ `raw.databento_ohlcv_daily` (source: `databento_`, type: `ohlcv_daily`)
- ✅ `raw.fred_economic` (source: `fred_`, type: `economic`)
- ✅ `raw.eia_petroleum` (source: `eia_`, type: `petroleum`)

### 3.2 Job Naming

**Plan Pattern**: `{source}_{category}.ts` (e.g., `fred_rates_and_spreads.ts`)  
**Existing Pattern**: `{source}_{action}_job.ts` (e.g., `databento_ingest_job.ts`)

**Issue**: Inconsistent job naming.

**Fix**: Standardize on plan pattern (shorter, cleaner).

---

## 4. Data Lineage Validation

### 4.1 Ingestion → Raw

✅ **CORRECT**: Jobs write to `raw.*` tables  
✅ **CORRECT**: "Ingest Once, Use Everywhere" principle  
⚠️ **ISSUE**: Many `raw.*` tables don't exist yet (see Section 1)

### 4.2 Raw → Staging

**Plan**: Feature jobs read from `raw.*` and write to `staging.*`  
**Actual**: SQL macros reference `raw.*` directly (no staging layer for some sources)

**Issue**: Plan assumes staging layer, but some features are built directly from `raw.*`.

**Fix**: Clarify which sources need staging vs. direct feature engineering.

### 4.3 Staging → Features

**Plan**: Feature jobs build `features.*` from `staging.*`  
**Actual**: SQL macros build `features.*` from `raw.*` and `staging.*` (mixed)

**Issue**: Lineage is correct but inconsistent (some go raw→features, others raw→staging→features).

### 4.4 Features → Training

✅ **CORRECT**: Training reads from `features.daily_ml_matrix_zl`  
✅ **CORRECT**: AutoGluon reads from local DuckDB (synced from MotherDuck)

### 4.5 Training → Forecasts

✅ **CORRECT**: Models write to `forecasts.zl_predictions`  
✅ **CORRECT**: Dashboard reads from `forecasts.*`

---

## 5. Missing Dependencies

### 5.1 Critical Missing Tables (Block Execution)

1. **`raw.epa_rin_prices`** - **CRITICAL** - Referenced in SQL macros
2. **`raw.fred_rates_spreads`** OR use existing `raw.fred_economic`
3. **`raw.usda_fas_exports`** OR use existing `raw.usda_export_sales`
4. **`raw.usda_wasde_pdf`** OR use existing `raw.usda_wasde`

### 5.2 Missing Adapters

Plan specifies adapters in `trigger/ingestion/adapters/`:
- ✅ `api_client.ts` - Need to create
- ✅ `scrapecreators.ts` - Need to create (or use existing `src/ingestion/scrapecreators/`)
- ✅ `browser_anchor.ts` - Need to create (or use existing `src/ingestion/buckets/news/profarmer_anchor.py`)
- ✅ `pdf_ingest.ts` - Need to create
- ✅ `html_normalizer.ts` - Need to create

**Issue**: Adapters don't exist yet. Jobs will fail without them.

### 5.3 Missing Types

Plan specifies `trigger/types/jobMeta.ts`:
- ❌ **NOT CREATED** - Need to create this file with `JobMeta` interface

---

## 6. Big 8 Bucket Alignment

### 6.1 Bucket Names

**Plan**: Matches `AGENTS.md` exactly ✅
- `Crush`, `China`, `FX`, `Fed`, `Tariff`, `Biofuel`, `Energy`, `Volatility`

**Overlay Tags**: Also matches ✅
- `Weather`, `FlowsSentiment`, `Policy_Gov`, `Policy_Influence`, `Vegas`

### 6.2 Bucket Routing

**Plan**: Jobs tagged with buckets → feed to Big 8 specialists  
**Actual**: SQL macros build bucket scores in `features.big8_bucket_scores`

**Issue**: Plan assumes job-level routing, but actual implementation uses SQL-based bucket scoring.

**Fix**: Clarify that job tags are for **observability/dashboard routing**, not direct model input. Models still see all features via `features.daily_ml_matrix_zl`.

---

## 7. Recommendations

### 7.1 Immediate Actions (Before Execution)

1. **🔴 CRITICAL**: Create missing table definitions for:
   - `raw.epa_rin_prices` (referenced in SQL macros)
   - All 40+ missing policy/media tables OR consolidate into fewer tables

2. **🔴 CRITICAL**: Fix table name mismatches:
   - Update plan to use `raw.databento_ohlcv_daily` (not `raw.databento_futures`)
   - Update plan to use `raw.fred_economic` (not split into 3 tables)
   - Update plan to use `raw.usda_export_sales` and `raw.usda_wasde` (match existing)

3. **🟡 HIGH**: Create `trigger/types/jobMeta.ts` with `JobMeta` interface

4. **🟡 HIGH**: Create adapter files in `trigger/ingestion/adapters/`:
   - `api_client.ts`
   - `scrapecreators.ts`
   - `browser_anchor.ts`
   - `pdf_ingest.ts`
   - `html_normalizer.ts`

5. **🟡 HIGH**: Decide on existing Trigger jobs:
   - Move to plan structure OR update plan to match existing
   - **✅ RESOLVED**: Keep `profarmer_all_urls.ts` as primary (has all 22+ URLs), deprecate others

### 7.2 Medium Priority

1. Standardize table naming (use source prefix pattern)
2. Standardize job naming (use plan pattern)
3. Clarify staging layer usage (which sources need it)

### 7.3 Low Priority

1. Add lineage documentation
2. Add observability job implementations
3. Add feature job implementations

---

## 8. Execution Readiness Score

| Category | Score | Status |
|----------|-------|--------|
| **Schema Alignment** | 2/10 | ❌ **BLOCKED** - 40+ table mismatches |
| **Naming Conventions** | 6/10 | ⚠️ **NEEDS WORK** - Inconsistent patterns |
| **Data Lineage** | 8/10 | ✅ **GOOD** - Mostly correct |
| **Dependencies** | 3/10 | ❌ **BLOCKED** - Missing adapters/types |
| **Existing Code Alignment** | 5/10 | ⚠️ **NEEDS WORK** - Job structure conflicts |

**Overall Readiness**: **4.8/10** - **NOT READY FOR EXECUTION**

**Must Fix**: Schema mismatches and missing tables before any jobs can run.

---

## 9. Next Steps

1. **Review this document** with team
2. **Decide on table naming strategy** (match existing vs. create new)
3. **Create missing table definitions** (prioritize `raw.epa_rin_prices`)
4. **Fix plan table names** to match actual database
5. **Create adapter files** and `jobMeta.ts`
6. **Resolve existing Trigger job conflicts** (move or update plan)
7. **Re-review** after fixes

---

**Last Updated**: December 10, 2025  
**Reviewer**: AI Assistant (Auto)  
**Status**: ⚠️ **CRITICAL ISSUES IDENTIFIED** - Execution blocked until fixes

---

## 10. Updates (December 10, 2025)

### 10.1 ProFarmer Job Resolution ✅

**Decision**: Keep `profarmer_all_urls.ts` as the **PRIMARY** ProFarmer ingestion job.

**Rationale**:
- Has comprehensive URL coverage (22+ URLs)
- Includes all critical sections: daily editions, news, newsletters, analysis, commodities, weather
- Already writes to correct table: `raw.bucket_news`
- Well-structured with priority filtering

**Actions Taken**:
1. ✅ Created `trigger/ProFarmer/Scripts/profarmer_all_urls.ts` (moved from root)
2. ✅ Updated plan to reference `profarmer_all_urls.ts` instead of `profarmer_premium.ts`
3. ✅ Updated plan table name: `raw.bucket_news` (not `raw.profarmer`)
4. ✅ Marked `profarmer_ingest_job.ts` and `profarmer_anchor_scraper.ts` as **DEPRECATED**

**Next Steps**:
- Deprecate old ProFarmer jobs (keep for reference, don't use)
- Update Trigger.dev index to register new job location

### 10.2 Work List Reference

**Note**: User has a comprehensive work list that may cover most of the issues identified in this review. The work list should be consulted before creating missing table definitions and adapters.

**Recommendation**: Review work list first, then prioritize fixes based on what's already planned vs. what needs immediate attention.


