# AutoGluon Hybrid Implementation Plan Review

**Date:** December 10, 2025  
**Plan File:** `.cursor/plans/autogluon_hybrid_implementation_c2287cb0.plan.md`  
**Trigger Plan:** `/Users/zincdigital/.cursor/plans/cbi-v15-trigger-ingestion_4a9b434d.plan.md`  
**Status:** ⚠️ **ISSUES IDENTIFIED** - Needs updates to match current work

---

## Executive Summary

The AutoGluon plan is **architecturally sound** but has **critical mismatches** with:
1. Actual database table names
2. Trigger plan structure (domain-based directories)
3. ProFarmer job location
4. Preset references (inconsistent "BEST" vs `extreme_quality`)

**Overall Alignment Score**: **6.5/10** - Needs updates before execution

---

## 1. Table Name Mismatches (CRITICAL)

### 1.1 Phase 1.1: EPA RIN Prices

| Plan Says | Actual Database | Status | Fix Required |
|-----------|----------------|--------|--------------|
| `raw.eia_petroleum` (series_id: `rin_d3_price`) | ❌ **WRONG TARGET** | ❌ **MISMATCH** | Should be `raw.epa_rin_prices` |

**Issue**: Plan says to write EPA RIN prices to `raw.eia_petroleum` with series_id, but:
- SQL macros reference `raw.epa_rin_prices` (see `database/macros/anofox_guards.sql` line 94)
- Trigger plan specifies `raw.epa_rin_prices` (Section 5.5)
- Table definition doesn't exist yet (CRITICAL - must create)

**Fix**: 
- Update plan: `Target: raw.epa_rin_prices` (not `raw.eia_petroleum`)
- Create table definition: `database/models/01_raw/epa_rin_prices.sql`
- Update Trigger job to write to `raw.epa_rin_prices`

### 1.2 Phase 1.2: USDA Export Sales

| Plan Says | Actual Database | Status | Fix Required |
|-----------|----------------|--------|--------------|
| Creates `trigger/usda_export_sales.ts` | ✅ **CORRECT** | ✅ **OK** | No change |
| Target table not specified | `raw.usda_export_sales` (in `usda_data.sql`) | ⚠️ **MISSING** | Add target table to plan |

**Issue**: Plan doesn't specify target table. Trigger plan says `raw.usda_fas_exports` but actual table is `raw.usda_export_sales`.

**Fix**: Update plan to specify `Target: raw.usda_export_sales` (match existing table)

### 1.3 Phase 1.3: CFTC COT

| Plan Says | Actual Database | Status | Fix Required |
|-----------|----------------|--------|--------------|
| Creates `trigger/cftc_cot_ingest.ts` | ✅ **CORRECT** | ✅ **OK** | No change |
| Target table not specified | `raw.cftc_cot_disaggregated` + `raw.cftc_cot_tff` | ⚠️ **MISSING** | Add target table to plan |

**Issue**: Plan doesn't specify which CFTC table. SQL macros use `raw.cftc_cot_disaggregated`.

**Fix**: Update plan to specify `Target: raw.cftc_cot_disaggregated` (match SQL macros)

### 1.4 Phase 1.4: FRED Daily

| Plan Says | Actual Database | Status | Fix Required |
|-----------|----------------|--------|--------------|
| Creates `trigger/fred_daily_ingest.ts` | ✅ **CORRECT** | ✅ **OK** | No change |
| Target table not specified | `raw.fred_economic` (single table) | ⚠️ **MISSING** | Add target table to plan |

**Issue**: Plan doesn't specify target table. Trigger plan splits into 3 tables (`raw.fred_rates_spreads`, `raw.fred_financial_conditions`, `raw.fred_real_economy`) but actual database has single `raw.fred_economic` table.

**Fix**: Update plan to specify `Target: raw.fred_economic` (match existing table and SQL macros)

---

## 2. Trigger Job Structure Mismatches (HIGH)

### 2.1 Phase 1 Jobs - Wrong Directory Structure

| Plan Says | Trigger Plan Says | Status | Fix Required |
|-----------|-------------------|--------|--------------|
| `trigger/epa_rin_prices.ts` | `trigger/ingestion/energy_biofuels/epa_rin_prices.ts` | ❌ **MISMATCH** | Move to domain subdirectory |
| `trigger/usda_export_sales.ts` | `trigger/ingestion/trade_supply/usda_fas_exports.ts` | ❌ **MISMATCH** | Move + rename to match plan |
| `trigger/cftc_cot_ingest.ts` | `trigger/ingestion/trade_supply/cftc_cot_reports.ts` | ❌ **MISMATCH** | Move + rename to match plan |
| `trigger/fred_daily_ingest.ts` | `trigger/ingestion/macro/fred_rates_and_spreads.ts` (and others) | ❌ **MISMATCH** | Move to domain subdirectory |
| `trigger/farm_policy_news.ts` | `trigger/ingestion/media_ag_markets/farmpolicynews.ts` | ❌ **MISMATCH** | Move + rename to match plan |
| `trigger/farmdoc_daily.ts` | `trigger/ingestion/media_ag_markets/farmdoc_daily.ts` | ⚠️ **PARTIAL** | Move to domain subdirectory |

**Issue**: AutoGluon plan uses flat `trigger/` structure, but Trigger plan specifies domain-based subdirectories (`trigger/ingestion/{domain}/`).

**Fix**: Update all Phase 1 job paths to match Trigger plan structure:
- `trigger/epa_rin_prices.ts` → `trigger/ingestion/energy_biofuels/epa_rin_prices.ts`
- `trigger/usda_export_sales.ts` → `trigger/ingestion/trade_supply/usda_fas_exports.ts`
- `trigger/cftc_cot_ingest.ts` → `trigger/ingestion/trade_supply/cftc_cot_reports.ts`
- `trigger/fred_daily_ingest.ts` → `trigger/ingestion/macro/fred_rates_and_spreads.ts` (or split into multiple jobs)
- `trigger/farm_policy_news.ts` → `trigger/ingestion/media_ag_markets/farmpolicynews.ts`
- `trigger/farmdoc_daily.ts` → `trigger/ingestion/media_ag_markets/farmdoc_daily.ts`

### 2.2 Phase 5 Jobs - Wrong Directory Structure

| Plan Says | Trigger Plan Says | Status | Fix Required |
|-----------|-------------------|--------|--------------|
| `trigger/feeds/feed_main_model.ts` | Not in Trigger plan | ⚠️ **ORPHAN** | Clarify if this is needed |
| `trigger/feeds/feed_{bucket}_specialist.ts` | Not in Trigger plan | ⚠️ **ORPHAN** | Clarify if this is needed |
| `trigger/training/train_bucket_specialists.ts` | `trigger/training/train_big8_zl_models.ts` | ⚠️ **NAMING** | Align naming |
| `trigger/forecasts/daily_zl_forecast.ts` | Not in Trigger plan | ⚠️ **ORPHAN** | Clarify if this is needed |

**Issue**: Phase 5 jobs don't align with Trigger plan structure. Trigger plan has `trigger/training/` but different job names.

**Fix**: Align Phase 5 jobs with Trigger plan or clarify if these are additional jobs.

---

## 3. ProFarmer Job Reference Missing

### 3.1 Phase 1 - No ProFarmer Mention

**Issue**: AutoGluon plan doesn't mention ProFarmer, but:
- Trigger plan has `trigger/ProFarmer/Scripts/profarmer_all_urls.ts` (PRIMARY job)
- Job exists: `trigger/ProFarmer/Scripts/profarmer_all_urls.ts`
- Writes to: `raw.bucket_news` (correct table)
- Critical source for Crush, China, Biofuel buckets

**Fix**: Add ProFarmer to Phase 1:
```markdown
### 1.7 ProFarmer Premium (CRITICAL - PAID)

**Existing:** `trigger/ProFarmer/Scripts/profarmer_all_urls.ts` ✅ CREATED

**Source:** https://www.profarmer.com (authenticated access via Anchor)

**Coverage:** 22+ URLs (daily editions, news, newsletters, analysis, commodities, weather)

**Target:** `raw.bucket_news` (source: 'profarmer_all_urls')

**Frequency:** 3x daily (6 AM, 12 PM, 6 PM UTC)

**Buckets:** Crush, China, Biofuel, Weather, Tariff
```

---

## 4. Preset Reference Inconsistencies (MEDIUM)

### 4.1 Inconsistent Preset References

| Location | Says | Should Be | Status |
|----------|------|-----------|--------|
| Line 12 | `presets='extreme_quality'` | ✅ **CORRECT** | No change |
| Line 113 | `preset="BEST"` | ❌ **WRONG** | Should be `presets='extreme_quality'` |
| Line 123 | `presets='extreme_quality'` | ✅ **CORRECT** | No change |
| Line 288 | `"BEST" preset` | ❌ **WRONG** | Should be `presets='extreme_quality'` |
| Line 507 | `preset="BEST"` | ❌ **WRONG** | Should be `presets='extreme_quality'` |
| Line 509 | `preset="BEST"` | ❌ **WRONG** | Should be `presets='extreme_quality'` |
| Line 523-524 | `preset="extreme_quality"` | ⚠️ **QUOTES** | Should be `presets='extreme_quality'` (single quotes) |
| Line 576-577 | `presets='extreme_quality'` | ✅ **CORRECT** | No change |
| Line 739 | `preset="BEST"` | ❌ **WRONG** | Should be `presets='extreme_quality'` |

**Issue**: Plan has inconsistent preset references. Some say "BEST", some say `extreme_quality`, some use double quotes instead of single quotes.

**Fix**: Standardize ALL references to `presets='extreme_quality'` (single quotes, plural `presets`)

---

## 5. File Path Verification

### 5.1 Phase 0 Files - All Exist ✅

| File | Status | Verified |
|------|--------|----------|
| `scripts/sync_motherduck_to_local.py` | ✅ **EXISTS** | Confirmed |
| `database/models/03_features/core_macro_fx.sql` | ✅ **EXISTS** | Confirmed |
| `scripts/validation/verify_core_macro_fx.py` | ⚠️ **NEED TO CHECK** | Verify exists |
| `src/reporting/training_auditor.py` | ✅ **EXISTS** | Confirmed |
| `database/macros/anofox_guards.sql` | ✅ **EXISTS** | Confirmed |
| `config/bucket_feature_selectors.yaml` | ✅ **EXISTS** | Confirmed |

### 5.2 Phase 1 Files - Need Creation

| File | Trigger Plan Location | Status |
|------|----------------------|--------|
| `trigger/epa_rin_prices.ts` | `trigger/ingestion/energy_biofuels/epa_rin_prices.ts` | ❌ **WRONG PATH** |
| `trigger/usda_export_sales.ts` | `trigger/ingestion/trade_supply/usda_fas_exports.ts` | ❌ **WRONG PATH** |
| `trigger/cftc_cot_ingest.ts` | `trigger/ingestion/trade_supply/cftc_cot_reports.ts` | ❌ **WRONG PATH** |
| `trigger/fred_daily_ingest.ts` | `trigger/ingestion/macro/fred_rates_and_spreads.ts` | ❌ **WRONG PATH** |
| `trigger/farm_policy_news.ts` | `trigger/ingestion/media_ag_markets/farmpolicynews.ts` | ❌ **WRONG PATH** |
| `trigger/farmdoc_daily.ts` | `trigger/ingestion/media_ag_markets/farmdoc_daily.ts` | ❌ **WRONG PATH** |

---

## 6. Alignment with MASTER_PLAN.md

### 6.1 Architecture Alignment ✅

| Aspect | AutoGluon Plan | MASTER_PLAN.md | Status |
|--------|----------------|----------------|--------|
| Preset | `extreme_quality` (mostly) | `extreme_quality` | ✅ **ALIGNED** |
| Model Stack | 4 Layers (L0-L3) | 4 Layers (L0-L3) | ✅ **ALIGNED** |
| Training Location | Mac M4 local | Mac M4 local | ✅ **ALIGNED** |
| Storage | MotherDuck | MotherDuck | ✅ **ALIGNED** |
| Feature Engineering | SQL macros | SQL macros | ✅ **ALIGNED** |

### 6.2 Data Sources Alignment ⚠️

| Source | AutoGluon Plan | MASTER_PLAN.md | Status |
|--------|----------------|----------------|--------|
| ProFarmer | ❌ **MISSING** | ✅ Listed (line 100) | ❌ **MISMATCH** |
| Farm Policy News | ✅ Listed | ✅ Listed | ✅ **ALIGNED** |
| farmdoc Daily | ✅ Listed | ✅ Listed | ✅ **ALIGNED** |
| EPA RIN Prices | ✅ Listed | ✅ Listed | ✅ **ALIGNED** |

---

## 7. Alignment with Trigger Plan

### 7.1 Directory Structure ❌

**AutoGluon Plan**: Flat `trigger/` structure  
**Trigger Plan**: Domain-based `trigger/ingestion/{domain}/` structure

**Fix**: Update all Phase 1 job paths to match Trigger plan

### 7.2 Job Naming ⚠️

**AutoGluon Plan**: `usda_export_sales.ts`, `cftc_cot_ingest.ts`, `fred_daily_ingest.ts`  
**Trigger Plan**: `usda_fas_exports.ts`, `cftc_cot_reports.ts`, `fred_rates_and_spreads.ts`

**Fix**: Align job names with Trigger plan OR document why different names are used

### 7.3 ProFarmer Job ✅

**AutoGluon Plan**: Not mentioned  
**Trigger Plan**: `profarmer_all_urls.ts` (PRIMARY, has all 22+ URLs)  
**Actual**: `trigger/ProFarmer/Scripts/profarmer_all_urls.ts` exists

**Fix**: Add ProFarmer to Phase 1 (see Section 3.1)

---

## 8. Critical Fixes Required

### 8.1 Immediate (Before Execution)

1. **🔴 CRITICAL**: Fix EPA RIN target table
   - Change: `raw.eia_petroleum` → `raw.epa_rin_prices`
   - Create: `database/models/01_raw/epa_rin_prices.sql`

2. **🔴 CRITICAL**: Update all Phase 1 Trigger job paths
   - Move from `trigger/` to `trigger/ingestion/{domain}/`
   - Align job names with Trigger plan

3. **🟡 HIGH**: Add ProFarmer to Phase 1
   - Reference existing `trigger/ProFarmer/Scripts/profarmer_all_urls.ts`

4. **🟡 HIGH**: Fix preset references
   - Change all "BEST" → `presets='extreme_quality'`
   - Use single quotes consistently

5. **🟡 HIGH**: Add target tables to Phase 1 jobs
   - Specify `raw.usda_export_sales` for USDA
   - Specify `raw.cftc_cot_disaggregated` for CFTC
   - Specify `raw.fred_economic` for FRED

### 8.2 Medium Priority

1. Align Phase 5 jobs with Trigger plan structure
2. Verify all file paths exist
3. Add validation checkpoints for table creation

---

## 9. Recommendations

### 9.1 Update Plan Structure

**Current**: Flat `trigger/` structure  
**Should Be**: Domain-based `trigger/ingestion/{domain}/` structure

**Action**: Update all Phase 1 job paths to match Trigger plan

### 9.2 Add Missing Sources

**Missing**: ProFarmer (critical source, job already exists)  
**Action**: Add to Phase 1 as section 1.7

### 9.3 Standardize Preset References

**Current**: Mixed "BEST" and `extreme_quality`  
**Should Be**: All `presets='extreme_quality'` (single quotes, plural)

**Action**: Find/replace all "BEST" and `preset="extreme_quality"` → `presets='extreme_quality'`

### 9.4 Add Table Specifications

**Current**: Many Phase 1 jobs don't specify target tables  
**Should Be**: Every job specifies exact target table name

**Action**: Add `Target: raw.{table_name}` to each Phase 1 job

---

## 10. Summary of Required Changes

### Files to Update

1. `.cursor/plans/autogluon_hybrid_implementation_c2287cb0.plan.md`
   - Fix Phase 1 job paths (6 jobs)
   - Fix preset references (5 locations)
   - Add ProFarmer section
   - Add target tables to Phase 1 jobs
   - Fix EPA RIN target table

### Files to Create

1. `database/models/01_raw/epa_rin_prices.sql` (CRITICAL - referenced in SQL macros)

---

## 11. Alignment Scorecard

| Category | Score | Status |
|----------|-------|--------|
| **Architecture** | 9/10 | ✅ Excellent - matches MASTER_PLAN.md |
| **Table Names** | 4/10 | ❌ Poor - multiple mismatches |
| **Trigger Structure** | 3/10 | ❌ Poor - wrong directory structure |
| **Preset References** | 6/10 | ⚠️ Needs work - inconsistent |
| **File Paths** | 7/10 | ⚠️ Good - but wrong structure |
| **Data Sources** | 7/10 | ⚠️ Good - missing ProFarmer |

**Overall Score**: **6.5/10** - Needs updates before execution

---

**Last Updated**: December 10, 2025  
**Reviewer**: AI Assistant (Auto)  
**Status**: ⚠️ **ISSUES IDENTIFIED** - Update plan before execution







