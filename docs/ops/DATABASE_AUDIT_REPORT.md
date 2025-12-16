# CBI-V15 Database Audit Report

**Audit Date:** December 15, 2025
**Auditor:** Claude Code Assistant
**Database Version:** V15.1
**Audit Script:** `scripts/ops/audit_databases.py`

---

## Executive Summary

### Overall Status: ⚠️ **ATTENTION REQUIRED**

The CBI-V15 database infrastructure consists of:
- **1 MotherDuck cloud database** (md:cbi_v15) - Primary source of truth
- **3 local DuckDB files** - Multiple copies with inconsistent states

**Critical Findings:**
1. ✅ All 9 expected schemas exist in both MotherDuck and local DB
2. ❌ Most tables are **EMPTY** (no data) - only raw.fred_economic has 35,965 rows in MotherDuck
3. ⚠️ Schema drift: MotherDuck has 67 tables vs Local has 69 tables (inconsistent)
4. ⚠️ Multiple local DuckDB files exist with different modification times
5. ⚠️ Key feature tables (daily_ml_matrix_zl, bucket_scores) are EMPTY in both databases

---

## Database Topology

### MotherDuck Cloud (Primary Source of Truth)

**Connection:** `md:cbi_v15`
**Status:** ✅ Connected and accessible
**Total Tables:** 67 tables across 9 schemas

| Schema | Tables | Expected | Status | Notes |
|--------|--------|----------|--------|-------|
| raw | 13 | 13 | ✅ | Only fred_economic has data (35,965 rows) |
| staging | 10 | 7 | ⚠️ | 3 extra tables: china_daily, crush_daily, news_daily |
| features | 16 | 6 | ⚠️ | 10 extra tables created (all empty) |
| features_dev | 0 | ? | ✅ | Dev schema (expected to be empty) |
| training | 7 | 6 | ⚠️ | 1 extra table: daily_ml_matrix_zl |
| forecasts | 4 | 4 | ✅ | All tables present (all empty) |
| reference | 8 | 11 | ❌ | **Missing 3 tables** |
| ops | 8 | 7 | ⚠️ | 1 extra table: training_logs |
| explanations | 1 | 1 | ✅ | shap_values table present (empty) |

### Local DuckDB Files

#### Primary: `data/duckdb/cbi_v15.duckdb`
- **Size:** 9.3 MB
- **Last Modified:** 2025-12-15 13:38:10 (TODAY)
- **Status:** ✅ Active - Most recent modification
- **Total Tables:** 69 tables across 9 schemas
- **Data:** Reference tables have small amounts of seed data (8-10 rows)

#### Secondary: `src/data/duckdb/cbi_v15.duckdb`
- **Size:** 12 KB (essentially empty)
- **Last Modified:** 2025-12-15 09:14:16 (TODAY)
- **Status:** ⚠️ Nearly empty file, likely stale
- **Recommendation:** **DELETE** or archive

#### Archive: `archive/Data/duckdb/cbi_v15.duckdb`
- **Size:** 5.5 MB
- **Last Modified:** 2025-12-13 10:05:48 (2 days ago)
- **Status:** ⚠️ Stale backup
- **Recommendation:** **DELETE** or keep as backup only

---

## Detailed Findings

### 🔴 CRITICAL Issues

#### 1. Empty Data Tables
**Severity:** CRITICAL
**Impact:** System cannot function without data

All key tables are empty except `raw.fred_economic`:

| Table | MotherDuck | Local | Status |
|-------|------------|-------|--------|
| raw.databento_futures_ohlcv_1d | 0 rows | 0 rows | ❌ EMPTY |
| raw.fred_economic | 35,965 rows | 0 rows | ⚠️ MD only |
| staging.ohlcv_daily | 0 rows | 0 rows | ❌ EMPTY |
| features.daily_ml_matrix_zl | 0 rows | 0 rows | ❌ EMPTY |
| features.bucket_scores | 0 rows | 0 rows | ❌ EMPTY |
| forecasts.zl_predictions | 0 rows | 0 rows | ❌ EMPTY |

**Root Cause:** Ingestion jobs have not run successfully or data has been cleared.

**Action Required:**
1. Run all Trigger.dev ingestion jobs to populate raw tables
2. Run AnoFox feature engineering pipeline to populate staging/features
3. Verify Databento, CFTC, USDA, EIA, Weather collectors are operational

#### 2. Missing Reference Tables (MotherDuck)
**Severity:** CRITICAL
**Impact:** System cannot start without reference data

MotherDuck is missing 3 expected reference tables:
- `reference.feature_to_driver_group_map`
- `reference.geo_admin_regions`
- `reference.geo_countries`

**Action Required:**
1. Re-run database setup with `--force` flag:
   ```bash
   python scripts/setup_database.py --motherduck --force
   ```
2. Seed reference tables:
   ```bash
   python database/seeds/seed_reference_tables.py
   ```

#### 3. Reference Tables Empty (Both Databases)
**Severity:** CRITICAL
**Impact:** Cannot train models without symbol definitions

Critical reference tables are empty:
- `reference.symbols` - 0 rows (should have 33 symbols)
- `reference.regime_weights` - 0 rows
- `reference.train_val_test_splits` - 0/1 rows

**Action Required:**
```bash
# Seed all reference tables
python database/seeds/seed_symbols.py
python database/seeds/seed_regimes.py
python database/seeds/seed_splits.py
```

### 🟡 HIGH Priority Issues

#### 4. Schema Drift Between MotherDuck and Local
**Severity:** HIGH
**Impact:** Sync operations may fail, inconsistent behavior

| Schema | MotherDuck Tables | Local Tables | Drift |
|--------|------------------|--------------|-------|
| raw | 13 | 19 | +6 in local |
| staging | 10 | 7 | +3 in MD |
| features | 16 | 13 | +3 in MD |
| training | 7 | 6 | +1 in MD |
| ops | 8 | 7 | +1 in MD |
| reference | 8 | 12 | +4 in local |

**Tables in Local but not MotherDuck:**
- raw.fred_series_metadata
- raw.profarmer_articles
- raw.profarmer_crop_tour
- raw.scrapecreators_trump
- raw.tradingeconomics_calendar
- raw.tradingeconomics_indicators
- raw.v_cftc_cot_tff
- raw.weather_noaa
- reference.feature_to_driver_group_map
- reference.geo_admin_regions
- reference.geo_countries
- reference.weather_location_registry

**Tables in MotherDuck but not Local:**
- features.big8_bucket_scores
- features.news_signals
- features.tech_indicators
- ops.training_logs
- raw.noaa_weather_daily
- raw.tradingeconomics_commodities
- staging.china_daily
- staging.crush_daily
- staging.news_daily
- training.daily_ml_matrix_zl

**Action Required:**
1. Decide on canonical schema (recommend: MotherDuck DDL files)
2. Drop and recreate both databases from DDL:
   ```bash
   python scripts/setup_database.py --both --force
   ```
3. Seed reference data
4. Run fresh ingestion

#### 5. Multiple Local DuckDB Files
**Severity:** HIGH
**Impact:** Confusion, potential for using wrong database

3 local DuckDB files exist:
- `data/duckdb/cbi_v15.duckdb` - **PRIMARY** (9.3 MB, active today)
- `src/data/duckdb/cbi_v15.duckdb` - **STALE** (12 KB, nearly empty)
- `archive/Data/duckdb/cbi_v15.duckdb` - **OLD BACKUP** (5.5 MB, 2 days old)

**Action Required:**
1. Standardize on `data/duckdb/cbi_v15.duckdb` as the ONLY local database
2. Update all scripts to use this path (verify with grep)
3. Delete `src/data/duckdb/cbi_v15.duckdb`
4. Move `archive/Data/duckdb/cbi_v15.duckdb` to timestamped backup if needed:
   ```bash
   mv "archive/Data/duckdb/cbi_v15.duckdb" \
      "archive/Data/duckdb/cbi_v15_backup_20251213.duckdb"
   ```

### 🟢 MEDIUM Priority Issues

#### 6. Extra Tables Created (Materialized Views?)
**Severity:** MEDIUM
**Impact:** Storage overhead, potential confusion

Several schemas have more tables than expected by DDL:
- **features:** 16 tables (expected 6) - 10 extra bucket tables created
- **staging:** 10 tables (expected 7) - 3 extra staging tables
- **training:** 7 tables (expected 6) - 1 extra ml_matrix copy

This may be intentional (materialized views, cached computations) or accidental.

**Action Required:**
1. Review which tables are materialized views vs permanent tables
2. Update DDL documentation to reflect actual expected table count
3. Consider using `features_dev` schema for experimental tables

#### 7. Column Count Mismatch in Key Tables
**Severity:** MEDIUM
**Impact:** Potential for query failures if schema incompatible

| Table | MotherDuck Columns | Local Columns | Drift |
|-------|-------------------|---------------|-------|
| features.daily_ml_matrix_zl | 88 | 38 | +50 in MD |
| features.bucket_scores | 11 | 20 | -9 in MD |
| forecasts.zl_predictions | 33 | 15 | +18 in MD |
| staging.ohlcv_daily | 8 | 11 | -3 in MD |

**Action Required:**
1. Recreate from canonical DDL to ensure consistent schemas
2. Update DDL files if column differences are intentional

---

## Data Availability Summary

### MotherDuck Cloud

| Schema | Tables with Data | Total Tables | % Populated |
|--------|-----------------|--------------|-------------|
| raw | 1 | 13 | 7.7% |
| staging | 0 | 10 | 0% |
| features | 0 | 16 | 0% |
| training | 0 | 7 | 0% |
| forecasts | 0 | 4 | 0% |
| reference | 0 | 8 | 0% |
| ops | 0 | 8 | 0% |
| explanations | 0 | 1 | 0% |
| **TOTAL** | **1** | **67** | **1.5%** |

### Local DuckDB (Primary)

| Schema | Tables with Data | Total Tables | % Populated |
|--------|-----------------|--------------|-------------|
| raw | 0 | 19 | 0% |
| staging | 0 | 7 | 0% |
| features | 0 | 13 | 0% |
| training | 0 | 6 | 0% |
| forecasts | 0 | 4 | 0% |
| reference | 5 | 12 | 41.7% |
| ops | 0 | 7 | 0% |
| explanations | 0 | 1 | 0% |
| **TOTAL** | **5** | **69** | **7.2%** |

**Note:** Local reference tables have minimal seed data (8-10 rows each), not production data.

---

## Key Feature Tables Health Check

### features.daily_ml_matrix_zl
- **MotherDuck:** 0 rows, 88 columns
- **Local:** 0 rows, 38 columns
- **Status:** ❌ EMPTY (cannot train models)
- **Expected:** 3,000+ rows (daily observations since 2015)
- **Columns:** Mismatch suggests schema drift

### features.bucket_scores
- **MotherDuck:** 0 rows, 11 columns
- **Local:** 0 rows, 20 columns
- **Status:** ❌ EMPTY (dashboard cannot display Big 8 scores)
- **Expected:** 3,000+ rows
- **Columns:** Mismatch suggests schema drift

### forecasts.zl_predictions
- **MotherDuck:** 0 rows, 33 columns
- **Local:** 0 rows, 15 columns
- **Status:** ❌ EMPTY (no forecasts available)
- **Expected:** Daily predictions for 1, 5, 21, 63, 252 day horizons

### staging.ohlcv_daily
- **MotherDuck:** 0 rows, 8 columns
- **Local:** 0 rows, 11 columns
- **Status:** ❌ EMPTY (staging pipeline blocked)
- **Expected:** 100,000+ rows (33 symbols × 3,000+ days)

---

## Macros Status

**Note:** Macro audit was not performed as all tables are empty. Macros cannot be tested without data.

**Action Required:**
After data ingestion, verify macros are loaded:
```bash
python scripts/verify_macros.py
```

Expected macros:
- Technical indicators: `calc_rsi`, `calc_macd`, `calc_bollinger`, etc.
- Bucket features: `calc_bucket_crush`, `calc_bucket_fx`, etc.
- Big 8 scores: `calc_all_bucket_scores`

---

## Recommended Action Plan

### Phase 1: Database Cleanup & Schema Alignment (IMMEDIATE)

**Priority:** CRITICAL
**Estimated Time:** 30 minutes

```bash
# 1. Backup current state (if needed)
cp "data/duckdb/cbi_v15.duckdb" \
   "archive/backups/cbi_v15_pre_audit_$(date +%Y%m%d).duckdb"

# 2. Clean and recreate schemas from canonical DDL
python scripts/setup_database.py --both --force

# 3. Verify schemas created correctly
python scripts/ops/audit_databases.py | grep "SCHEMAS:"

# 4. Seed reference data
python database/seeds/seed_symbols.py
python database/seeds/seed_regimes.py
python database/seeds/seed_splits.py
python database/seeds/seed_reference_tables.py

# 5. Verify reference data seeded
python scripts/ops/audit_databases.py | grep -A 20 "REFERENCE:"
```

### Phase 2: Data Ingestion (HIGH PRIORITY)

**Priority:** HIGH
**Estimated Time:** 2-4 hours (depends on API rate limits)

```bash
# 1. FRED Economic Data (already has 35k rows in MD, sync to local)
python trigger/FRED/Scripts/fred_seed_harvest.ts

# 2. Databento OHLCV (33 symbols, ~3 years)
python trigger/DataBento/Scripts/collect_daily.py

# 3. CFTC Commitment of Traders
python trigger/CFTC/Scripts/ingest_cot.py

# 4. USDA Reports
python trigger/USDA/Scripts/ingest_wasde.py
python trigger/USDA/Scripts/ingest_export_sales.py

# 5. EIA Energy Data
python trigger/EIA_EPA/Scripts/collect_eia_biofuels.py

# 6. Weather Data
python trigger/Weather/Scripts/ingest_weather.py

# 7. Verify data ingested
python scripts/ops/audit_databases.py | grep "rows"
```

### Phase 3: Feature Engineering (MEDIUM PRIORITY)

**Priority:** MEDIUM
**Estimated Time:** 1-2 hours

```bash
# 1. Build staging tables (cleaned/normalized)
python src/engines/anofox/build_features.py --stage staging

# 2. Build feature tables (technical indicators, buckets)
python src/engines/anofox/build_features.py --stage features

# 3. Build training matrix
python src/engines/anofox/build_training.py

# 4. Verify features created
python scripts/ops/audit_databases.py | grep "daily_ml_matrix_zl"
```

### Phase 4: Sync & Consolidation (LOW PRIORITY)

**Priority:** LOW
**Estimated Time:** 30 minutes

Ingest → MotherDuck → Feature Engineering → Sync → Local Training → Upload Predictions → Dashboard

```bash
# 1. Sync MotherDuck → Local for training
python scripts/sync_motherduck_to_local.py

# 2. Delete stale local databases
rm "src/data/duckdb/cbi_v15.duckdb"

# 3. Move archive to timestamped backup
mv "archive/Data/duckdb/cbi_v15.duckdb" \
   "archive/Data/duckdb/cbi_v15_backup_20251213.duckdb"

# 4. Update .gitignore to prevent accidental commits
echo "src/data/duckdb/*.duckdb" >> .gitignore

# 5. Final audit
python scripts/ops/audit_databases.py > docs/ops/DATABASE_AUDIT_POST_FIX.md
```

---

## Scripts Used by Database Location

Based on grep analysis, the following scripts access local DuckDB:

### Scripts using `data/duckdb/` (PRIMARY - CORRECT) ✅
- `scripts/setup_database.py` - Main setup script
- `scripts/sync_motherduck_to_local.py` - Sync script
- `scripts/check_database_status.py` - Status checker
- `scripts/cleanup_database.py` - Cleanup utility
- `database/seeds/seed_*.py` - All seed scripts
- `database/tests/harness.py` - Test harness
- Most training scripts

### Scripts using `src/data/duckdb/` (SECONDARY - INCORRECT) ⚠️
- `database/migrations/migrate.py` - **NEEDS UPDATE**
- `database/tests/harness.py` - Fallback path
- Legacy scripts in `archive/`

**Action Required:**
Update `database/migrations/migrate.py` to use `data/duckdb/` path.

---

## Connection String Reference

### MotherDuck Cloud
```python
# Python
import duckdb
conn = duckdb.connect(f"md:cbi_v15?motherduck_token={MOTHERDUCK_TOKEN}")

# SQL (after ATTACH)
SELECT * FROM cbi_v15.raw.fred_economic;
```

### Local DuckDB (Recommended Path)
```python
# Python
conn = duckdb.connect("data/duckdb/cbi_v15.duckdb")

# Can also ATTACH MotherDuck from local connection
conn.execute(f"ATTACH 'md:cbi_v15?motherduck_token={token}' AS md_source")
```

### IntelliJ IDEA Database Connection
```
Type: DuckDB
URL: jdbc:duckdb:md:cbi_v15?motherduck_token=<YOUR_TOKEN>
Database: cbi_v15
```

Or for local:
```
URL: jdbc:duckdb:/Volumes/Satechi Hub/CBI-V15/data/duckdb/cbi_v15.duckdb
```

---

## Monitoring & Maintenance

### Daily Checks
```bash
# Quick health check
python scripts/ops/check_data_availability.py

# Verify ingestion completion
python scripts/ops/ingestion_status.py
```

### Weekly Audits
```bash
# Full database audit
python scripts/ops/audit_databases.py > docs/ops/weekly_audit_$(date +%Y%m%d).md

# Sync MotherDuck → Local if training
python scripts/sync_motherduck_to_local.py
```

### Monthly Tasks
- Review and archive old local backups
- Verify database size (should be 50-500 MB with full data)
- Check for schema drift
- Update reference data if symbol list changes

---

## Appendix: Expected vs Actual Table Counts

| Schema | Expected | MotherDuck | Local | Status |
|--------|----------|------------|-------|--------|
| reference | 11 | 8 | 12 | ⚠️ Drift |
| raw | 13 | 13 | 19 | ⚠️ Drift |
| staging | 7 | 10 | 7 | ⚠️ Drift |
| features | 6 | 16 | 13 | ⚠️ Drift |
| training | 6 | 7 | 6 | ⚠️ Drift |
| forecasts | 4 | 4 | 4 | ✅ Match |
| ops | 7 | 8 | 7 | ⚠️ Drift |
| explanations | 1 | 1 | 1 | ✅ Match |
| **TOTAL** | **55** | **67** | **69** | ⚠️ **Drift** |

**Conclusion:** Actual table counts exceed DDL expectations due to:
1. Materialized bucket tables in features schema
2. Additional staging tables for specialized pipelines
3. Extra raw tables for new data sources
4. Missing DDL files for some tables

**Recommendation:** Update DDL documentation to match production schema, or clean up extra tables.

---

## Contact & Support

**Database Administrator:** Chris (US Oil Solutions)
**Documentation:** `database/README.md`
**Issue Tracker:** GitHub Issues
**Audit Script:** `scripts/ops/audit_databases.py`

---

**Report Generated:** 2025-12-15 14:31:49
**Next Audit Due:** 2025-12-22 (Weekly)
**Status:** ⚠️ **ATTENTION REQUIRED - Follow Phase 1-4 action plan**
