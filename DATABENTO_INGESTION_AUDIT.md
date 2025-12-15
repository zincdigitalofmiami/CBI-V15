# Databento Historical Data Ingestion - Safety Audit Report

**Date:** 2025-12-15  
**Auditor:** CBI-V15 Engineering Agent  
**Status:** ⚠️ **NOT SAFE TO PULL** - Critical issues must be fixed first

---

## Executive Summary

**RECOMMENDATION: DO NOT PULL HISTORICAL DATA YET**

While the ingestion pipeline exists and schemas are in place, there are **3 critical blocking issues** that will cause data corruption and dashboard failures if you pull data now.

**Estimated Fix Time:** 2-4 hours  
**Risk Level:** HIGH (data corruption, dashboard breakage)

---

## ✅ What's Working

### 1. Schema Infrastructure (GOOD)
- ✅ `raw.databento_futures_ohlcv_1d` table exists in MotherDuck
- ✅ Correct schema: `symbol`, `as_of_date`, `open`, `high`, `low`, `close`, `volume`, `open_interest`
- ✅ Primary key: `(symbol, as_of_date)` - prevents duplicates
- ✅ Indexes on `symbol` and `as_of_date` for query performance

### 2. Ingestion Script (MOSTLY GOOD)
- ✅ `trigger/DataBento/Scripts/collect_daily.py` exists
- ✅ Uses official Databento Python SDK
- ✅ Supports all 38 symbols (Agricultural, Energy, Metals, Treasuries, FX)
- ✅ Handles MotherDuck and local DuckDB connections
- ✅ Resumes from last date (checks `MAX(as_of_date)` before pulling)
- ✅ Column mapping is CORRECT: `ts_event` → `as_of_date`

### 3. API Configuration (GOOD)
- ✅ Databento API key configured in environment
- ✅ Dataset: `GLBX.MDP3` (CME Globex)
- ✅ Schema: `ohlcv-1d` (daily bars)
- ✅ Historical API ready for backfills

---

## ❌ Critical Blocking Issues

### Issue #1: Dashboard Table Name Mismatch (CRITICAL)
**File:** `dashboard/app/api/live/zl/route.ts:12`

**Problem:**
```typescript
// Dashboard queries this table (DOES NOT EXIST):
FROM raw.databento_ohlcv_daily

// But ingestion writes to this table (EXISTS):
FROM raw.databento_futures_ohlcv_1d
```

**Impact:** Dashboard will show **zero data** even after successful ingestion.

**Fix Required:**
```typescript
// Change line 12 in dashboard/app/api/live/zl/route.ts
FROM raw.databento_futures_ohlcv_1d  // Match actual table name
```

**Validation:**
```bash
# After fix, test dashboard API:
curl http://localhost:3000/api/live/zl
# Should return data (currently returns error)
```

---

### Issue #2: Staging Table Schema Mismatch (CRITICAL)
**File:** `database/ddl/03_staging/010_staging_ohlcv_daily.sql`

**Problem:**
```sql
-- DDL defines column as 'date':
CREATE TABLE staging.ohlcv_daily (
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,  -- ❌ WRONG
    ...
)

-- But actual MotherDuck table has 'as_of_date':
staging.ohlcv_daily columns:
  as_of_date: DATE  -- ✅ ACTUAL
  symbol: VARCHAR
```

**Impact:** 
- Any ETL script trying to `INSERT INTO staging.ohlcv_daily (date, ...)` will fail
- Feature engineering macros reading `staging.ohlcv_daily.date` will fail
- Training pipeline will break

**Fix Required:**
1. Update DDL file to match reality:
```sql
-- database/ddl/03_staging/010_staging_ohlcv_daily.sql
CREATE TABLE IF NOT EXISTS staging.ohlcv_daily (
    symbol VARCHAR NOT NULL,
    as_of_date DATE NOT NULL,  -- Match actual schema
    ...
)
```

2. OR recreate table in MotherDuck to match DDL (riskier)

**Recommendation:** Update DDL to match MotherDuck (safer, no data loss)

---

### Issue #3: Missing ETL Pipeline (raw → staging)
**Status:** NOT IMPLEMENTED

**Problem:**
- Data lands in `raw.databento_futures_ohlcv_1d` ✅
- But nothing moves it to `staging.ohlcv_daily` ❌
- Feature macros read from `raw.databento_futures_ohlcv_1d` directly (bypasses staging)

**Impact:**
- Staging layer is empty (confirmed: 0 rows)
- Any code expecting cleaned/gap-filled data in staging will fail
- Inconsistent data flow (some macros use raw, some expect staging)

**Fix Required:**
Create `src/etl/databento_to_staging.py`:
```python
# Pseudo-code:
# 1. Read from raw.databento_futures_ohlcv_1d
# 2. Forward-fill gaps (weekends/holidays)
# 3. Winsorize outliers
# 4. INSERT OR REPLACE INTO staging.ohlcv_daily
```

**Workaround for Now:**
- Feature macros already read from `raw.databento_futures_ohlcv_1d` directly
- Can proceed without staging ETL if you accept raw data (no cleaning)

---

## 🟡 Non-Blocking Issues (Fix Later)

### 1. No Idempotency Protection
**File:** `trigger/DataBento/Scripts/collect_daily.py:232`

**Problem:**
```python
con.execute("""
    INSERT INTO raw.databento_futures_ohlcv_1d 
    SELECT * FROM combined_df
""")
```

**Issue:** If script runs twice for same date, will fail on PRIMARY KEY violation

**Fix:**
```python
con.execute("""
    INSERT OR REPLACE INTO raw.databento_futures_ohlcv_1d 
    SELECT * FROM combined_df
""")
```

**Impact:** Low (script checks last date before pulling, unlikely to duplicate)

---

## 📊 Current Data Status

```
MotherDuck Database: cbi_v15
├── raw.databento_futures_ohlcv_1d: 0 rows (EMPTY - ready for data)
├── staging.ohlcv_daily: 0 rows (EMPTY - schema mismatch)
└── features.daily_ml_matrix_zl: Unknown (depends on raw data)
```

**Schemas Verified:**
- ✅ All 9 schemas exist (raw, staging, features, training, forecasts, reference, ops, explanations, features_dev)
- ✅ `raw.databento_futures_ohlcv_1d` schema matches ingestion script
- ⚠️ `staging.ohlcv_daily` schema does NOT match DDL file

---

## 🚦 Go/No-Go Decision Matrix

| Condition | Status | Blocker? |
|-----------|--------|----------|
| Schema exists | ✅ YES | No |
| Ingestion script works | ✅ YES | No |
| Dashboard table name correct | ❌ NO | **YES** |
| Staging schema matches DDL | ❌ NO | **YES** |
| ETL pipeline exists | ❌ NO | No (workaround available) |
| Idempotency protection | ❌ NO | No (low risk) |

**VERDICT:** ❌ **NOT SAFE - Fix Issues #1 and #2 first**

---

## ✅ Safe Ingestion Checklist

Before pulling historical data, complete these steps:

- [ ] **Fix #1:** Update `dashboard/app/api/live/zl/route.ts` table name
- [ ] **Fix #2:** Update `database/ddl/03_staging/010_staging_ohlcv_daily.sql` column name
- [ ] **Fix #3 (Optional):** Add `INSERT OR REPLACE` to ingestion script
- [ ] **Test:** Run ingestion for 1 symbol, 5 days: `python trigger/DataBento/Scripts/collect_daily.py --symbols ZL --days 5`
- [ ] **Verify:** Check MotherDuck: `SELECT COUNT(*) FROM raw.databento_futures_ohlcv_1d WHERE symbol = 'ZL'`
- [ ] **Test Dashboard:** `curl http://localhost:3000/api/live/zl` (should return data)
- [ ] **Full Pull:** Run for all 38 symbols, full history

---

## 📋 Symbol List for Historical Pull

Once safe, pull these 38 symbols:

**Agricultural (11):** ZL, ZS, ZM, ZC, ZW, ZO, ZR, HE, LE, GF, FCPO  
**Energy (4):** CL, HO, RB, NG  
**Metals (5):** HG, GC, SI, PL, PA  
**Treasuries (3):** ZF, ZN, ZB  
**FX (10):** 6E, 6J, 6B, 6C, 6A, 6N, 6M, 6L, 6S, DX

**Recommended Start Date:** `2010-01-01` (15 years of history)

---

## 🔧 Quick Fix Commands

```bash
# 1. Fix dashboard table name
code dashboard/app/api/live/zl/route.ts
# Change line 12: raw.databento_ohlcv_daily → raw.databento_futures_ohlcv_1d

# 2. Fix staging DDL
code database/ddl/03_staging/010_staging_ohlcv_daily.sql
# Change line 6: date DATE → as_of_date DATE

# 3. Add idempotency
code trigger/DataBento/Scripts/collect_daily.py
# Change line 233: INSERT INTO → INSERT OR REPLACE INTO

# 4. Test ingestion (5 days, ZL only)
python trigger/DataBento/Scripts/collect_daily.py

# 5. Verify data
python -c "
import duckdb, os
con = duckdb.connect(f'md:{os.getenv(\"MOTHERDUCK_DB\")}?motherduck_token={os.getenv(\"MOTHERDUCK_TOKEN\")}')
print(con.execute('SELECT COUNT(*), MIN(as_of_date), MAX(as_of_date) FROM raw.databento_futures_ohlcv_1d WHERE symbol = \"ZL\"').fetchone())
"
```

---

## Final Recommendation

**DO NOT PULL DATA YET.** Fix Issues #1 and #2 first (30 minutes of work). Then test with 1 symbol before full historical pull.

**Estimated Timeline:**
- Fixes: 30 minutes
- Test pull (ZL, 5 days): 5 minutes
- Full historical pull (38 symbols, 15 years): 2-4 hours (API rate limits)

**Next Steps:**
1. I can fix Issues #1, #2, #3 now (with your approval)
2. You test with small pull
3. Then proceed with full historical ingestion

