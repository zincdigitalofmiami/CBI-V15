# PHASE 0: CRITICAL INFRASTRUCTURE & BUG FIXES

**Goal:** Fix critical data pipeline bugs and establish local DuckDB mirror for Mac M4 training  
**Status:** IN PROGRESS  
**Dependencies:** None (foundational)  
**Estimated Time:** 8-12 hours  
**Risk Level:** HIGH (blocks all downstream work)

---

## 📋 TASKS (13 total)

### Task 0.1: Fix Databento Column Name (HIGH RISK)
**UUID:** `6MtEwcw77bDwzJNvLEiGoZ`

**Problem:** Script uses `date` but `raw.databento_ohlcv_daily` expects `as_of_date`

**File:** `trigger/DataBento/Scripts/collect_daily.py`

**Changes:**
```python
# Lines 146, 160, 165, 196, 234
# BEFORE:
df.rename(columns={'ts_event': 'date'})

# AFTER:
df.rename(columns={'ts_event': 'as_of_date'})
```

**Validation:**
```bash
python trigger/DataBento/Scripts/collect_daily.py --symbol ZL --days 5
# Expected: Data inserted into raw.databento_ohlcv_daily without column errors
```

**Risk:** HIGH - Blocks all market data ingestion

---

### Task 0.2: Fix FRED Table Reference (HIGH RISK)
**UUID:** `4ujmzbaXED6dCh1ZcErpPB`

**Problem:** SQL macros read from `raw.fred_observations` but collectors write to `raw.fred_economic`

**File:** `database/macros/big8_bucket_features.sql`

**Changes:**
```sql
-- Find all references to raw.fred_observations
-- Replace with raw.fred_economic
FROM raw.fred_economic  -- NOT raw.fred_observations
```

**Validation:**
```sql
SELECT * FROM calc_all_bucket_scores() LIMIT 5;
-- Expected: No 'table not found' errors, bucket scores computed
```

**Risk:** HIGH - Blocks Fed, Volatility bucket features

---

### Task 0.3: Fix EIA Table Reference (HIGH RISK)
**UUID:** `9rdTjajP2vimCqtxe7EJH5`

**Problem:** Previously mismatched EIA targets; current collectors and macros both use `raw.eia_biofuels`

**File:** `database/macros/big8_bucket_features.sql` (lines 295-298)

**Changes:**
```sql
-- Update Biofuel bucket macro
FROM raw.eia_biofuels
WHERE series_id LIKE 'rin_%'
```

**Validation:**
```sql
SELECT biofuel_bucket_score 
FROM calc_all_bucket_scores() 
WHERE as_of_date = (SELECT MAX(as_of_date) FROM raw.eia_biofuels);
-- Expected: Biofuel bucket score computed without NULL
```

**Risk:** HIGH - Blocks Biofuel bucket features

---

### Task 0.4: Add Missing FRED Series (MEDIUM RISK)
**UUID:** `kE1WyMwBfBtuB1YiBqwtPZ`

**Problem:** Missing DFEDTARU (Fed Funds Target) and VIXCLS (VIX) in SERIES list

**File:** `trigger/FRED/Scripts/collect_fred_financial_conditions.py`

**Changes:**
```python
SERIES = [
    'DFEDTARU',  # ADD: Fed Funds Target Rate (Upper Bound)
    'VIXCLS',    # ADD: CBOE Volatility Index
    # ... existing series
]
```

**Validation:**
```bash
python src/ingestion/fred/collect_fred_financial_conditions.py
# Then check:
SELECT COUNT(*) FROM raw.fred_economic WHERE series_id IN ('DFEDTARU', 'VIXCLS');
# Expected: Both series present with >100 observations each
```

**Big 8 Impact:** Fed (bucket 4), Volatility (bucket 8)

---

### Task 0.5: Create Local DuckDB Mirror Directory (LOW RISK)
**UUID:** `wFsu5iA46BBYcWfv8koaun`

**Purpose:** Create directories for local DuckDB mirror and AutoGluon model artifacts

**Commands:**
```bash
mkdir -p data/duckdb
mkdir -p data/models

# Create .gitignore files
echo "*.duckdb" > data/duckdb/.gitignore
echo "*.pkl" > data/models/.gitignore
echo "*.joblib" >> data/models/.gitignore
echo "models/" >> data/models/.gitignore
```

**Validation:**
```bash
ls -la data/duckdb/ && ls -la data/models/
# Expected: Directories exist, .gitignore files present
```

---

### Task 0.6: Create MotherDuck → Local DuckDB Sync Script (CRITICAL)
**UUID:** `mr9UoXJiEvQaNScku3WxJL`

**Purpose:** Sync features from MotherDuck to local DuckDB for 100-1000x faster Mac M4 training

**File:** `scripts/sync_motherduck_to_local.py`

**Implementation:**
```python
import duckdb
import os
from dotenv import load_env

load_env()

def sync_motherduck_to_local():
    # Connect to MotherDuck
    md_token = os.getenv('MOTHERDUCK_TOKEN')
    md_conn = duckdb.connect(f'md:cbi_v15?motherduck_token={md_token}')
    
    # Connect to local DuckDB
    local_conn = duckdb.connect('data/duckdb/cbi_v15.duckdb')
    
    # Sync tables
    tables = [
        'features.daily_ml_matrix',
        'training.daily_ml_matrix_zl',
        'training.bucket_predictions'
    ]
    
    for table in tables:
        print(f"Syncing {table}...")
        df = md_conn.execute(f"SELECT * FROM {table}").df()
        local_conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df")
        print(f"  ✓ {len(df)} rows synced")
    
    local_conn.close()
    md_conn.close()

if __name__ == '__main__':
    sync_motherduck_to_local()
```

**Validation:**
```bash
python scripts/sync_motherduck_to_local.py
ls -lh data/duckdb/cbi_v15.duckdb
# Expected: Local database created, >10MB size
```

**Risk:** CRITICAL - Without this, Mac M4 training will be 100-1000x slower

---

### Task 0.7: Create Data Quality Validation Framework (CRITICAL)
**UUID:** `ib3RnkBofBnwsbozG8dsHx`

**Purpose:** Enforce 6-dimension data quality checks on EVERY ingestion pipeline

**File:** `src/validation/data_quality_checks.py`

**Implementation (6 Dimensions):**
1. **Accuracy:** Verify data matches source
2. **Completeness:** Check for NULLs in join keys (as_of_date, series_id)
3. **Consistency:** Verify staging matches raw row-for-row
4. **Timeliness:** Check data freshness (max_age_days=1)
5. **Validity:** Check positive prices, valid ranges
6. **Uniqueness:** Check for duplicate rows on (as_of_date, series_id)

```python
def validate_ingestion(df, source, expected_schema):
    """Validate data quality before writing to database."""
    checks = {}

    # 1. Completeness
    checks['completeness'] = df[['as_of_date', 'series_id']].notna().all().all()

    # 2. Uniqueness
    checks['uniqueness'] = not df.duplicated(subset=['as_of_date', 'series_id']).any()

    # 3. Validity
    if 'value' in df.columns:
        checks['validity'] = (df['value'] > 0).all()

    # 4. Timeliness
    max_date = df['as_of_date'].max()
    checks['timeliness'] = (pd.Timestamp.now() - max_date).days <= 1

    # 5. Consistency (schema match)
    checks['consistency'] = set(df.columns) == set(expected_schema)

    all_passed = all(checks.values())
    return all_passed, checks
```

**Validation:**
```bash
python -c 'from src.validation.data_quality_checks import validate_ingestion; print("Validator ready")'
```

---

### Task 0.8: Add ON CONFLICT Idempotency Logic (HIGH RISK)
**UUID:** `wL9sbRjMLzcGRni2RHnirw`

**Purpose:** All Trigger.dev jobs must be idempotent (can run twice without duplicating data)

**Files to Update:**
- `src/ingestion/databento/collect_daily.py`
- `src/ingestion/fred/collect_fred_*.py`
- `src/ingestion/eia/collect_eia_biofuels.py`
- `src/ingestion/usda/ingest_export_sales.py`
- `src/ingestion/cftc/ingest_cot.py`

**Pattern:**
```sql
INSERT INTO raw.{table} (as_of_date, series_id, value, source)
VALUES (?, ?, ?, ?)
ON CONFLICT (as_of_date, series_id) DO UPDATE SET
    value = EXCLUDED.value,
    updated_at = CURRENT_TIMESTAMP;
```

**Validation:**
```bash
# Run each ingestion job twice
python src/ingestion/databento/collect_daily.py --symbol ZL --days 5
python src/ingestion/databento/collect_daily.py --symbol ZL --days 5

# Check row count unchanged
SELECT COUNT(*) FROM raw.databento_ohlcv_daily WHERE symbol = 'ZL';
# Expected: Same count on both runs
```

---

### Task 0.9: Fix setup_database.py Deleted File Reference (HIGH RISK)
**UUID:** `237KEwaYT5cJ4J2vBQWjoH`

**Problem:** Line 71 references 'fred_macro.sql' which was deleted in EIA/EPA split

**File:** `scripts/setup_database.py`

**Changes:**
```python
# Line ~71 in setup_raw_tables() function
# REMOVE this line:
'fred_macro.sql',  # DELETE - file no longer exists
```

**Validation:**
```bash
python scripts/setup_database.py --both
# Expected: No 'file not found' errors for fred_macro.sql
```

---

### Task 0.10: Remove All TSci Dependencies (MEDIUM RISK)
**UUID:** `kPK2BQMnyUnBFejmNJVa8E`

**Purpose:** TSci has been dropped from V15.1 - remove all references

**Files to Update:**

1. **src/engines/base_engine.py** (Line ~10)
```python
# BEFORE:
"""TSci is no longer the primary orchestrator; it is optional/legacy."""

# AFTER:
"""Optional utility for SQL execution (no TSci)."""
```

2. **src/engines/anofox/anofox_bridge.py** (Lines 1-10, 26-31, 69-76)
```python
# Replace all instances of:
"Historically called by TSci..."
# With:
"Generic utility for..."
```

3. **Archive TSci Documentation (5 files):**
```bash
# Move to docs/_archive/
mv docs/project_docs/timeseriesscientist_integration.md docs/_archive/
mv docs/project_docs/tsci_anofox_architecture.md docs/_archive/
mv docs/project_docs/tsci_verification_report.md docs/_archive/
mv docs/project_docs/admin_tsci_integration_1764726594190.webp docs/_archive/
mv docs/project_docs/admin_tsci_verified_1764726658521.webp docs/_archive/
```

**Validation:**
```bash
grep -r 'TSci\|TimeSeriesScientist\|tsci' database/ src/ docs/ AGENTS.md
# Expected: No results (all TSci references removed)
```

**Dependencies:** Phase -1 complete (docs/_archive/ directory exists)

---

### Task 0.11: Parameterize Environment Variables (HIGH RISK)
**UUID:** `qGNNkACKGH6SYY6swMcHXG`

**Purpose:** Remove all hard-coded paths (e.g., /Volumes/Satechi Hub/...)

**Files to Audit:**
- All scripts in `scripts/`
- All ingestion scripts in `src/ingestion/`
- All training scripts in `src/training/`
- All Trigger.dev jobs in `trigger/`

**Pattern:**
```python
# BEFORE:
db_path = '/Volumes/Satechi Hub/CBI-V15/data/duckdb/cbi_v15.duckdb'

# AFTER:
import os
from pathlib import Path

db_path = os.getenv('LOCAL_DUCKDB_PATH', 'data/duckdb/cbi_v15.duckdb')
db_path = Path(db_path).resolve()
```

**Update .env.example:**
```bash
MOTHERDUCK_TOKEN=your_token_here
DATABENTO_API_KEY=your_key_here
FRED_API_KEY=your_key_here
EIA_API_KEY=your_key_here
EPA_RIN_SOURCE=https://www.epa.gov/...
LOCAL_DUCKDB_PATH=data/duckdb/cbi_v15.duckdb
MODEL_ARTIFACTS_PATH=data/models/
```

**Validation:**
```bash
grep -r '/Volumes/Satechi Hub' scripts/ src/ trigger/
# Expected: No results (all hard-coded paths removed)
```

---

### Task 0.12: Validate Phase 0 Complete (CRITICAL)
**UUID:** `t9nbeyciHZfQHJgBP7qoWM`

**Validation Commands:**
```bash
# 1. Setup database
python scripts/setup_database.py --both

# 2. Test Databento ingestion
python src/ingestion/databento/collect_daily.py --symbol ZL --days 5

# 3. Test FRED ingestion
python src/ingestion/fred/collect_fred_financial_conditions.py

# 4. Sync to local DuckDB
python scripts/sync_motherduck_to_local.py

# 5. System status
bash scripts/system_status.sh
```

**Expected Outputs:**
- ✅ All schemas created (8 schemas)
- ✅ Databento data in raw.databento_ohlcv_daily (5 rows for ZL)
- ✅ FRED data in raw.fred_economic (DFEDTARU, VIXCLS present)
- ✅ Local DuckDB mirror created at data/duckdb/cbi_v15.duckdb
- ✅ System status shows all green checks

**Success Criteria:**
- No 'table not found' errors
- No 'column not found' errors
- Data flows MotherDuck → Local successfully

**⚠️ STOP:** Phase 1-5 cannot proceed without Phase 0 complete

---

## 📊 PHASE 0 SUMMARY

| Metric | Value |
|--------|-------|
| **Total Tasks** | 13 |
| **High Risk** | 7 tasks |
| **Medium Risk** | 1 task |
| **Low Risk** | 1 task |
| **Critical** | 4 tasks |
| **Estimated Time** | 8-12 hours |

**Critical Path:** Tasks 0.1-0.6 must complete before any training can begin

**Next Phase:** Phase 1 (Critical Data Feeds)
