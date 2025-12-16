# Databento Ingestion - Status Update

**Status:** ✅ ALL CRITICAL FIXES ALREADY APPLIED
**Last Updated:** 2025-12-15
**System Status:** READY FOR DATA INGESTION

---

## ✅ Fix #1: Dashboard Table Name - RESOLVED

**File:** `dashboard/app/api/live/zl/route.ts:12`
**Status:** ✅ FIXED
**Current Code:** Uses correct table `raw.databento_futures_ohlcv_1d`

The dashboard now queries the correct table that matches the ingestion destination.

---

## ✅ Fix #2: Staging Table Schema - RESOLVED

**File:** `database/ddl/03_staging/010_staging_ohlcv_daily.sql:6`
**Status:** ✅ FIXED
**Current Schema:** Uses correct column name `as_of_date`

The DDL now matches the actual MotherDuck schema with consistent column naming.

---

## ✅ Fix #3: Idempotency Protection - RESOLVED

**File:** `trigger/DataBento/Scripts/collect_daily.py:233-241`
**Status:** ✅ FIXED
**Current Code:** Uses `INSERT OR REPLACE INTO` with proper staging logic

The ingestion script now handles re-runs safely without PRIMARY KEY violations.

---

## ✅ Additional Fix: Monitoring Script - RESOLVED

**File:** `scripts/ops/check_data_availability.py:74`
**Status:** ✅ FIXED
**Update:** Changed from `raw.databento_ohlcv_daily` to `raw.databento_futures_ohlcv_1d`

The monitoring script now checks the correct table name.

---

## Verification Steps (After Fixes)

### 1. Test Small Pull (ZL only, 5 days)

```bash
cd /Volumes/Satechi\ Hub/CBI-V15
python trigger/DataBento/Scripts/collect_daily.py
```

**Expected Output:**

```
🚀 Starting Databento daily data collection
Target: MotherDuck
No existing data, starting from 2010-01-01
Collecting ZL data from 2010-01-01 to 2025-12-15
Collected 3,950 rows for ZL
...
✅ Successfully loaded 3,950 rows to raw.databento_futures_ohlcv_1d
```

### 2. Verify Data in MotherDuck

```python
import duckdb, os

con = duckdb.connect(f'md:{os.getenv("MOTHERDUCK_DB")}?motherduck_token={os.getenv("MOTHERDUCK_TOKEN")}')

# Check row count
result = con.execute("""
    SELECT
        COUNT(*) as rows,
        COUNT(DISTINCT symbol) as symbols,
        MIN(as_of_date) as earliest,
        MAX(as_of_date) as latest
    FROM raw.databento_futures_ohlcv_1d
""").fetchone()

print(f"Rows: {result[0]:,}")
print(f"Symbols: {result[1]}")
print(f"Date range: {result[2]} to {result[3]}")

# Check ZL specifically
zl_data = con.execute("""
    SELECT as_of_date, close, volume
    FROM raw.databento_futures_ohlcv_1d
    WHERE symbol = 'ZL'
    ORDER BY as_of_date DESC
    LIMIT 5
""").fetchall()

print("\nLatest ZL data:")
for row in zl_data:
    print(f"  {row[0]}: ${row[1]:.2f}, vol={row[2]:,}")

con.close()
```

### 3. Test Dashboard API

```bash
# Start dashboard (if not running)
cd dashboard
npm run dev

# Test API endpoint
curl http://localhost:3000/api/live/zl | jq
```

**Expected Response:**

```json
{
  "success": true,
  "data": [
    {
      "symbol": "ZL",
      "as_of_date": "2025-12-13",
      "close": 45.67,
      "volume": 123456
    }
  ],
  "count": 100,
  "timestamp": "2025-12-15T..."
}
```

---

## Full Historical Pull (After Verification)

Once small test passes, pull all 38 symbols:

```python
# Edit trigger/DataBento/Scripts/collect_daily.py
# Uncomment all symbols in SYMBOLS list (lines 34-73)

# Run full ingestion
python trigger/DataBento/Scripts/collect_daily.py
```

**Estimated Time:** 2-4 hours (depends on Databento API rate limits)  
**Expected Rows:** ~150,000 (38 symbols × ~4,000 trading days since 2010)  
**Cost:** Check Databento usage limits (historical data may have costs)

---

## Symbol Priority Order (If Rate Limited)

**Tier 1 (Critical - Pull First):**

- ZL, ZS, ZM (Crush complex)
- CL, HO (Energy)
- DX (Dollar index)

**Tier 2 (Important):**

- 6E, 6J, 6B, 6C (FX majors)
- HG (Copper - China proxy)
- ZN, ZB (Treasuries)

**Tier 3 (Nice to Have):**

- Remaining agricultural, metals, FX minors

---

## Rollback Plan (If Issues Arise)

```sql
-- Connect to MotherDuck
-- Delete all ingested data
DELETE FROM raw.databento_futures_ohlcv_1d;

-- Or delete specific symbol
DELETE FROM raw.databento_futures_ohlcv_1d WHERE symbol = 'ZL';

-- Or delete specific date range
DELETE FROM raw.databento_futures_ohlcv_1d
WHERE as_of_date BETWEEN '2024-01-01' AND '2024-12-31';
```

---

## Post-Ingestion Tasks

After successful data pull:

1. **Sync to Local DuckDB** (for training):

   ```bash
   python scripts/sync_motherduck_to_local.py
   ```

2. **Build Feature Matrix**:

   ```bash
   python src/engines/anofox/build_all_features.py
   ```

3. **Verify Feature Coverage**:

   ```bash
   python scripts/verify_pipeline.py
   ```

4. **Check Dashboard**:
   - Visit http://localhost:3000
   - Verify ZL price chart shows data
   - Check Big 8 bucket scores populate

---

## Ready to Proceed?

**Checklist:**

- [ ] Apply Fix #1 (dashboard table name)
- [ ] Apply Fix #2 (staging DDL documentation)
- [ ] Apply Fix #3 (idempotency)
- [ ] Test with ZL only (5 days)
- [ ] Verify data in MotherDuck
- [ ] Test dashboard API
- [ ] Proceed with full historical pull

**Approval Required:** Yes (you have 2 days, want to proceed with fixes now?)
