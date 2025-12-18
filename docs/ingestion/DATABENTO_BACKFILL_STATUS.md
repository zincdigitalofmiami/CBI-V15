# Databento Backfill Status

**Last Updated:** 2025-12-16 13:50 PST

---

## ✅ WHAT YOU HAVE NOW

### Daily Futures (15+ years)
- **Table:** `raw.databento_futures_ohlcv_1d`
- **Symbols:** 56 (will be 87 when backfill completes)
- **Rows:** 218,994
- **Date Range:** 2010-06-07 to 2025-12-15 (15.5 years)
- **Status:** ✅ ACTIVE

**Current symbols (56):**
- Soy Complex: ZL, ZS, ZM ✅
- Grains: ZC, ZW, ZO, ZR, KE ✅
- Energy: CL, HO, RB, NG, BZ, QM, QG ✅
- Metals: GC, SI, HG, PA, PL, MGC ✅
- Rates: ZT, ZF, ZN, ZB, UB, GE, SR1, SR3 ✅
- Equity: ES, NQ, YM, RTY, MES, MNQ, MYM, M2K ✅
- FX: 6A-6Z, 6R, 6L ✅
- Livestock: HE, LE, GF ✅
- Crypto: BTC, ETH, MBT, MET ✅

---

## ⏳ CURRENTLY BACKFILLING

### 1. Missing 31 Daily Symbols (15-year backfill)
**Script:** `scripts/backfill_missing_symbols.py`  
**Process ID:** 55596  
**Status:** ⏳ IN PROGRESS (Chunk 1/11)  
**ETA:** 60-90 minutes

**Symbols being added (31):**
```
Interest Rate Swaps: 10Y, 2YY, 30Y, 5YY
Rates: ZQ, Z3N, TN, FF
Metals: MSI, QI, QO, ALI
Energy: MCL, QH, QU
FX Micros: M6E, M6A, M6B
Equity: EMD, NIY, GD
Grains: XC, XW, XK, ZE
Livestock/Dairy: DC, DY, DA
Lumber: LBS
Volatility: VX, VXM
```

**Progress:**
- Chunk 1/11: 10Y, 2YY, 30Y (requesting data...)
- Chunks 2-11: Pending

**Monitoring:**
```bash
tail -f /tmp/missing_symbols_backfill.log
```

---

### 2. Hourly Futures (1-year backfill)
**Script:** `src/ingestion/databento/collect_hourly.py`  
**Process ID:** 24525 (check if still running)  
**Status:** ⏳ IN PROGRESS or ✅ COMPLETE  
**Target:** 351 days, 33 GLBX symbols

**Current hourly data:**
- **Rows:** 6,751 (Dec 2-15, 2025)
- **Symbols:** 33
- **Target rows:** ~170,000 (when complete)

**Monitoring:**
```bash
tail -f /tmp/hourly_backfill_retry.log
```

---

## 🎯 FINAL TARGET STATE

### Daily Futures
- **Symbols:** 87 total (56 existing + 31 backfilling)
- **Rows:** ~350,000 (estimated)
- **Coverage:** Full 15-year history for all CME GLBX.MDP3 symbols

### Hourly Futures
- **Symbols:** 33 core symbols
- **Rows:** ~170,000
- **Coverage:** Last 365 days intraday bars

### Options (Ready to test)
- **Table:** `raw.databento_options_ohlcv_1d`
- **Script:** `src/ingestion/databento/collect_options_daily.py`
- **Status:** ⚠️ Schema created, script ready, needs testing

---

## 📊 VALIDATION QUERIES

### Check daily symbol count
```sql
SELECT COUNT(DISTINCT symbol) as symbols,
       COUNT(*) as total_rows
FROM raw.databento_futures_ohlcv_1d;
```

### Check hourly coverage
```sql
SELECT symbol,
       COUNT(*) as bars,
       MIN(ts_event) as earliest,
       MAX(ts_event) as latest
FROM raw.databento_futures_ohlcv_1h
GROUP BY symbol
ORDER BY symbol;
```

### Identify missing symbols
```sql
WITH target_symbols AS (
    SELECT UNNEST(['ZL','ZS','ZM','ZC','ZW',...]) as symbol
)
SELECT target_symbols.symbol
FROM target_symbols
LEFT JOIN (
    SELECT DISTINCT symbol FROM raw.databento_futures_ohlcv_1d
) existing
ON target_symbols.symbol = existing.symbol
WHERE existing.symbol IS NULL;
```

---

## ⚠️ KNOWN ISSUES

1. **504 Gateway Timeouts:**
   - **Cause:** Large requests (31 symbols × 15 years)
   - **Solution:** Chunking to 3 symbols per request with retry logic
   - **Status:** ✅ IMPLEMENTED

2. **ICE Softs Missing:**
   - **Symbols:** KC, SB, CC, CT, OJ, DX
   - **Cause:** Require IFUS.IMPACT subscription (not GLBX.MDP3)
   - **Solution:** Excluded from default symbol list
   - **Status:** ✅ DOCUMENTED

3. **Hourly Process Aborts:**
   - **Cause:** Large DataFrame processing (351 days × 33 symbols at once)
   - **Solution:** Reduced chunk size to 7 days with retry logic
   - **Status:** ✅ FIXED

---

## 🔧 NEXT STEPS

1. **Wait for backfills to complete** (~1-2 hours)
2. **Verify final symbol count:**
   ```bash
   python3 -c "import duckdb,os; con=duckdb.connect('md:cbi_v15?motherduck_token='+os.getenv('MOTHERDUCK_TOKEN')); print('Symbols:', con.execute('SELECT COUNT(DISTINCT symbol) FROM raw.databento_futures_ohlcv_1d').fetchone()[0])"
   ```
3. **Test options ingestion:**
   ```bash
   python3 src/ingestion/databento/collect_options_daily.py --dry-run
   ```
4. **Add FRED spot prices:**
   - Create `src/ingestion/fred/collect_fred_spot_prices.py`
   - Pull 150+ FRED series from user's list
5. **Schedule daily updates:**
   - Daily futures: Run `collect_daily.py` at 1 AM UTC
   - Hourly futures: Run `collect_hourly.py --days 1` every hour
   - Options: Run `collect_options_daily.py` at 2 AM UTC

---

## 📞 MONITORING COMMANDS

### Check all Databento processes
```bash
ps aux | grep -E "(databento|backfill)" | grep python
```

### Check backfill logs
```bash
tail -f /tmp/missing_symbols_backfill.log  # Daily missing symbols
tail -f /tmp/hourly_backfill_retry.log     # Hourly backfill
```

### Check MotherDuck status
```bash
python3 -c "import duckdb,os; con=duckdb.connect('md:cbi_v15?motherduck_token='+os.getenv('MOTHERDUCK_TOKEN')); result=con.execute('SELECT COUNT(DISTINCT symbol), COUNT(*) FROM raw.databento_futures_ohlcv_1d').fetchone(); print(f'{result[0]} symbols, {result[1]:,} rows')"
```

---

**Status will be updated when backfills complete.**



