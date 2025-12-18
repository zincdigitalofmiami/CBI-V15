# Databento Data Segmentation Architecture

**Last Updated:** 2025-12-16  
**Status:** ✅ IMPLEMENTED

---

## Overview

Databento GLBX.MDP3 data is segmented by **asset type** (spot/futures/options) and **resolution** (daily/hourly) to optimize storage, query performance, and cost control.

---

## Segmentation Strategy

### 1. **Futures Data (Primary)**

| Table                              | Resolution | Symbols | Timeframe    | Purpose                  |
| ---------------------------------- | ---------- | ------- | ------------ | ------------------------ |
| `raw.databento_futures_ohlcv_1d`  | Daily      | 56      | 15+ years    | Core training (all)      |
| `raw.databento_futures_ohlcv_1h`  | Hourly     | 33      | 1 year       | Intraday volatility      |

**Coverage:**
- **Daily (1d):** ALL symbols from GLBX.MDP3 (CME/CBOT/NYMEX/COMEX)
  - Agricultural: ZL, ZS, ZM, ZC, ZW, KE, ZO, ZR, HE, LE, GF, LB
  - Energy: CL, BZ, HO, RB, NG, QM, QG
  - Metals: GC, SI, HG, PA, PL, MGC
  - Rates: ZT, ZF, ZN, ZB, UB, GE, SR1, SR3
  - Equity: ES, NQ, RTY, YM, MES, MNQ, M2K, MYM
  - FX: 6A, 6B, 6C, 6E, 6J, 6L, 6M, 6N, 6P, 6R, 6S, 6W, 6Z
  - Crypto: BTC, ETH, MBT, MET
  
- **Hourly (1h):** CORE symbols only (for intraday signals)
  - Soy complex: ZL, ZS, ZM
  - Grains: ZC, ZW
  - Energy: CL, HO, RB, NG
  - Metals: GC, SI, HG
  - Rates: ZN, ZB, ZF
  - Equity: ES, NQ
  - FX: 6E, 6J, 6B, 6C

**Exclusions:**
- ❌ ICE softs (KC, SB, CC, CT, OJ) - require IFUS.IMPACT subscription
- ❌ DX (Dollar Index) - appears to be ICE, not GLBX

---

### 2. **Options Data (NEW)**

| Table                               | Resolution | Symbols | Timeframe | Purpose                     |
| ----------------------------------- | ---------- | ------- | --------- | --------------------------- |
| `raw.databento_options_ohlcv_1d`   | Daily      | 19      | 1+ years  | Volatility surface, greeks  |

**Coverage:** Symbols with active options markets
- Soy complex: ZL, ZS, ZM
- Grains: ZC, ZW
- Energy: CL, NG, HO, RB
- Metals: GC, SI, HG
- Rates: ZN, ZB, ZF
- Equity: ES, NQ
- FX: 6E, 6J

**Schema:**
- Full options chain per underlying (all strikes, all expirations)
- Greeks (delta, gamma, theta, vega) if available from Databento
- Implied volatility
- Strike price, expiration date, option type (C/P)

**Note:** Options require specific instrument ID lookups or parent filtering. Initial implementation may be incremental.

---

### 3. **Spot Prices (FRED Integration)**

| Source  | Table                  | Symbols              | Timeframe | Purpose               |
| ------- | ---------------------- | -------------------- | --------- | --------------------- |
| FRED    | `raw.fred_spot_prices` | Commodity spot rates | 15+ years | Futures-spot spreads  |

**Coverage:**
- DCOILWTICO - WTI Crude Oil Spot
- GOLDAMGBD228NLBM - Gold Spot (London PM Fix)
- DHHNGSP - Henry Hub Natural Gas Spot
- COPPER - Copper Spot
- SILVER - Silver Spot
- CORN - Corn Spot (USDA/CBOT reference)
- SOYBEAN - Soybean Spot (USDA reference)

**Purpose:** Calculate futures-spot basis for carry trades and convergence signals.

---

## Data Flow

```
┌─────────────────────┐
│  Databento API      │
│  GLBX.MDP3          │
└──────┬──────────────┘
       │
       ├──► Futures Daily (all 56 symbols) → raw.databento_futures_ohlcv_1d
       │    └─► 15+ years of history
       │
       ├──► Futures Hourly (core 33 symbols) → raw.databento_futures_ohlcv_1h
       │    └─► Last 365 days (chunked 7-day pulls)
       │
       └──► Options Daily (19 symbols) → raw.databento_options_ohlcv_1d
            └─► Full chains (all strikes/expiries)

┌─────────────────────┐
│  FRED API           │
└──────┬──────────────┘
       │
       └──► Spot Prices → raw.fred_spot_prices
            └─► 15+ years of history
```

---

## Ingestion Scripts

| Script                                                       | Frequency       | Target Table                          | Status      |
| ------------------------------------------------------------ | --------------- | ------------------------------------- | ----------- |
| `src/ingestion/databento/collect_daily.py`                   | Daily (1 AM)    | `raw.databento_futures_ohlcv_1d`      | ✅ ACTIVE   |
| `src/ingestion/databento/collect_hourly.py`                  | Hourly          | `raw.databento_futures_ohlcv_1h`      | ✅ ACTIVE   |
| `src/ingestion/databento/collect_options_daily.py`           | Daily (2 AM)    | `raw.databento_options_ohlcv_1d`      | ⚠️ SETUP    |
| `src/ingestion/fred/collect_fred_spot_prices.py`             | Daily (1 AM)    | `raw.fred_spot_prices`                | ⚠️ NEEDED   |

---

## Storage Optimization

### Why Segment?

1. **Granularity mismatch:**
   - Daily: DATE column (efficient for 15 years)
   - Hourly: TIMESTAMP column (precise intraday)
   
2. **Volume control:**
   - Hourly is ~252 trading days × 6.5h × 4 bars/h = ~6,500 bars/year/symbol
   - Daily is ~252 bars/year/symbol
   - Mixing them bloats daily queries by 26x

3. **Precision requirements:**
   - Daily: 2 decimal places (`DECIMAL(10, 2)`)
   - Hourly: 4 decimal places (`DECIMAL(10, 4)`) for tick precision

4. **Query performance:**
   - Feature macros read daily table only (training)
   - Volatility models read hourly table only (regime detection)
   - No join overhead

5. **Cost control:**
   - Hourly limited to last 365 days (rolling window)
   - Daily keeps full 15-year history
   - Options chains pruned to near-ATM strikes after 90 days

---

## Backfill Strategy

### Futures Daily (15 years)
- ✅ COMPLETE: 218,994 rows, 56 symbols, 2010-06-07 to 2025-12-15
- Incremental: Resumes from `MAX(as_of_date)`
- Full refresh: Re-run with `--days 5475` (15 years)

### Futures Hourly (1 year)
- ✅ IN PROGRESS: 51 chunks of 7 days (retry logic for 504 errors)
- Expected: ~170,000 rows (33 symbols × 365 days × 15.6 bars/day avg)
- Chunking: 7-day windows to avoid gateway timeouts
- Retry: Exponential backoff for transient API errors

### Options Daily
- ⚠️ SETUP REQUIRED: Databento options API requires instrument ID lookups
- Strategy: Request full chains for front 2 expiries + near-ATM strikes
- Backfill: Start with last 90 days, then extend to 1 year

---

## Future Enhancements

1. **Options filtering:**
   - Store only front 2 expiries + 5 strikes ATM ±2 (reduces storage by ~90%)
   - Archive deep OTM strikes after 30 days

2. **Hourly compression:**
   - After 365 days, aggregate hourly → 4h bars (6 bars/day)
   - Keeps 2+ years of intraday data without storage explosion

3. **Spot price expansion:**
   - Add Bloomberg spot APIs (BCOM commodities)
   - Add EIA spot prices (petroleum products)

4. **Real-time updates:**
   - Databento Live API for sub-second updates (ZL only)
   - Stream to `raw.databento_live_ticks` (Redis buffer → DuckDB hourly flush)

---

## Validation Queries

```sql
-- Check daily futures coverage
SELECT COUNT(*) as rows, COUNT(DISTINCT symbol) as symbols, 
       MIN(as_of_date) as earliest, MAX(as_of_date) as latest
FROM raw.databento_futures_ohlcv_1d;

-- Check hourly futures coverage
SELECT COUNT(*) as rows, COUNT(DISTINCT symbol) as symbols,
       MIN(ts_event) as earliest, MAX(ts_event) as latest
FROM raw.databento_futures_ohlcv_1h;

-- Check options coverage (after ingestion)
SELECT symbol, COUNT(DISTINCT contract_symbol) as contracts,
       COUNT(*) as rows
FROM raw.databento_options_ohlcv_1d
GROUP BY symbol
ORDER BY symbol;

-- Check spot prices (after ingestion)
SELECT series_id, COUNT(*) as rows, MIN(date) as earliest, MAX(date) as latest
FROM raw.fred_spot_prices
GROUP BY series_id;
```

---

## Summary

✅ **Futures Daily:** 15+ years, all GLBX symbols  
✅ **Futures Hourly:** 1 year, core symbols (intraday vol)  
⚠️ **Options Daily:** Setup in progress (volatility surface)  
⚠️ **Spot Prices:** FRED integration needed (basis trades)

**Storage:** ~220k daily + ~170k hourly + ~50k options = **~440k total rows**  
**Cost:** Databento usage-based (~$50-200/mo depending on request volume)  
**Maintenance:** Incremental daily updates + rolling hourly window



