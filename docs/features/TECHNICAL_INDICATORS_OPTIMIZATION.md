# Technical Indicators Optimization - 15 Year Initial Load

**Date**: November 28, 2025  
**Challenge**: Calculate all technical indicators for 15 years of price data efficiently  
**Goal**: Reduce compute costs, optimize performance

---

## 🔍 Library Comparison

### Option 1: pandas-ta (Currently in Requirements)

**Pros**:
- ✅ Pure Python (no C dependencies)
- ✅ 100+ indicators built-in
- ✅ Easy to use, pandas-native
- ✅ Good documentation
- ✅ Already in requirements.txt

**Cons**:
- ❌ Slower than TA-Lib (pure Python)
- ❌ Not optimized for large datasets
- ❌ Single-threaded

**Performance**: ~10-15 seconds per symbol per year (15 years = 2.5-3.75 minutes per symbol)

**Cost**: Cloud Functions compute (free tier) or Mac M4 (local)

---

### Option 2: TA-Lib (Industry Standard)

**Pros**:
- ✅ **Fastest** - C library, highly optimized
- ✅ Industry standard (used by GS, JPM, etc.)
- ✅ 150+ indicators
- ✅ Vectorized operations
- ✅ Battle-tested

**Cons**:
- ❌ Requires C library installation
- ❌ macOS installation can be tricky
- ❌ Not pure Python

**Performance**: ~2-3 seconds per symbol per year (15 years = 30-45 seconds per symbol)

**Cost**: Same compute, but 5-10x faster = less compute time

**Installation**:
```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Or use conda
conda install -c conda-forge ta-lib
```

---

### Option 3: DuckDB/MotherDuck SQL UDFs (Recommended for Initial Load)

**Pros**:
- ✅ **Zero compute cost** (runs on DuckDB/MotherDuck, uses query budget)
- ✅ **Vectorized** - processes entire columns at once
- ✅ **Parallel** - DuckDB/MotherDuck parallelizes automatically
- ✅ **No data transfer** - data stays in DuckDB/MotherDuck
- ✅ **Scalable** - handles billions of rows

**Cons**:
- ❌ More complex SQL
- ❌ Limited indicator library (need to implement)
- ❌ Less flexible than Python

**Performance**: Processes 15 years in seconds (parallelized)

**Cost**: Uses DuckDB/MotherDuck query budget (first 1 TB free, then $5/TB)

**Example**:
```sql
-- RSI calculation in DuckDB/MotherDuck
CREATE TEMP FUNCTION calculate_rsi(prices ARRAY<FLOAT64>, period INT64)
RETURNS FLOAT64 AS (
  -- RSI calculation logic
);

-- Apply to entire column
SELECT 
  date,
  symbol,
  close,
  calculate_rsi(ARRAY_AGG(close) OVER (
    ORDER BY date 
    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
  ), 14) AS rsi_14
FROM `cbi-v15.raw.databento_daily_ohlcv`
WHERE symbol = 'ZL'
```

---

### Option 4: vectorbt (Advanced)

**Pros**:
- ✅ Optimized for backtesting
- ✅ Vectorized operations
- ✅ Good for technical indicators
- ✅ Built on NumPy

**Cons**:
- ❌ More complex API
- ❌ Overkill for simple indicators
- ❌ Not as fast as TA-Lib

**Performance**: ~5-8 seconds per symbol per year

---

## 🎯 Recommended Approach: Hybrid Strategy

### Phase 1: Initial Load (15 Years) - DuckDB/MotherDuck SQL UDFs

**Why**: 
- Zero compute cost (uses DuckDB/MotherDuck query budget)
- Fastest for bulk processing
- Parallelized automatically
- No data transfer needed

**Implementation**:
```sql
-- Create DuckDB/MotherDuck UDFs for common indicators
-- RSI, MACD, Bollinger Bands, Moving Averages, etc.
-- Process all 15 years in one query
```

**Cost**: Uses DuckDB/MotherDuck query budget (~50-100 GB for 15 years = FREE within 1 TB limit)

**Time**: Minutes (not hours)

---

### Phase 2: Incremental Updates - pandas-ta (Python)

**Why**:
- Easy to maintain
- Flexible for new indicators
- Runs on Mac M4 (free compute)
- Good for daily updates

**Implementation**:
```python
# Daily updates: Only calculate for new data
# Use pandas-ta for flexibility
```

**Cost**: Mac M4 compute (free)

**Time**: Seconds per day

---

### Phase 3: Complex Indicators - TA-Lib (If Needed)

**Why**:
- If pandas-ta is too slow
- For advanced indicators
- Performance-critical calculations

**Implementation**:
```python
# Use TA-Lib for performance-critical indicators
# Fallback to pandas-ta for others
```

---

## 📊 Performance Comparison

### 15 Years of Data (Single Symbol)

| Method | Time | Cost | Notes |
|--------|------|------|-------|
| **DuckDB/MotherDuck SQL UDFs** | ~2-5 minutes | $0.00 (free tier) | ✅ **BEST** |
| **TA-Lib (Python)** | ~30-45 seconds | Mac M4 (free) | Fast but requires data export |
| **pandas-ta (Python)** | ~2.5-3.75 minutes | Mac M4 (free) | Slower but easier |
| **vectorbt** | ~1.25-2 minutes | Mac M4 (free) | Good middle ground |

### Cost Analysis

**DuckDB/MotherDuck SQL Approach**:
- Query: ~50-100 GB for 15 years
- Cost: $0.00 (within 1 TB free tier) ✅
- Time: 2-5 minutes
- **Total**: **$0.00, 2-5 minutes**

**Python Approach**:
- Export: ~1 GB Parquet
- Compute: Mac M4 (free)
- Time: 30 seconds - 4 minutes
- Upload: ~1 GB back to DuckDB/MotherDuck
- **Total**: **$0.00, 1-5 minutes** (but slower)

---

## 🚀 Recommended Implementation

### Step 1: Create DuckDB/MotherDuck UDFs for Common Indicators

```sql
-- anofox/includes/technical_indicators_udf.sqlx

-- RSI
CREATE TEMP FUNCTION calculate_rsi(prices ARRAY<FLOAT64>, period INT64)
RETURNS FLOAT64 AS (
  -- RSI calculation
);

-- MACD
CREATE TEMP FUNCTION calculate_macd(
  prices ARRAY<FLOAT64>, 
  fast_period INT64, 
  slow_period INT64, 
  signal_period INT64
)
RETURNS STRUCT<macd FLOAT64, signal FLOAT64, histogram FLOAT64> AS (
  -- MACD calculation
);

-- Bollinger Bands
CREATE TEMP FUNCTION calculate_bollinger(
  prices ARRAY<FLOAT64>, 
  period INT64, 
  std_dev FLOAT64
)
RETURNS STRUCT<upper FLOAT64, middle FLOAT64, lower FLOAT64> AS (
  -- Bollinger calculation
);
```

### Step 2: Initial Load Script (DuckDB/MotherDuck)

```sql
-- anofox/definitions/03_features/technical_indicators_initial_load.sqlx

config {
  type: "table",
  schema: "features",
  name: "technical_indicators_15y"
}

SELECT 
  date,
  symbol,
  close,
  
  -- RSI
  calculate_rsi(
    ARRAY_AGG(close) OVER (
      PARTITION BY symbol 
      ORDER BY date 
      ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
    ),
    14
  ) AS rsi_14,
  
  -- MACD
  calculate_macd(
    ARRAY_AGG(close) OVER (
      PARTITION BY symbol 
      ORDER BY date 
      ROWS BETWEEN 25 PRECEDING AND CURRENT ROW
    ),
    12, 26, 9
  ) AS macd,
  
  -- Bollinger Bands
  calculate_bollinger(
    ARRAY_AGG(close) OVER (
      PARTITION BY symbol 
      ORDER BY date 
      ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ),
    20, 2.0
  ) AS bollinger_bands,
  
  -- Moving Averages
  AVG(close) OVER (
    PARTITION BY symbol 
    ORDER BY date 
    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
  ) AS sma_10,
  
  AVG(close) OVER (
    PARTITION BY symbol 
    ORDER BY date 
    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
  ) AS sma_20,
  
  -- ATR
  AVG(high - low) OVER (
    PARTITION BY symbol 
    ORDER BY date 
    ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
  ) AS atr_14
  
FROM `${ref("databento_daily_ohlcv")}`
WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 15 YEAR)
```

## 🎯 Final Recommendation

### For Initial 15-Year Load: **DuckDB/MotherDuck SQL UDFs** ✅

**Why**:
1. ✅ Zero compute cost (uses query budget, within free tier)
2. ✅ Fastest (parallelized, vectorized)
3. ✅ No data transfer (stays in DuckDB/MotherDuck)
4. ✅ Scalable (handles billions of rows)

**Implementation**:
- Create DuckDB/MotherDuck UDFs for common indicators
- Process all 15 years in one query
- Store results in `features.technical_indicators_15y`

### For Daily Updates: **pandas-ta (Python)** ✅

**Why**:
1. ✅ Easy to maintain
2. ✅ Flexible for new indicators
3. ✅ Runs on Mac M4 (free)
4. ✅ Good for incremental updates

**Implementation**:
- Use pandas-ta for daily calculations
- Only process new data
- Append to DuckDB/MotherDuck table

---

## 📋 Implementation Checklist

- [ ] Create DuckDB/MotherDuck UDFs for common indicators (RSI, MACD, Bollinger, MAs, ATR)
- [ ] Create initial load script (process 15 years)
- [ ] Test on single symbol first
- [ ] Run for all symbols
- [ ] Create incremental update script (pandas-ta)
- [ ] Schedule incremental updates (daily)

---

## ✅ Summary

**Best Approach**: **DuckDB/MotherDuck SQL UDFs for initial load** + **pandas-ta for incremental**

**Cost**: **$0.00** (all within free tiers)

**Performance**: **2-5 minutes for 15 years** (vs hours with Python loops)

**Scalability**: **Handles billions of rows** (DuckDB/MotherDuck parallelization)

---

**Last Updated**: November 28, 2025
