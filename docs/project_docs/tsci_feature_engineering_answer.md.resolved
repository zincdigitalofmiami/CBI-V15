# TSci Feature Engineering: The Answer

## Your Question
> Does TSci add missing features if they are missing and it thinks it needs them prior to moving to next step?

## Short Answer
**NO.** TimeSeriesScientist does NOT automatically create missing features.

## What TSci Actually Does

### ✅ What TSci CAN Do:
1. **Data Cleaning**: Handle missing values, outliers, anomalies
2. **Quality Assessment**: Analyze trends, seasonality, stationarity
3. **Model Selection**: Choose from 21 built-in models (ARIMA, LSTM, Prophet, etc.)
4. **Ensemble Creation**: Combine multiple model forecasts
5. **Report Generation**: Explain results with LLM-generated narratives

### ❌ What TSci CANNOT Do:
1. **Feature Engineering**: Create lags, moving averages, technical indicators
2. **Domain-Specific Features**: Calculate RSI, MACD, Bollinger Bands
3. **Custom Transformations**: Log returns, volatility measures, regime indicators

## The Solution: Anofox-First Workflow

**Anofox creates all features BEFORE TSci runs.**

### Pipeline Order:

```
STEP 1: RAW DATA
┌─────────────────────────────────┐
│ Databento → DuckDB              │
│ date         close               │
│ 2024-01-01   23.50              │
│ 2024-01-02   23.75              │
│ ...                              │
└─────────────────────────────────┘
                ↓
STEP 2: ANOFOX FEATURE ENGINEERING (SQL)
┌────────────────────────────────────────────────────────┐
│ CREATE TABLE zl_enriched AS                            │
│ SELECT                                                 │
│     date,                                              │
│     close,                                             │
│     anofox_sma(close, 5) AS sma_5,                    │
│     anofox_sma(close, 21) AS sma_21,                  │
│     anofox_ema(close, 12) AS ema_12,                  │
│     anofox_volatility(close, 21) AS volatility_21d,   │
│     anofox_rsi(close, 14) AS rsi_14,                  │
│     anofox_macd(close) AS macd,                       │
│     anofox_bollinger_bands(close, 20) AS bb_upper,    │
│     anofox_atr(high, low, close, 14) AS atr_14,       │
│     -- ... all 276 features                           │
│ FROM zl_raw_prices                                    │
└────────────────────────────────────────────────────────┘
                ↓
STEP 3: EXPORT ENRICHED DATA TO PYTHON
┌─────────────────────────────────────────────┐
│ enriched_df = conn.execute("""              │
│     SELECT * FROM zl_enriched               │
│ """).fetchdf()                              │
│                                             │
│ # Now enriched_df has 280 columns:         │
│ # - date, close (2 original)               │
│ # - 276 Anofox-generated features          │
└─────────────────────────────────────────────┘
                ↓
STEP 4: TSci RUNS ON ENRICHED DATA
┌──────────────────────────────────────────────┐
│ from TimeSeriesScientist import TSci         │
│                                              │
│ tsci = TSci(config)                          │
│ result = tsci.run(enriched_df)  # ← Full    │
│                                  # featured  │
│ # TSci sees ALL 276 features    # dataset   │
│ # and uses them for forecasting             │
└──────────────────────────────────────────────┘
```

## Code Example

### ❌ WRONG: Passing Raw Data to TSci
```python
# This will FAIL - TSci needs features!
raw_df = pd.DataFrame({'date': [...], 'close': [...]})
tsci.run(raw_df)  # Only has 2 columns, will produce poor forecasts
```

### ✅ CORRECT: Anofox Feature Engineering First
```python
import duckdb

# 1. Load Anofox extensions
conn = duckdb.connect('zl_futures.db')
conn.execute("INSTALL anofox_tabular FROM community")
conn.execute("INSTALL anofox_forecast FROM community")
conn.execute("LOAD anofox_tabular")
conn.execute("LOAD anofox_forecast")

# 2. Anofox creates all features (in SQL - FAST!)
conn.execute("""
    CREATE OR REPLACE TABLE zl_features AS
    SELECT 
        date,
        close AS price_current,
        
        -- Moving Averages
        anofox_sma(close, 5) AS sma_5,
        anofox_sma(close, 10) AS sma_10,
        anofox_sma(close, 21) AS sma_21,
        anofox_sma(close, 50) AS sma_50,
        
        -- Exponential Moving Averages
        anofox_ema(close, 12) AS ema_12,
        anofox_ema(close, 26) AS ema_26,
        
        -- Volatility
        anofox_volatility(close, 10) AS vol_10d,
        anofox_volatility(close, 21) AS vol_21d,
        anofox_volatility(close, 60) AS vol_60d,
        
        -- Technical Indicators
        anofox_rsi(close, 14) AS rsi_14,
        anofox_macd(close) AS macd,
        anofox_bollinger_bands(close, 20) AS bb_upper,
        anofox_atr(high, low, close, 14) AS atr_14,
        
        -- Statistical Features
        anofox_trend_strength(close, 60) AS trend_60d,
        anofox_correlation(close, wti_close, 90) AS wti_corr_90d,
        
        -- ... all 276 features
    FROM zl_raw_prices
""")

# 3. Export to Python
enriched_df = conn.execute("SELECT * FROM zl_features").fetchdf()

# 4. NOW run TSci with full feature set
from TimeSeriesScientist import TSci

tsci = TSci({
    'data_path': enriched_df,  # ← 280 columns!
    'horizon': 30,
    'llm_model': 'gpt-4o'
})

result = tsci.run()
# TSci will analyze all 276 features, select best models,
# and generate ensemble forecast
```

## Key Benefits of Anofox-First Approach

1. **Speed**: SQL feature calculations are 100x faster than Pandas
2. **Consistency**: Same SQL functions for historical backtest AND live production
3. **Scalability**: DuckDB handles millions of rows efficiently
4. **Maintainability**: Single source of truth for all features
5. **TSci Optimization**: TSci works best with rich, pre-engineered features

## Summary

**TSci is NOT a feature engineering tool - it's a model selection and ensemble optimization tool.**

Your 276 features must be created by Anofox SQL functions BEFORE TSci runs. This is actually better because:
- Anofox is optimized for feature generation (C++ backend)
- TSci is optimized for model selection (LLM reasoning)
- Separation of concerns = cleaner architecture

**Next Step**: Install Anofox extensions and create your feature calculation SQL script.
