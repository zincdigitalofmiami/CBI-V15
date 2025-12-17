# Anofox API Reference

**Date:** December 3, 2024  
**Status:** Installed and verified in DuckDB

---

## Installation

All three Anofox extensions are installed and loaded:

```sql
INSTALL anofox_tabular FROM community;
INSTALL anofox_forecast FROM community;
INSTALL anofox_statistics FROM community;

LOAD anofox_tabular;
LOAD anofox_forecast;
LOAD anofox_statistics;
```

**Status:** ✅ All extensions installed successfully

---

## Anofox Tabular Functions

**Purpose:** Data quality and anomaly detection

### `anofox_gap_fill()`

Fill gaps in time series data.

**Parameters:**

- `date`: Date column
- `value`: Value column to fill
- `method`: Fill method ('linear', 'forward', 'backward')
- `max_gap`: Maximum gap size (e.g., '5 days')

**Example:**

```sql
SELECT anofox_gap_fill(
    date, close,
    method := 'linear',
    max_gap := '5 days'
) FROM raw.zl_prices;
```

### `anofox_outlier_detect()`

Detect outliers in data.

**Parameters:**

- `column`: Column to analyze
- `method`: Detection method ('zscore', 'isolation_forest')
- `threshold`: Threshold value (e.g., 3.0 for zscore)

**Example:**

```sql
SELECT anofox_outlier_detect(
    close,
    method := 'zscore',
    threshold := 3.0
) FROM raw.zl_prices;
```

---

## Anofox Statistics Functions

**Purpose:** Feature engineering and statistical calculations

### `anofox_volatility()`

Calculate volatility.

**Parameters:**

- `column`: Price column
- `window`: Window size (e.g., 21)

**Example:**

```sql
SELECT anofox_volatility(close, window := 21) AS vol_21d
FROM raw.zl_prices;
```

### `anofox_trend_strength()`

Calculate trend strength.

**Parameters:**

- `column`: Price column
- `window`: Window size (e.g., 60)

**Example:**

```sql
SELECT anofox_trend_strength(close, window := 60) AS trend_60d
FROM raw.zl_prices;
```

### `anofox_sma()`

Simple moving average.

**Parameters:**

- `column`: Price column
- `period`: Period (e.g., 5, 20, 50)

**Example:**

```sql
SELECT anofox_sma(close, 5) AS sma_5
FROM raw.zl_prices;
```

### `anofox_rsi()`

Relative Strength Index.

**Parameters:**

- `column`: Price column
- `period`: Period (e.g., 14)

**Example:**

```sql
SELECT anofox_rsi(close, 14) AS rsi_14
FROM raw.zl_prices;
```

### `anofox_correlation()`

Calculate correlation between two series.

**Parameters:**

- `series1`: First series
- `series2`: Second series
- `window`: Window size (e.g., 90)

**Example:**

```sql
SELECT anofox_correlation(zl_close, wti_close, window := 90) AS zl_wti_corr
FROM staging.market_daily;
```

---

## Anofox Forecast Functions

**Purpose:** Time-series forecasting

### `TS_FORECAST()`

Generate forecasts using statistical methods.

**Parameters:**

- `data`: Subquery or table with date and value columns
- `date_column`: Name of date column
- `value_column`: Name of value column
- `method`: Forecast method ('AutoETS', 'ARIMA', 'Prophet', 'TBATS')
- `horizon`: Forecast horizon (number of periods)

**Example:**

```sql
SELECT TS_FORECAST(
    (SELECT date, close FROM raw.zl_prices),
    'date', 'close',
    method := 'AutoETS',
    horizon := 30
) AS forecast;
```

---

## Integration with Training Pipeline

**Usage Pattern:**

1. Data preprocessing → `anofox_gap_fill()`, `anofox_outlier_detect()`
2. Feature engineering → `anofox_volatility()`, `anofox_trend_strength()`, `anofox_sma()`, `anofox_rsi()`
3. Forecasting → `TS_FORECAST()` with various methods (legacy SQL-based forecasting)
4. Quality checks → `anofox_forecast_quality()` (if available)

**See:** `src/engines/anofox/anofox_bridge.py` for Python wrapper implementation.

---

**Last Updated:** December 3, 2024
