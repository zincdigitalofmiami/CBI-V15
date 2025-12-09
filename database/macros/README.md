# Macros - Reusable SQL Functions

## Purpose
SQL macros and user-defined functions (UDFs) for feature engineering and data manipulation.

## What Belongs Here
- `features.sql` - Feature calculation macros (TS_*, moving averages, etc.)
- `eda.sql` - Exploratory data analysis helpers
- `prep.sql` - Data preparation utilities (gap filling, outlier removal)

## Pattern
DuckDB macro syntax:
```sql
CREATE OR REPLACE MACRO TS_SMA(values, window_size) AS (
    -- Simple moving average
    AVG(values) OVER (ORDER BY trade_date ROWS BETWEEN window_size - 1 PRECEDING AND CURRENT ROW)
);
```

## Usage
Macros are loaded by `00_init` and available in all subsequent SQL.

## Current Macros
- `features.sql` - TS_SMA, TS_EMA, TS_RSI, TS_MACD, Big 8 aggregations

