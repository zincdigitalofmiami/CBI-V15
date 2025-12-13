-- AnoFox SQL Guards
-- Data quality checks that fail fast if structural breaks are detected.
-- Prevents garbage data from wasting Mac M4 training time.

-- Guard #1: Check for structural breaks (price anomalies)
CREATE OR REPLACE FUNCTION anofox_check_anomaly(
    price_series DOUBLE[],
    confidence_level DOUBLE DEFAULT 0.99
) RETURNS BOOLEAN AS $$
    -- Detect if recent price is beyond N standard deviations
    -- Used to fail ingestion if data is corrupt
    SELECT 
        ABS(price_series[ARRAY_LENGTH(price_series)] - AVG(price_series)) 
        <= confidence_level * STDDEV(price_series)
$$;

-- Guard #2: Check for sufficient data coverage
CREATE OR REPLACE FUNCTION anofox_check_coverage(
    table_name VARCHAR,
    min_rows INTEGER DEFAULT 100,
    max_days_gap INTEGER DEFAULT 7
) RETURNS TABLE(
    status VARCHAR,
    row_count INTEGER,
    max_gap_days INTEGER,
    min_date DATE,
    max_date DATE
) AS $$
    SELECT 
        CASE 
            WHEN COUNT(*) < min_rows THEN 'INSUFFICIENT_DATA'
            WHEN MAX(date_gap) > max_days_gap THEN 'DATA_GAP_DETECTED'
            ELSE 'PASS'
        END as status,
        COUNT(*)::INTEGER as row_count,
        MAX(date_gap)::INTEGER as max_gap_days,
        MIN(as_of_date) as min_date,
        MAX(as_of_date) as max_date
    FROM (
        SELECT 
            as_of_date,
            LEAD(as_of_date) OVER (ORDER BY as_of_date) - as_of_date as date_gap
        FROM table_name
    )
$$;

-- Guard #3: Check for NaN/NULL pollution
CREATE OR REPLACE FUNCTION anofox_check_nulls(
    table_name VARCHAR,
    critical_columns VARCHAR[],
    max_null_pct DOUBLE DEFAULT 0.05
) RETURNS TABLE(
    column_name VARCHAR,
    null_count INTEGER,
    null_pct DOUBLE,
    status VARCHAR
) AS $$
    SELECT 
        column_name,
        SUM(CASE WHEN column_value IS NULL THEN 1 ELSE 0 END)::INTEGER as null_count,
        AVG(CASE WHEN column_value IS NULL THEN 1.0 ELSE 0.0 END) as null_pct,
        CASE 
            WHEN AVG(CASE WHEN column_value IS NULL THEN 1.0 ELSE 0.0 END) > max_null_pct 
            THEN 'FAIL'
            ELSE 'PASS'
        END as status
    FROM table_name
    UNPIVOT (column_value FOR column_name IN (SELECT UNNEST(critical_columns)))
    GROUP BY column_name
$$;

-- Guard #4: Weekly→Daily Fill (CRITICAL for Biofuel bucket)
-- Carries forward weekly values to daily frequency
-- Prevents AutoGluon from dropping 80% of rows due to NaN
CREATE OR REPLACE MACRO fill_weekly_to_daily(
    daily_table, 
    weekly_table, 
    date_col, 
    value_col
) AS TABLE (
    SELECT 
        d.as_of_date,
        LAST_VALUE(w.value_col IGNORE NULLS) OVER (
            ORDER BY d.as_of_date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as value_col_filled
    FROM daily_table d
    LEFT JOIN weekly_table w ON d.date_col = w.week_ending_date
);

-- Example usage in big8_bucket_features.sql:
-- SELECT * FROM fill_weekly_to_daily(
--     raw.databento_futures_ohlcv_1d,
--     raw.epa_rin_prices,
--     'as_of_date',
--     'rin_d4_price'
-- );













