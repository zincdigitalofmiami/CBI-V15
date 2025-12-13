-- Utility Macros
-- Common helper functions for feature engineering

-- Safe division (avoids divide by zero)
CREATE OR REPLACE MACRO safe_divide(numerator, denominator) AS (
    CASE 
        WHEN denominator = 0 OR denominator IS NULL THEN NULL
        ELSE numerator / denominator
    END
);

-- Winsorize outliers at given percentiles
CREATE OR REPLACE MACRO winsorize(value, lower_pct, upper_pct) AS (
    CASE
        WHEN value < PERCENTILE_CONT(lower_pct) WITHIN GROUP (ORDER BY value) OVER ()
        THEN PERCENTILE_CONT(lower_pct) WITHIN GROUP (ORDER BY value) OVER ()
        WHEN value > PERCENTILE_CONT(upper_pct) WITHIN GROUP (ORDER BY value) OVER ()
        THEN PERCENTILE_CONT(upper_pct) WITHIN GROUP (ORDER BY value) OVER ()
        ELSE value
    END
);

-- Z-score normalization
CREATE OR REPLACE MACRO zscore(value) AS (
    (value - AVG(value) OVER ()) / NULLIF(STDDEV(value) OVER (), 0)
);

-- Rolling z-score (lookback only - no leakage)
CREATE OR REPLACE MACRO rolling_zscore(value, window_days) AS (
    (value - AVG(value) OVER (ORDER BY date ROWS BETWEEN window_days PRECEDING AND CURRENT ROW))
    / NULLIF(STDDEV(value) OVER (ORDER BY date ROWS BETWEEN window_days PRECEDING AND CURRENT ROW), 0)
);

-- Percentile rank (0-100)
CREATE OR REPLACE MACRO percentile_rank(value) AS (
    100.0 * PERCENT_RANK() OVER (ORDER BY value)
);

-- Rolling percentile rank (lookback window)
CREATE OR REPLACE MACRO rolling_percentile_rank(value, window_days) AS (
    100.0 * PERCENT_RANK() OVER (
        ORDER BY date 
        ROWS BETWEEN window_days PRECEDING AND CURRENT ROW
    )
);

-- Log return calculation
CREATE OR REPLACE MACRO log_return(current_price, previous_price) AS (
    LN(safe_divide(current_price, previous_price))
);

-- Simple return calculation
CREATE OR REPLACE MACRO simple_return(current_price, previous_price) AS (
    safe_divide(current_price - previous_price, previous_price)
);

-- Exponential moving average weight
CREATE OR REPLACE MACRO ema_weight(span) AS (
    2.0 / (span + 1)
);

-- Clamp value to range
CREATE OR REPLACE MACRO clamp(value, min_val, max_val) AS (
    CASE
        WHEN value < min_val THEN min_val
        WHEN value > max_val THEN max_val
        ELSE value
    END
);

-- Sign function (-1, 0, 1)
CREATE OR REPLACE MACRO sign_direction(value) AS (
    CASE
        WHEN value > 0 THEN 1
        WHEN value < 0 THEN -1
        ELSE 0
    END
);

-- Is weekend check
CREATE OR REPLACE MACRO is_weekend(date_val) AS (
    EXTRACT(DOW FROM date_val) IN (0, 6)
);

-- Trading days between dates (approximate)
CREATE OR REPLACE MACRO approx_trading_days(start_date, end_date) AS (
    FLOOR((end_date - start_date) * 5.0 / 7.0)
);

