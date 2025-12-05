-- AnoFox Ensemble Forecasting Script
-- Generates forecasts from all 7 buckets and ensembles them based on regime weights.

-- 1. Generate Individual Bucket Forecasts
CREATE OR REPLACE TABLE forecasts_temp AS
SELECT 'biofuel' as bucket, * FROM anofox_predict('model_biofuel')
UNION ALL
SELECT 'china', * FROM anofox_predict('model_china')
UNION ALL
SELECT 'trade', * FROM anofox_predict('model_trade')
UNION ALL
SELECT 'weather', * FROM anofox_predict('model_weather')
UNION ALL
SELECT 'energy', * FROM anofox_predict('model_energy')
UNION ALL
SELECT 'fed', * FROM anofox_predict('model_fed')
UNION ALL
SELECT 'volatility', * FROM anofox_predict('model_volatility');

-- 2. Calculate Regime Weights (Dynamic)
-- Detect current regime using structural break detection
CREATE OR REPLACE TABLE regime_weights_current AS
SELECT 
    CASE 
        WHEN anofox_regime_detect(close, method := 'structural_break') = 'high_vol' THEN 
            STRUCT_PACK(biofuel:=0.1, china:=0.1, trade:=0.1, weather:=0.1, energy:=0.1, fed:=0.1, volatility:=0.4)
        ELSE 
            STRUCT_PACK(biofuel:=0.2, china:=0.2, trade:=0.1, weather:=0.2, energy:=0.1, fed:=0.1, volatility:=0.1)
    END as weights
FROM staging.market_daily 
ORDER BY date DESC LIMIT 1;

-- 3. Ensemble Aggregation
CREATE OR REPLACE TABLE final_forecast AS
SELECT
    f.date,
    SUM(f.forecast_value * 
        CASE f.bucket
            WHEN 'biofuel' THEN w.weights.biofuel
            WHEN 'china' THEN w.weights.china
            WHEN 'trade' THEN w.weights.trade
            WHEN 'weather' THEN w.weights.weather
            WHEN 'energy' THEN w.weights.energy
            WHEN 'fed' THEN w.weights.fed
            WHEN 'volatility' THEN w.weights.volatility
        END
    ) as ensemble_forecast,
    -- Confidence Interval Aggregation (Conservative: Max Width)
    MIN(f.lower_bound) as ensemble_lower,
    MAX(f.upper_bound) as ensemble_upper
FROM forecasts_temp f, regime_weights_current w
GROUP BY f.date;

