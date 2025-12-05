-- AnoFox Bucket Training Script
-- Trains 7 specialized models for each bucket using AutoML selection.

-- 1. Biofuel Policy Bucket
CALL anofox_train_forecast(
    model_name => 'model_biofuel',
    table_name => 'staging.bucket_biofuel',
    target_column => 'price_impact_score',
    horizon => 30,
    models => ['AutoETS', 'AutoARIMA', 'MSTL']
);

-- 2. China Demand Bucket
CALL anofox_train_forecast(
    model_name => 'model_china',
    table_name => 'staging.bucket_china',
    target_column => 'demand_volume',
    horizon => 30,
    models => ['AutoETS', 'TBATS'] -- TBATS good for complex seasonality
);

-- 3. Trade Relations Bucket
CALL anofox_train_forecast(
    model_name => 'model_trade',
    table_name => 'staging.bucket_trade',
    target_column => 'tariff_impact',
    horizon => 30,
    models => ['AutoETS']
);

-- 4. Weather/Supply Bucket
CALL anofox_train_forecast(
    model_name => 'model_weather',
    table_name => 'staging.bucket_weather',
    target_column => 'yield_impact',
    horizon => 30,
    models => ['AutoARIMA', 'Prophet'] -- Prophet handles weather well
);

-- 5. Energy Markets Bucket
CALL anofox_train_forecast(
    model_name => 'model_energy',
    table_name => 'staging.bucket_energy',
    target_column => 'crude_correlation',
    horizon => 30,
    models => ['AutoETS', 'Theta']
);

-- 6. Fed Policy Bucket
CALL anofox_train_forecast(
    model_name => 'model_fed',
    table_name => 'staging.bucket_fed',
    target_column => 'rate_impact',
    horizon => 30,
    models => ['AutoARIMA']
);

-- 7. Volatility Regime Bucket
CALL anofox_train_forecast(
    model_name => 'model_volatility',
    table_name => 'staging.bucket_volatility',
    target_column => 'vix_level',
    horizon => 30,
    models => ['GARCH', 'AutoETS'] -- GARCH for volatility
);

