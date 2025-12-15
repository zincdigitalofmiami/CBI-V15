-- Features: Technical Indicators (All Symbols)
-- RSI, MACD, Bollinger, ATR for 33 symbols

CREATE TABLE IF NOT EXISTS features.technical_indicators_all_symbols (
    as_of_date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    -- Price & Basic Features
    close DECIMAL(10, 2),
    lag_close_1d DECIMAL(10, 2),
    lag_close_5d DECIMAL(10, 2),
    lag_close_21d DECIMAL(10, 2),
    log_ret_1d DECIMAL(10, 6),
    log_ret_5d DECIMAL(10, 6),
    log_ret_21d DECIMAL(10, 6),
    sma_5 DECIMAL(10, 2),
    sma_10 DECIMAL(10, 2),
    sma_21 DECIMAL(10, 2),
    sma_50 DECIMAL(10, 2),
    sma_200 DECIMAL(10, 2),
    volatility_21d DECIMAL(10, 6),
    -- RSI
    rsi_14 DECIMAL(5, 2),
    -- MACD
    macd DECIMAL(10, 4),
    macd_signal DECIMAL(10, 4),
    macd_histogram DECIMAL(10, 4),
    -- Bollinger Bands
    bb_upper DECIMAL(10, 2),
    bb_middle DECIMAL(10, 2),
    bb_lower DECIMAL(10, 2),
    bb_position DECIMAL(10, 4),
    bb_width_pct DECIMAL(10, 4),
    -- ATR
    atr_14 DECIMAL(10, 4),
    tr_pct DECIMAL(10, 6),
    -- Stochastic
    stoch_k DECIMAL(5, 2),
    stoch_d DECIMAL(5, 2),
    -- Momentum
    roc_10d DECIMAL(10, 6),
    roc_21d DECIMAL(10, 6),
    roc_63d DECIMAL(10, 6),
    momentum_10d DECIMAL(10, 2),
    momentum_21d DECIMAL(10, 2),
    -- Volume
    volume BIGINT,
    avg_volume_21d DECIMAL(15, 2),
    volume_ratio DECIMAL(10, 4),
    volume_zscore DECIMAL(10, 4),
    obv BIGINT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (as_of_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_tech_indicators_symbol 
    ON features.technical_indicators_all_symbols(symbol);
