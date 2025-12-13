-- Features: Technical Indicators (All Symbols)
-- RSI, MACD, Bollinger, ATR for 33 symbols

CREATE TABLE IF NOT EXISTS features.tech_indicators (
    date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    -- Moving averages
    sma_5 DECIMAL(10, 2),
    sma_20 DECIMAL(10, 2),
    sma_50 DECIMAL(10, 2),
    sma_200 DECIMAL(10, 2),
    ema_12 DECIMAL(10, 2),
    ema_26 DECIMAL(10, 2),
    -- RSI
    rsi_14 DECIMAL(5, 2),
    rsi_7 DECIMAL(5, 2),
    -- MACD
    macd_line DECIMAL(10, 4),
    macd_signal DECIMAL(10, 4),
    macd_histogram DECIMAL(10, 4),
    -- Bollinger Bands
    bb_upper DECIMAL(10, 2),
    bb_middle DECIMAL(10, 2),
    bb_lower DECIMAL(10, 2),
    bb_width DECIMAL(10, 6),
    bb_pct DECIMAL(5, 4),
    -- ATR
    atr_14 DECIMAL(10, 4),
    atr_21 DECIMAL(10, 4),
    -- Trend
    trend_strength_60d DECIMAL(5, 4),
    -- Volume
    volume_sma_20 BIGINT,
    volume_zscore DECIMAL(6, 4),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_tech_indicators_symbol 
    ON features.tech_indicators(symbol);

