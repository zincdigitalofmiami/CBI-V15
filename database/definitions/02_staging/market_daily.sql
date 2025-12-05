-- Aligned OHLCV with trading calendar
CREATE TABLE IF NOT EXISTS staging.ohlcv_daily (
    as_of_date    DATE NOT NULL,
    symbol        TEXT NOT NULL,
    open          DOUBLE,
    high          DOUBLE,
    low           DOUBLE,
    close         DOUBLE,
    volume        BIGINT,
    is_trading_day BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (as_of_date, symbol)
);
