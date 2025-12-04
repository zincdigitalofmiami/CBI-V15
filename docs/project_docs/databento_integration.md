# Databento Integration Guide

## Overview

Databento provides institutional-grade market data from CME Globex MDP 3.0, the world's largest derivatives marketplace. We'll use this to replace Yahoo Finance for ZL (Soybean Oil) futures data.

## API Credentials

- **Live Data Key**: `db-8uKak7BPpJejVjqxtJ4xnh9sGWYHE`
- **Dataset**: `GLBX.MDP3` (CME Globex)
- **Symbol**: `ZL` (Soybean Oil Futures)

## Data Feed Details

**CME Globex MDP 3.0** features:
- Full order book with every order event
- Event-driven messaging with Simple Binary Encoding (SBE)
- Optimized for low latency and low bandwidth
- Real-time tick data included with subscription
- Historical data available on usage-based or subscription

## Python Client Setup

### Installation
```bash
pip install databento
```

### Historical API (For Backfill)
```python
import databento as db

client = db.Historical('db-8uKak7BPpJejVjqxtJ4xnh9sGWYHE')

# Fetch 1 year of daily OHLCV for ZL
data = client.timeseries.get_range(
    dataset='GLBX.MDP3',
    symbols=['ZL'],
    schema='ohlcv-1d',  # Daily bars
    start='2024-01-01',
    end='2024-12-31',
    stype_in='continuous'  # Continuous contract
)

# Convert to DataFrame
df = data.to_df()
```

### Live API (For Real-Time)
```python
import databento as db

# For streaming real-time data
client = db.Live('db-8uKak7BPpJejVjqxtJ4xnh9sGWYHE')

# Subscribe to ZL futures
client.subscribe(
    dataset='GLBX.MDP3',
    schema='trades',  # Or 'ohlcv-1m' for 1-minute bars
    symbols=['ZL']
)

for record in client:
    print(record)
```

## MotherDuck Schema Design

### Table: `zl_futures_ohlcv`

```sql
CREATE TABLE zl_futures_ohlcv (
    date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2) NOT NULL,
    volume BIGINT,
    open_interest BIGINT,
    source VARCHAR DEFAULT 'databento',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol)
);

-- Partition by year for performance
-- MotherDuck handles this automatically
```

### Table: `forecasts`

```sql
CREATE TABLE forecasts (
    forecast_date DATE NOT NULL,
    target_date DATE NOT NULL,
    horizon VARCHAR NOT NULL,  -- '1W', '1M', '3M', '6M', '12M'
    p10 DECIMAL(10, 2),  -- Upside (10th percentile)
    p50 DECIMAL(10, 2) NOT NULL,  -- Expected (median)
    p90 DECIMAL(10, 2),  -- Downside (90th percentile)
    model_version VARCHAR,
    confidence_score DECIMAL(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (forecast_date, target_date, horizon)
);
```

### Table: `big_8_drivers`

```sql
CREATE TABLE big_8_drivers (
    date DATE NOT NULL,
    crush_margin DECIMAL(10, 4),
    china_imports DECIMAL(10, 2),
    dollar_index DECIMAL(10, 4),
    fed_policy_rate DECIMAL(5, 2),
    tariff_index DECIMAL(10, 2),
    biofuel_demand DECIMAL(10, 2),
    crude_oil_price DECIMAL(10, 2),
    vix DECIMAL(10, 4),
    PRIMARY KEY (date)
);
```

## Migration from Yahoo Finance

### Before (Yahoo Finance CSV)
```python
url = f"https://query1.finance.yahoo.com/v7/finance/download/ZL=F?..."
response = await fetch(url)
csvText = await response.text()
```

### After (Databento)
```python
import databento as db

client = db.Historical('db-8uKak7BPpJejVjqxtJ4xnh9sGWYHE')
data = client.timeseries.get_range(
    dataset='GLBX.MDP3',
    symbols=['ZL'],
    schema='ohlcv-1d',
    start='2024-01-01',
    end='2024-12-31'
)

# Push to MotherDuck
import duckdb

con = duckdb.connect('md:my_db')  # MotherDuck connection
con.execute("""
    INSERT INTO zl_futures_ohlcv (date, symbol, open, high, low, close, volume)
    SELECT * FROM data_df
""")
```

## Next Steps

1. ✅ Install `databento` on AnoFox (local machine)
2. ✅ Test Historical API connection
3. ✅ Backfill 15 years of ZL data → Parquet → MotherDuck
4. ✅ Set up Live API for daily updates
5. ✅ Create Next.js API route to query MotherDuck
6. ✅ Update Dashboard to display real Databento data

## Vercel Environment Variables

```
DATABENTO_API_KEY=db-8uKak7BPpJejVjqxtJ4xnh9sGWYHE
MOTHERDUCK_TOKEN=<your_motherduck_token>
```

## Notes

- **Real-time included**: No extra cost for live data with CME subscription
- **Historical pricing**: Usage-based or included with subscription
- **SBE encoding**: More efficient than CSV/JSON
- **Full order book**: Every order event (not just aggregated depth)
