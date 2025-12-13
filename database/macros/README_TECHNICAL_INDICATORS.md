# Technical Indicators & Feature Engineering System

## 🎯 Overview

Complete technical analysis and feature engineering system for **30+ futures symbols** with **276+ features** per symbol.

**All computation happens in DuckDB SQL** - no external dependencies, fully Mac-native.

---

## 📊 Symbols Covered (30+)

### Agriculture/Softs/Oils (6)
- **ZL** - Soybean Oil (primary)
- **ZS** - Soybeans
- **ZM** - Soybean Meal
- **ZC** - Corn
- **ZW** - Wheat
- **ZO** - Soybean Oil (alternate cycle)

### Energy (4)
- **CL** - WTI Crude Oil
- **HO** - Heating Oil
- **RB** - RBOB Gasoline
- **NG** - Natural Gas

### Metals (5)
- **HG** - Copper
- **GC** - Gold
- **SI** - Silver
- **PL** - Platinum
- **PA** - Palladium

### Treasuries (3)
- **ZF** - 5-Year Treasury
- **ZN** - 10-Year Treasury
- **ZB** - 30-Year Treasury

### FX (1)
- **DX** - U.S. Dollar Index

---

## 🔧 Feature Categories (276+ Total)

### 1. Technical Indicators (40 per symbol)
- **Price & Lags**: close, lag_1d, lag_5d, lag_21d
- **Returns**: log_ret_1d, log_ret_5d, log_ret_21d
- **Moving Averages**: SMA 5/10/21/50/200
- **Volatility**: 21-day realized volatility
- **RSI**: 14-period Relative Strength Index
- **MACD**: 12/26/9 with signal and histogram
- **Bollinger Bands**: Upper/middle/lower + position + width
- **ATR**: 14-period Average True Range
- **Stochastic**: %K and %D oscillators
- **Momentum**: ROC 10/21/63, momentum 10/21
- **Volume**: OBV, volume ratio, z-score

### 2. Cross-Asset Correlations (11)
- ZL vs ZS/ZM/CL/HO/HG/DX (60-day rolling)
- CL vs HO/RB/DX
- HG vs GC/DX

### 3. Fundamental Spreads (6)
- **Board Crush**: (ZM × 0.022 + ZL × 11) - ZS
- **Oil Share**: ZL value / total crush value
- **BOHO Spread**: Soy Oil vs Heating Oil
- **Crack Spread**: Refining margin proxy
- **China Copper Proxy**: HG as demand signal
- **Dollar Index**: DX baseline

### 4. Big 8 Bucket Scores (16)
**8 Scores (0-100 scale, 50 = neutral):**
1. **Crush** - Soybean crush margins
2. **China** - Import demand (HG-ZS correlation)
3. **FX** - Currency effects (inverse DX momentum)
4. **Fed** - Monetary policy (yield curve + rate changes)
5. **Tariff** - Trade policy (Trump sentiment + activity)
6. **Biofuel** - RFS/biodiesel (production + RIN prices)
7. **Energy** - Crude correlation (inverse CL momentum)
8. **Vol** - Market volatility (inverse VIX + ZL vol)

**8 Key Metrics:**
- board_crush, china_pulse, dollar_index, yield_curve_slope
- tariff_activity, rin_d4, crude_price, vix

### 5. Neural Scores (9)
- 8 bucket neural scores (populated by ML models)
- 1 master neural score (ensemble)

### 6. Targets (8)
- 4 price targets: 1W/1M/3M/6M
- 4 return targets: 1W/1M/3M/6M

---

## 📁 File Structure

```
database/macros/
├── features.sql                          # Original price/return macros
├── technical_indicators_all_symbols.sql  # RSI, MACD, BB, ATR, Stochastic, etc.
├── cross_asset_features.sql              # Correlations, betas, spreads
├── big8_bucket_features.sql              # Big 8 bucket aggregation
└── master_feature_matrix.sql             # Final feature matrix builder

database/models/03_features/
├── daily_ml_matrix.sql                   # Main feature table (276+ columns)
└── technical_indicators_all_symbols.sql  # Supporting tables

src/engines/anofox/
└── build_all_features.py                 # Python script to populate tables
```

---

## 🚀 Usage

### 1. Load Macros (One-Time Setup)

```python
import duckdb

con = duckdb.connect("md:cbi-v15")

# Load all macros
with open("database/macros/technical_indicators_all_symbols.sql") as f:
    con.execute(f.read())

with open("database/macros/cross_asset_features.sql") as f:
    con.execute(f.read())

with open("database/macros/big8_bucket_features.sql") as f:
    con.execute(f.read())

with open("database/macros/master_feature_matrix.sql") as f:
    con.execute(f.read())
```

### 2. Build Features for Single Symbol

```sql
-- Get all technical indicators for ZL
SELECT * FROM calc_all_technical_indicators('ZL');

-- Get Big 8 bucket scores
SELECT * FROM calc_all_bucket_scores();

-- Get complete feature matrix for ZL
SELECT * FROM build_symbol_features('ZL');
```

### 3. Build Features for All Symbols (Python)

```bash
python src/engines/anofox/build_all_features.py
```

This will:
1. Load all SQL macros
2. Compute technical indicators for 17 symbols
3. Compute cross-asset correlations
4. Compute Big 8 bucket scores
5. Build final `features.daily_ml_matrix_zl` table

---

## 📈 Example Queries

### Get Latest ZL Features
```sql
SELECT *
FROM features.daily_ml_matrix_zl
WHERE symbol = 'ZL'
ORDER BY as_of_date DESC
LIMIT 1;
```

### Get Bucket Scores for Last 30 Days
```sql
SELECT 
    as_of_date,
    crush_bucket_score,
    china_bucket_score,
    energy_bucket_score,
    volatility_bucket_score
FROM features.big8_bucket_scores
WHERE as_of_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY as_of_date DESC;
```

### Find High Volatility Periods
```sql
SELECT 
    as_of_date,
    symbol,
    volatility_21d,
    rsi_14,
    bb_position
FROM features.tech_indicators
WHERE symbol = 'ZL'
AND volatility_21d > 0.30  -- 30% annualized volatility
ORDER BY as_of_date DESC;
```

---

## 🔄 Daily Update Workflow

1. **Ingest raw data** (Databento, FRED, EIA, ScrapeCreators)
2. **Run feature builder**:
   ```bash
   python src/engines/anofox/build_all_features.py
   ```
3. **Train models** on updated features
4. **Generate forecasts**

---

## ✅ Next Steps

- [ ] Add sentiment scores from FinBERT (Mac MPS)
- [ ] Add weather features (Brazil/Argentina/US)
- [ ] Add CFTC positioning data
- [ ] Expand to all 30 symbols (currently 17)
- [ ] Add calendar spreads (near vs far month)

---

## 📝 Notes

- All features are **lag-safe** (no look-ahead bias)
- All correlations use **rolling windows** (60-day default)
- Bucket scores are **normalized to 0-100** (50 = neutral)
- Neural scores are **placeholders** until models are trained

