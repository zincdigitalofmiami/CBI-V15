# TSci Feature Engineering: The Answer

## Your Question
> Does TSci add missing features if they are missing and it thinks it needs them prior to moving to next step?

## Short Answer
**NO.** TimeSeriesScientist does NOT automatically create missing features.

## What TSci Actually Does

### ✅ What TSci CAN Do:
1. **Data Cleaning**: Handle missing values, outliers, anomalies
2. **Quality Assessment**: Analyze trends, seasonality, stationarity
3. **Model Selection**: Choose from 21 built-in models (ARIMA, LSTM, Prophet, etc.)
4. **Ensemble Creation**: Combine multiple model forecasts
5. **Report Generation**: Explain results with LLM-generated narratives

### ❌ What TSci CANNOT Do:
1. **Feature Engineering**: Create lags, moving averages, technical indicators
2. **Domain-Specific Features**: Calculate RSI, MACD, Bollinger Bands
3. **Custom Transformations**: Log returns, volatility measures, regime indicators

## The Solution: AnoFox-First Workflow

**AnoFox creates all features BEFORE TSci runs.**

### Pipeline Order:

```
STEP 1: RAW DATA (Ingestion)
┌─────────────────────────────────┐
│ Databento → DuckDB (raw schema) │
│ date, symbol, close             │
└─────────────────────────────────┘
                ↓
STEP 2: ANOFOX FEATURE ENGINEERING (SQL Macros)
┌─────────────────────────────────────────────────────┐
│ database/macros/*.sql → features schema             │
│ 93+ features per symbol × 38 symbols                │
└─────────────────────────────────────────────────────┘
                ↓
STEP 3: EXPORT TO PYTHON
┌─────────────────────────────────────────────────────┐
│ enriched_df = conn.execute(                         │
│     "SELECT * FROM build_symbol_features('ZL')"     │
│ ).fetchdf()                                         │
└─────────────────────────────────────────────────────┘
                ↓
STEP 4: TSci RUNS ON ENRICHED DATA
┌─────────────────────────────────────────────────────┐
│ TSci sees all 93+ features, selects best models,    │
│ generates ensemble forecast                         │
└─────────────────────────────────────────────────────┘
```

## Feature Inventory (93+ Features Per Symbol)

### Price & Technical Indicators (40 features)
Built by `database/macros/features.sql` and `technical_indicators_all_symbols.sql`:

| Category | Features | Count |
|----------|----------|-------|
| **Price/Close** | `close` | 1 |
| **Lags** | `lag_close_1d`, `lag_close_2d`, `lag_close_3d`, `lag_close_5d`, `lag_close_10d`, `lag_close_21d` | 6 |
| **Log Returns** | `log_ret_1d`, `log_ret_2d`, `log_ret_3d`, `log_ret_5d`, `log_ret_10d`, `log_ret_21d` | 6 |
| **Moving Averages** | `sma_5`, `sma_10`, `sma_21`, `sma_50`, `sma_200` | 5 |
| **MA Distances** | `dist_sma_5`, `dist_sma_10`, `dist_sma_21`, `dist_sma_50`, `dist_sma_200` | 5 |
| **Volatility** | `volatility_21d` | 1 |
| **Volume** | `avg_volume_21d`, `volume_ratio`, `volume_zscore`, `obv` | 4 |
| **Range** | `daily_range_pct` | 1 |
| **RSI** | `rsi_14` | 1 |
| **MACD** | `macd`, `macd_signal`, `macd_histogram` | 3 |
| **Bollinger** | `bb_upper`, `bb_middle`, `bb_lower`, `bb_position`, `bb_width_pct` | 5 |
| **ATR** | `atr_14`, `tr_pct` | 2 |
| **Stochastic** | `stoch_k`, `stoch_d` | 2 |
| **Momentum** | `roc_10d`, `roc_21d`, `roc_63d`, `momentum_10d`, `momentum_21d` | 5 |

### Cross-Asset Correlations (11 features)
Built by `database/macros/cross_asset_features.sql`:

| Feature | Description |
|---------|-------------|
| `corr_zl_zs_60d` | ZL-Soybean correlation |
| `corr_zl_zm_60d` | ZL-Meal correlation |
| `corr_zl_cl_60d` | ZL-Crude correlation |
| `corr_zl_ho_60d` | ZL-Heating Oil correlation |
| `corr_zl_hg_60d` | ZL-Copper correlation |
| `corr_zl_dx_60d` | ZL-Dollar correlation |
| `corr_cl_ho_60d` | Crude-HO correlation |
| `corr_cl_rb_60d` | Crude-RBOB correlation |
| `corr_cl_dx_60d` | Crude-Dollar correlation |
| `corr_hg_gc_60d` | Copper-Gold correlation |
| `corr_hg_dx_60d` | Copper-Dollar correlation |

### Fundamental Spreads (6 features)
Built by `calc_fundamental_spreads()`:

| Feature | Formula |
|---------|---------|
| `board_crush_spread` | (ZM × 0.022 + ZL × 11) - ZS |
| `oil_share_of_crush` | (ZL × 11) / total crush value |
| `boho_spread` | (ZL/100 × 7.5) - HO |
| `crack_spread` | ((RB + HO) / 2) - CL |
| `china_copper_proxy` | HG close |
| `dollar_index` | DX close |

### Big 8 Bucket Scores (16 features)
Built by `database/macros/big8_bucket_features.sql`:

| Bucket | Score Feature | Key Metrics |
|--------|---------------|-------------|
| **Crush** | `crush_bucket_score` | `board_crush`, `oil_share`, `zl_spec_net_pct` |
| **China** | `china_bucket_score` | `china_pulse`, `copper_momentum` |
| **FX** | `fx_bucket_score` | `dollar_index`, `dollar_momentum` |
| **Fed** | `fed_bucket_score` | `yield_curve_slope`, `fed_rate_change_21d` |
| **Tariff** | `tariff_bucket_score` | `tariff_activity`, `trump_sentiment_7d` |
| **Biofuel** | `biofuel_bucket_score` | `rin_d4`, `biodiesel_momentum` |
| **Energy** | `energy_bucket_score` | `crude_price`, `boho_momentum` |
| **Volatility** | `volatility_bucket_score` | `vix`, `zl_volatility` |

### Targets (8 features)
Built by `feat_targets_block()`:

| Feature | Description |
|---------|-------------|
| `target_price_1w` | Close in 5 trading days |
| `target_price_1m` | Close in 21 trading days |
| `target_price_3m` | Close in 63 trading days |
| `target_price_6m` | Close in 126 trading days |
| `target_ret_1w` | Log return over 5 days |
| `target_ret_1m` | Log return over 21 days |
| `target_ret_3m` | Log return over 63 days |
| `target_ret_6m` | Log return over 126 days |

### CFTC COT Features (12+ features per symbol)
Built by `database/macros/big8_cot_enhancements.sql`:

| Feature | Description |
|---------|-------------|
| `managed_money_net_pct_oi` | Speculator positioning |
| `prod_merc_net_pct_oi` | Commercial hedger positioning |
| `spec_hedger_spread` | Divergence signal |
| `extreme_positioning_signal` | Crowded trade reversal |
| `smart_money_signal` | Commercial vs spec divergence |

## Total Feature Count

| Category | Count |
|----------|-------|
| Price & Technical | 40 |
| Cross-Asset Correlations | 11 |
| Fundamental Spreads | 6 |
| Big 8 Bucket Scores | 16 |
| Targets | 8 |
| CFTC COT (varies by symbol) | 12+ |
| **Total Per Symbol** | **93+** |

### Across All Symbols
- 38 futures symbols × 93+ features = **3,500+ total features**
- Not all features apply to all symbols (e.g., CFTC COT only for covered contracts)

## SQL Macro Entry Points

```sql
-- Get all features for a single symbol
SELECT * FROM build_symbol_features('ZL');

-- Get all features for all symbols
SELECT * FROM build_all_symbols_features();

-- Get Big 8 bucket scores only
SELECT * FROM calc_all_bucket_scores();

-- Get technical indicators only
SELECT * FROM calc_all_technical_indicators('ZL');
```

## Python Integration

```python
import duckdb

# Connect to DuckDB
conn = duckdb.connect('Data/duckdb/cbi_v15.duckdb')

# Load AnoFox SQL macros
conn.execute(open('database/macros/features.sql').read())
conn.execute(open('database/macros/big8_bucket_features.sql').read())
conn.execute(open('database/macros/cross_asset_features.sql').read())
conn.execute(open('database/macros/technical_indicators_all_symbols.sql').read())
conn.execute(open('database/macros/master_feature_matrix.sql').read())

# Build feature matrix for ZL
enriched_df = conn.execute("SELECT * FROM build_symbol_features('ZL')").fetchdf()

# Now enriched_df has 93+ columns - pass to TSci
from TimeSeriesScientist import TSci

tsci = TSci({
    'data_path': enriched_df,
    'horizon': 30,
    'llm_model': 'gpt-4o'
})
result = tsci.run()
```

## Key Files

| File | Purpose |
|------|---------|
| `database/macros/features.sql` | Price, returns, volatility, targets |
| `database/macros/technical_indicators_all_symbols.sql` | RSI, MACD, Bollinger, etc. |
| `database/macros/cross_asset_features.sql` | Correlations, betas, spreads |
| `database/macros/big8_bucket_features.sql` | Big 8 thematic scores |
| `database/macros/big8_cot_enhancements.sql` | CFTC positioning features |
| `database/macros/master_feature_matrix.sql` | Combines all above |

## Key Benefits of Anofox-First Approach

1. **Speed**: SQL feature calculations are 100x faster than Pandas
2. **Consistency**: Same SQL functions for historical backtest AND live production
3. **Scalability**: DuckDB handles millions of rows efficiently
4. **Maintainability**: Single source of truth for all features
5. **TSci Optimization**: TSci works best with rich, pre-engineered features

## Summary

**TSci is NOT a feature engineering tool - it's a model selection and ensemble optimization tool.**

Your 276 features must be created by Anofox SQL functions BEFORE TSci runs. This is actually better because:
- Anofox is optimized for feature generation (C++ backend)
- TSci is optimized for model selection (LLM reasoning)
- Separation of concerns = cleaner architecture

**Next Step**: Install Anofox extensions and create your feature calculation SQL script.
