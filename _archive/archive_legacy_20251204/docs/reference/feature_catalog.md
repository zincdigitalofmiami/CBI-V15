# Feature Catalog

**Status:** Production  
**Last Updated:** December 3, 2025

## Production Features (80-120)

Features migrated from CBI-V15 and validated via AnoFox feature importance analysis.

## Feature Categories

### Price Features
- `databento_zl_close` - ZL settlement price
- `databento_zl_volume` - Trading volume
- `databento_zl_open_interest` - Open interest

### Technical Indicators
- `zl_sma_5`, `zl_sma_20`, `zl_sma_50` - Simple moving averages
- `zl_rsi_14` - Relative Strength Index
- `zl_volatility_21d` - 21-day annualized volatility
- `zl_trend_strength_60d` - 60-day trend strength

### Macro Features
- `fred_dxy` - Dollar Index
- `fred_vix` - VIX volatility
- `fred_treasury_10y` - 10-year yield
- `fred_fed_funds` - Fed Funds Rate

### Commodity Correlations
- `corr_zl_wti_90d` - ZL/WTI 90-day correlation
- `corr_zl_brl_90d` - ZL/BRL correlation
- `palm_oil_spread` - ZL - Palm Oil price differential

### Weather Features
- `weather_us_iowa_prcp_mm` - Iowa precipitation
- `weather_br_mato_grosso_tavg_c` - Brazil Mato Grosso temperature
- `weather_argentina_drought_zscore` - Argentina drought Z-score

### Policy Features
- `policy_trump_score` - Trump policy impact score
- `eia_rin_price_d4` - D4 RIN price
- `usda_wasde_world_soyoil_prod` - USDA world soy oil production

### Positioning Features
- `cftc_managed_money_netlong` - CFTC managed money net long
- `cftc_producer_merchant_short` - Producer/merchant short positions

## Feature Selection Criteria

- Correlation with ZL returns > 0.1 OR
- Domain importance (per quant team) OR
- AnoFox feature importance rank ≤ 100

## Pruned Features

Originally 276 features. Pruned to 80-120 based on:
- Redundancy (collinearity > 0.9)
- Low correlation (< 0.05)
- Data quality issues

Pruning rationale documented in `docs/archive/feature_pruning_log.md`

