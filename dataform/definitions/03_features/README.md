# Feature Engineering Layer

This directory contains all feature engineering transformations for the ZL forecasting system.

## Overview

Features are organized by domain:
- **Technical Indicators**: RSI, MACD, Bollinger, ATR, moving averages
- **Big 8 Drivers**: Crush margin, China imports, Dollar, Fed, Tariffs, Biofuels, Crude, VIX
- **Cross-Asset**: Correlations, betas, spreads
- **Weather**: Anomalies, buckets, ENSO regimes
- **Political**: FEC signals, Trump policy, silence detection
- **Positioning**: CFTC managed money, crowding signals

## Key Files

### Big 8 Drivers
- `big_eight_signals.sqlx` - Master Big 8 aggregation
- `crush_margin_daily.sqlx` - ZS + ZM - ZL processing margin
- `biodiesel_margin_daily.sqlx` - Biofuel margin + RIN prices

### Cross-Asset Features
- `cross_asset_correlations.sqlx` - Rolling correlations (ZL-FCPO, ZL-HO, ZL-DXY, ZL-CL)
- `palm_features_daily.sqlx` - Palm oil spread, ratio, correlations

### Technical Indicators
- `technical_indicators.sqlx` - RSI, MACD, Bollinger, ATR, MAs
- Note: Advanced indicators (pandas-ta) computed in Python, stored here

### Political Intelligence
- `fec_policy_signals.sqlx` - FEC political intelligence features
- `silence_signals.sqlx` - Tweet gap detection
- `trump_policy_signals.sqlx` - Trump policy extraction
- `zl_trump_impact.sqlx` - ZL impact from Trump actions

### Weather Features
- `weather_anomalies_daily.sqlx` - Weather z-scores, anomalies, GDD deviations

### Positioning Features
- `cftc_positions_daily.sqlx` - Managed money positioning, crowding signals

### Master Table
- `daily_ml_matrix.sqlx` - Master feature table with nested STRUCTs
  - Flattened view: `vw_daily_ml_flat` (for Mac training exports)

## Usage

Features are automatically built by Dataform DAG. To rebuild:

```bash
cd dataform
dataform run --tags features
```

## Dependencies

- `02_staging/market_daily.sqlx`
- `02_staging/fred_macro_clean.sqlx`
- `02_staging/weather_regions_aggregated.sqlx`
- `02_staging/cftc_positions.sqlx`
- `02_staging/news_bucketed.sqlx`

## Data Quality

All features are validated by assertions in `05_assertions/`:
- `assert_big_eight_complete.sqlx` - Ensures all Big 8 drivers present
- `assert_feature_collinearity.sqlx` - Fails if correlation > 0.85

## References

- [Big 8 Drivers Documentation](docs/features/BIG_EIGHT_DRIVERS.md)
- [Feature Calculations](docs/features/CALCULATIONS.md)
- [Technical Indicators](docs/features/TECHNICAL_INDICATORS.md)

---

**Last Updated**: November 28, 2025

