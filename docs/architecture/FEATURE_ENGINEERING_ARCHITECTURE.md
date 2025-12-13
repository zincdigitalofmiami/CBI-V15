# Feature Engineering Architecture - Complete System

## 🏗️ System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  RAW DATA LAYER (DuckDB/MotherDuck)                             │
│  • raw.databento_ohlcv_daily (30+ symbols, 2000-present)       │
│  • raw.fred_daily (60+ macro series)                            │
│  • raw.eia_biofuels (biodiesel, RIN prices)                     │
│  • raw.scrapecreators_trump_posts (sentiment)                   │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  SQL MACROS (24 Reusable Functions)                             │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ TECHNICAL INDICATORS (8 macros)                           │ │
│  │ • calc_rsi(sym, period)                                   │ │
│  │ • calc_macd(sym, fast, slow, signal)                      │ │
│  │ • calc_bollinger(sym, period, num_std)                    │ │
│  │ • calc_atr(sym, period)                                   │ │
│  │ • calc_stochastic(sym, period, smooth)                    │ │
│  │ • calc_momentum(sym)                                      │ │
│  │ • calc_volume_indicators(sym)                             │ │
│  │ • calc_all_technical_indicators(sym) → 40 features        │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ CROSS-ASSET FEATURES (5 macros)                           │ │
│  │ • calc_pairwise_correlation(sym1, sym2, window)           │ │
│  │ • calc_rolling_beta(sym, benchmark, window)               │ │
│  │ • calc_fundamental_spreads() → 6 spreads                  │ │
│  │ • calc_calendar_spreads(near, far)                        │ │
│  │ • calc_correlation_matrix(window) → 11 correlations       │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ BIG 8 BUCKET SCORES (9 macros)                            │ │
│  │ • calc_crush_bucket_score()                               │ │
│  │ • calc_china_bucket_score()                               │ │
│  │ • calc_fx_bucket_score()                                  │ │
│  │ • calc_fed_bucket_score()                                 │ │
│  │ • calc_tariff_bucket_score()                              │ │
│  │ • calc_biofuel_bucket_score()                             │ │
│  │ • calc_energy_bucket_score()                              │ │
│  │ • calc_volatility_bucket_score()                          │ │
│  │ • calc_all_bucket_scores() → 16 features                  │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ MASTER BUILDERS (2 macros)                                │ │
│  │ • build_symbol_features(sym) → 93 features                │ │
│  │ • build_all_symbols_features() → 17 symbols               │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  FEATURE TABLES (5 Tables)                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ features.technical_indicators_all_symbols                 │ │
│  │ • 17 symbols × ~6,000 days = ~100K rows                   │ │
│  │ • 40 columns (RSI, MACD, BB, ATR, Stoch, etc.)            │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ features.cross_asset_correlations                         │ │
│  │ • ~6,000 rows (daily)                                     │ │
│  │ • 11 columns (ZL-ZS, ZL-CL, CL-HO, etc.)                  │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ features.fundamental_spreads                              │ │
│  │ • ~6,000 rows (daily)                                     │ │
│  │ • 6 columns (board_crush, BOHO, crack, etc.)              │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ features.big8_bucket_scores                               │ │
│  │ • ~6,000 rows (daily)                                     │ │
│  │ • 16 columns (8 scores + 8 metrics)                       │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ features.daily_ml_matrix_zl (MASTER TABLE)                │ │
│  │ • ~6,000 rows (ZL daily, 2000-present)                    │ │
│  │ • 93 columns (all features combined)                      │ │
│  │   - 40 technical indicators                               │ │
│  │   - 11 cross-asset correlations                           │ │
│  │   - 6 fundamental spreads                                 │ │
│  │   - 16 Big 8 bucket features                              │ │
│  │   - 9 neural scores (placeholders)                        │ │
│  │   - 8 targets                                             │ │
│  │   - 3 metadata (date, symbol, regime)                     │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  PYTHON BUILD SCRIPT                                            │
│  src/engines/anofox/build_all_features.py                       │
│  • Loads all SQL macros                                         │
│  • Executes feature computation                                 │
│  • Populates all 5 feature tables                               │
│  • Runtime: 2-5 minutes on Mac M4                               │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  READY FOR ML TRAINING                                          │
│  • AutoGluon TabularPredictor (Big 8 bucket specialists)        │
│  • AutoGluon TimeSeriesPredictor (core ZL forecaster)           │
│  • AutoGluon stacking + WeightedEnsemble_L2 + Monte Carlo       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Feature Count Breakdown

| Category | Features | Source |
|----------|----------|--------|
| **Technical Indicators** | 40 | `calc_all_technical_indicators()` |
| **Cross-Asset Correlations** | 11 | `calc_correlation_matrix()` |
| **Fundamental Spreads** | 6 | `calc_fundamental_spreads()` |
| **Big 8 Bucket Scores** | 8 | `calc_all_bucket_scores()` |
| **Big 8 Key Metrics** | 8 | `calc_all_bucket_scores()` |
| **Neural Scores** | 9 | Populated by ML models |
| **Targets** | 8 | `feat_targets_block()` |
| **Metadata** | 3 | as_of_date, symbol, regime |
| **TOTAL** | **93** | Per symbol |

---

## 🔄 Daily Update Workflow (Aligned with V15.1)

```bash
# 1. Ingest raw data (Databento, FRED, EIA, ScrapeCreators)
python trigger/DataBento/Scripts/collect_daily.py
python trigger/FRED/Scripts/collect_fred_rates_curve.py
python trigger/EIA_EPA/Scripts/collect_eia_biofuels.py
python trigger/ScrapeCreators/Scripts/collect_news_buckets.py

# 2. Build features (SQL-first via AnoFox)
python src/engines/anofox/build_all_features.py

# 3. Train models (AutoGluon stack – see MASTER_PLAN)
#    Big 8 buckets → AutoGluon TabularPredictor
#    Core ZL       → AutoGluon TimeSeriesPredictor

# 4. Generate forecasts & risk metrics
#    Upload forecasts to MotherDuck (forecasts.*)
#    Run Monte Carlo on final forecasts only
```

---

## 🎯 Key Design Principles

### 1. **100% SQL-Native**
- All feature engineering in DuckDB SQL
- No pandas/numpy dependencies for features
- Portable across Mac/Linux/Windows

### 2. **Lag-Safe**
- No look-ahead bias
- All features use LAG/LEAD properly
- Targets use LEAD (future values)

### 3. **Reusable Macros**
- Parameterized functions
- Apply to any symbol
- Easy to extend

### 4. **Incremental Updates**
- INSERT OR REPLACE pattern
- Only compute new dates
- Fast daily updates

### 5. **Mac-Native**
- No cloud compute required
- Runs on Mac M4
- MotherDuck for storage only

---

## ✅ What's Complete

- [x] 24 SQL macros for feature engineering
- [x] 5 feature tables with proper schemas
- [x] Python build script
- [x] Documentation
- [x] 40 technical indicators per symbol
- [x] 11 cross-asset correlations
- [x] 6 fundamental spreads
- [x] 8 Big 8 bucket scores
- [x] 8 targets (1W/1M/3M/6M)

---

## 🚧 Next Steps (Conceptual)

1. **Train AutoGluon Models**
   - 8 Big 8 bucket specialists (TabularPredictor, quantile mode)
   - Core ZL forecaster (TimeSeriesPredictor, quantile mode)

2. **Add Sentiment Scores**
   - FinBERT on Mac MPS
   - Process ScrapeCreators news

3. **Add Weather Features**
   - Brazil/Argentina/US rainfall
   - Drought indices

4. **Add CFTC Positioning**
   - Net non-commercial positions
   - Open interest

5. **Expand to All 30 Symbols**
   - Currently: 17 symbols
   - Target: 30+ symbols

---

## 📝 Usage Examples

See `database/macros/README_TECHNICAL_INDICATORS.md` for detailed usage examples.
